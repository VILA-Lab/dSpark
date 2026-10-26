import torch
import torch.nn.functional as F

MASK_ID = 151666
EOS_ID = 151643


def get_logits(model, x):
    """Full-attention forward; Dream's logits at position i - 1 predict token i, so shift them by one"""
    logits = model(x, "full", None).logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def block_progress(block):
    return 1.0 - block["mask_count"] / block["total_masks"]


def apply_early_stop(x, blocks, prompt_len, eos_id, mask_id):
    """Fill everything after the first decoded EOS with EOS and recount the masks of every block.
    Returns the position of the first EOS, or None."""
    is_eos = x[:, prompt_len:] == eos_id
    if not is_eos.any():
        return None
    pos = torch.arange(is_eos.shape[1], device=x.device).unsqueeze(0)
    first_eos = prompt_len + int(torch.where(is_eos, pos, is_eos.shape[1]).amin(dim=1)[0])
    x[:, first_eos + 1:] = eos_id
    for bid in sorted(blocks):
        if bid == 0:
            continue
        blk = blocks[bid]
        if blk["start"] > first_eos:
            if blk["mask_count"] != 0:
                blk["mask_count"], blk["is_complete"] = 0, True
        elif blk["start"] <= first_eos < blk["end"]:
            remaining = int((x[:, blk["start"]:first_eos + 1] == mask_id).sum())
            if remaining != blk["mask_count"]:
                blk["mask_count"] = remaining
                if remaining == 0:
                    blk["is_complete"] = True
    still_masked = x == mask_id
    for bid in sorted(blocks):
        if bid == 0:
            continue
        blk = blocks[bid]
        actual = int(still_masked[:, blk["start"]:blk["end"]].sum())
        if actual != blk["mask_count"]:
            blk["mask_count"] = actual
            if actual == 0:
                blk["is_complete"] = True
    return first_eos


@torch.no_grad()
def generate_multi_block(
    model,
    prompt,
    gen_length=256,
    block_length=32,
    factor=1.0,
    block_add_threshold=0.1,
    decoded_token_threshold=0.95,
    mask_id=MASK_ID,
    eos_id=EOS_ID,
):
    """Multi-block parallel decoding.

    A new block is opened once the last block is block_add_threshold decoded, and becomes fully active once its
    predecessor is decoded_token_threshold decoded. Decoding stops at the first EOS.

    Args:
        model: Dream model.
        prompt: A tensor of shape (1, L).
    Returns:
        The sequence (prompt + gen_length tokens, EOS-filled after the first EOS) and the number of forwards.
    """
    P = prompt.shape[1]
    total_len = P + gen_length
    x = F.pad(prompt, (0, gen_length), value=mask_id)
    num_blocks = gen_length // block_length
    blocks = {0: {"start": 0, "end": P, "mask_count": 0, "total_masks": P, "is_complete": True}}
    end = min(P + block_length, total_len)
    blocks[1] = {"start": P, "end": end, "mask_count": end - P, "total_masks": end - P,
                 "is_complete": 1.0 >= decoded_token_threshold}
    next_bid = 2
    nfe = 0
    while True:
        # mask index is taken before the EOS fill, as in the reference pipeline
        mask_index = x == mask_id
        if mask_index[:, P:].sum() == 0 and next_bid > num_blocks:
            break
        first_eos = apply_early_stop(x, blocks, P, eos_id, mask_id)
        has_eos = first_eos is not None
        if has_eos:
            # blocks after the EOS are complete
            while next_bid <= num_blocks:
                start = P + (next_bid - 1) * block_length
                if start <= first_eos:
                    break
                end = min(start + block_length, total_len)
                blocks[next_bid] = {"start": start, "end": end, "mask_count": 0, "total_masks": end - start,
                                    "is_complete": True}
                next_bid += 1
            if (x == mask_id)[:, P:].sum() == 0:
                break
        # activate blocks whose predecessor is decoded enough
        for bid in sorted(blocks):
            if bid > 0 and not blocks[bid]["is_complete"]:
                if block_progress(blocks[bid - 1]) >= decoded_token_threshold:
                    blocks[bid]["is_complete"] = True
        # open the next block
        if next_bid <= num_blocks and not has_eos:
            last = blocks[max(blocks)]
            if block_progress(last) >= block_add_threshold or last["mask_count"] == 0:
                start = P + (next_bid - 1) * block_length
                end = min(start + block_length, total_len)
                if end > start:
                    blocks[next_bid] = {"start": start, "end": end,
                                        "mask_count": int((x[:, start:end] == mask_id).sum()),
                                        "total_masks": end - start,
                                        "is_complete": block_progress(blocks[next_bid - 1]) >= decoded_token_threshold}
                    next_bid += 1
        rightmost = max((bid for bid, blk in blocks.items() if blk["is_complete"] or blk["mask_count"] > 0), default=0)
        if rightmost == 0:
            break

        logits = get_logits(model, x)
        nfe += 1
        decode_mask = mask_index.clone()
        decode_mask[:, blocks[rightmost]["end"]:] = False

        # commit the most confident masked tokens
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0 = torch.argmax(logits, dim=-1)
        conf = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(decode_mask, conf, torch.full_like(conf, float("-inf")))
        transfer = torch.zeros_like(decode_mask)
        n = int(decode_mask.sum())
        if n > 0:
            ranked = torch.sort(conf[decode_mask], descending=True)[0]
            k = 0
            for k in range(1, n):
                if float(ranked[k]) < 1.0 - factor / (k + 2):
                    break
            # the most confident token is always committed; as in the reference implementation, stopping at the
            # last rank still commits it
            if k == 0 or k == n - 1:
                k += 1
            transfer[0, torch.topk(conf[0], k=min(k, n)).indices] = True

        # guaranteed progress: the first fully active block with masks commits at least its lowest-entropy token
        first_active = next((bid for bid in sorted(blocks)
                             if bid > 0 and blocks[bid]["is_complete"] and blocks[bid]["mask_count"] > 0), None)
        if first_active is not None:
            s, e = blocks[first_active]["start"], blocks[first_active]["end"]
            if not transfer[:, s:e].any():
                entropy = -(p[:, s:e] * torch.log(p[:, s:e] + 1e-12)).sum(dim=-1)
                entropy = torch.where(decode_mask[:, s:e], entropy, torch.inf)
                transfer[0, s + int(entropy[0].argmin())] = True

        x[transfer] = x0[transfer]
        for bid in sorted(blocks):
            blk = blocks[bid]
            if bid > 0 and blk["mask_count"] > 0:
                done = int(transfer[:, blk["start"]:blk["end"]].sum())
                if done > 0:
                    blk["mask_count"] -= done
        if nfe > 10000:
            break
    return x, nfe
