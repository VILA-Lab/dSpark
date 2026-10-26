import argparse
import bisect
import html
import time
from functools import lru_cache

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from generate import EOS_ID, MASK_ID, generate_multi_block

EOT_ID = 126348
GEN_LENGTH = 256
EXAMPLE = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many "
           "clips did Natalia sell altogether in April and May? Please reason step by step, and put your final answer "
           "within \\boxed{}.")
CSS = """
.tokens {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.65;
         white-space: pre-wrap; word-break: break-word; min-height: 530px}
.mask {display: inline-block; margin: 0 2px; padding: 0 4px; border-radius: 4px; font-size: 9px; line-height: 15px;
       color: var(--block-label-text-color); background: var(--background-fill-secondary);
       border: 1px solid var(--border-color-primary)}
.new {background: var(--primary-100); border-radius: 3px}
.dark .new {background: var(--primary-800)}
.stats {margin-top: 12px; padding-top: 8px; border-top: 1px solid var(--border-color-primary);
        color: var(--body-text-color-subdued); font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px}
"""


@torch.no_grad()
def generate_standard(model, prompt, gen_length=GEN_LENGTH, block_length=32, mask_id=MASK_ID):
    """Standard decoding: one token per forward, the most confident masked token of the current block."""
    P = prompt.shape[1]
    x = torch.full((1, P + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :P] = prompt
    nfe = 0
    for start in range(P, P + gen_length, block_length):
        end = start + block_length
        while (x[0, start:end] == mask_id).any():
            logits = model(x).logits[0, start:end]
            nfe += 1
            conf, x0 = F.softmax(logits.to(torch.float64), dim=-1).max(dim=-1)
            conf = torch.where(x[0, start:end] == mask_id, conf, -torch.inf)
            j = int(conf.argmax())
            x[0, start + j] = x0[j]
    return x, nfe


def record(model, decode, prompt):
    """Decode the prompt, keeping the sequence seen by every forward and the time it started."""
    run = {"states": [], "times": []}

    def hook(module, args, kwargs):
        run["states"].append((args[0] if args else kwargs["input_ids"])[0].tolist())
        run["times"].append(time.perf_counter() - start)

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    start = time.perf_counter()
    try:
        x, run["nfe"] = decode(model, prompt.to(model.device))
        run["final"] = x[0].tolist()
    finally:
        handle.remove()
    run["end"] = time.perf_counter() - start
    return run


@lru_cache(maxsize=None)
def piece(token_id):
    return html.escape(tokenizer.decode([token_id], skip_special_tokens=True))


def panel(response, previous, stats, done=False):
    """Decoded tokens as text and MASK for the others; tokens that were masked in `previous` are highlighted."""
    parts = []
    for t, before in zip(response, previous):
        if t == MASK_ID:
            parts.append('<span class="mask">MASK</span>')
        elif t not in (EOS_ID, EOT_ID):
            parts.append(f'<span class="new">{piece(t)}</span>' if before == MASK_ID else piece(t))
    return f'<div class="tokens">{"".join(parts)}</div><div class="stats{" done" if done else ""}">{stats}</div>'


def replay(run, t, prompt_len):
    """The panel of a recorded run t seconds after it started."""
    done = t >= run["end"]
    if done:
        k, response = run["nfe"], run["final"][prompt_len:]
        previous = response
    else:
        k = max(bisect.bisect_right(run["times"], t) - 1, 0)
        response = run["states"][k][prompt_len:]
        previous = run["states"][k - 1][prompt_len:] if k > 0 else response
    n_tokens = sum(1 for tok in response if tok not in (MASK_ID, EOS_ID, EOT_ID))
    elapsed = min(t, run["end"])
    stats = (f"Tokens: {n_tokens} | Time: {elapsed:.2f}s | TPS: {n_tokens / max(elapsed, 1e-6):.0f} | NFE: {k} | "
             f"TPF: {n_tokens / max(k, 1):.2f}")
    return panel(response, previous, stats, done)


def race(prompt):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                         tokenize=False)
    ids = torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"])[None]
    blank = [MASK_ID] * GEN_LENGTH
    # decode with each model in turn, then replay both runs side by side at their real speed
    yield [panel(blank, blank, "Decoding..."), panel(blank, blank, "Waiting...")]
    runs = [record(ours, lambda m, p: generate_multi_block(m, p, GEN_LENGTH, factor=args.factor), ids)]
    yield [panel(blank, blank, "Waiting..."), panel(blank, blank, "Decoding...")]
    runs.append(record(base, generate_standard, ids))
    start = time.perf_counter()
    while True:
        t = time.perf_counter() - start
        yield [replay(run, t, ids.shape[1]) for run in runs]
        if t >= max(run["end"] for run in runs):
            break
        time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="output_model/dspark_llada")
    parser.add_argument("--base_model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--factor", type=float, default=1.5)
    args = parser.parse_args()

    # dSpark on the first GPU and the base model on the last one (the same GPU if there is only one)
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ours = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    ours = ours.to(devices[0]).eval()
    base = AutoModel.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    base = base.to(devices[-1]).eval()
    warmup = torch.tensor(tokenizer("Hello")["input_ids"])[None]
    generate_multi_block(ours, warmup.to(ours.device), gen_length=32)
    generate_standard(base, warmup.to(base.device), gen_length=32)

    with gr.Blocks(title="dSpark", fill_width=True) as demo:
        prompt = gr.Textbox(EXAMPLE, label="Prompt", lines=2, submit_btn=True)
        with gr.Row(equal_height=True):
            left = gr.HTML(label="dSpark-LLaDA", show_label=True, container=True, padding=True)
            right = gr.HTML(label="LLaDA-8B-Instruct", show_label=True, container=True, padding=True)
        prompt.submit(race, prompt, [left, right], show_progress="hidden")
    demo.launch(theme=gr.themes.Default(primary_hue="teal"), css=CSS)
