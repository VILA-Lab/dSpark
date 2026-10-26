import contextlib
import glob
import json
import os
import random
import sys
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed

MASK_ID = 151666
EOS_ID = 151643


def shift_logits(logits):
    """Dream's logits at position i - 1 predict token i; shift them onto the positions they predict"""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loading a YAML Configuration File"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_config(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Override config values from command line args like 'training.learning_rate=1e-5'"""
    for override in overrides:
        # Skip DeepSpeed/distributed training args (--local_rank, etc.)
        if override.startswith("--") or "=" not in override:
            continue
        key_path, value = override.split("=", 1)
        *parents, key = key_path.split(".")
        target = config
        for p in parents:
            target = target.setdefault(p, {})
        parsed = yaml.safe_load(value)
        # YAML reads no/yes/on/off as booleans, keep them as strings
        if isinstance(parsed, bool) and value.lower() not in ("true", "false"):
            parsed = value
        target[key] = parsed
    return config


def get_deepspeed_config() -> Dict[str, Any]:
    """Creating a DeepSpeed ZeRO-2 Configuration"""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }


def prepare_model(config: Dict[str, Any]):
    """Prepare the model and tokenizer with LoRA"""
    model = AutoModel.from_pretrained(config["model"]["name"], torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], trust_remote_code=True)
    lora = config["lora"]
    lora_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def load_data(path, gen_length=256, max_length=512, neg_ratio=0.25, seed=0):
    """Load correct responses (with their oracle schedule) and failed trajectories from the data shards.
    Failed trajectories are sampled at neg_ratio per correct response (a negative ratio keeps all of them)."""
    correct, failed = [], []
    for file in sorted(glob.glob(path)):
        with open(file) as f:
            for line in f:
                r = json.loads(line)
                if len(r["prompt_ids"]) + gen_length > max_length:
                    continue
                if r["is_correct"]:
                    correct.append(r)
                if "failed_response" in r:
                    failed.append(r)
    if neg_ratio >= 0:
        failed = random.Random(seed).sample(failed, min(len(failed), round(neg_ratio * len(correct))))
    print(f"Loaded {len(correct)} correct responses and {len(failed)} failed trajectories from {path}")
    return correct, failed


class DSparkDataset(Dataset):
    """Training states, drawn again at every access.

    A correct response gives a state of its oracle schedule: a random block and round, with the positions of the
    block committed before that round revealed and everything after the block masked. The loss covers the masked
    positions of the block and, with pipelined_window from the second round on, those of the next block.
    A failed trajectory gives the base model's own state after a random round; all its masked positions are
    anchored to the base model.
    """

    def __init__(self, correct, failed, block_length=32, pipelined_window=True):
        self.block_length = block_length
        self.pipelined_window = pipelined_window
        self.correct = []
        for r in correct:
            blocks = sorted(set(r["block"]) - {-1})
            if blocks:
                y = torch.tensor(r["prompt_ids"] + r["response"])
                block, round_ = torch.tensor(r["block"]), torch.tensor(r["round"])
                self.correct.append((len(r["prompt_ids"]), y, block, round_, blocks))
        self.failed = []
        for r in failed:
            committed = [j for j, t in enumerate(r["failed_round"]) if t >= 0]
            if committed:
                y = torch.tensor(r["prompt_ids"] + r["failed_response"])
                rounds = sorted({r["failed_round"][j] for j in committed})
                end = (max(committed) // block_length + 1) * block_length
                self.failed.append((len(r["prompt_ids"]), y, torch.tensor(r["failed_round"]), rounds, end))

    def __len__(self):
        return len(self.correct) + len(self.failed)

    def __getitem__(self, i):
        if i < len(self.correct):
            return self.correct_state(*self.correct[i])
        return self.failed_state(*self.failed[i - len(self.correct)])

    def correct_state(self, P, y, block, round_, blocks):
        b = random.choice(blocks)
        in_block = block == b
        r = random.randrange(1 + int(round_[in_block].max()))
        lo, hi = P + b * self.block_length, P + (b + 1) * self.block_length
        x = y.clone()
        x[hi:] = MASK_ID
        x[P:][in_block & (round_ >= r)] = MASK_ID
        loss_mask = torch.zeros_like(x, dtype=torch.bool)
        loss_mask[lo:hi] = x[lo:hi] == MASK_ID
        # the pipelined decoder already predicts the next block once the current one is partly decoded
        if self.pipelined_window and r >= 1:
            for b2 in [k for k in blocks if k > b][:1]:
                lo2, hi2 = P + b2 * self.block_length, P + (b2 + 1) * self.block_length
                loss_mask[lo2:hi2] = x[lo2:hi2] == MASK_ID
        return {"input_ids": x, "labels": y, "loss_mask": loss_mask, "kl_mask": torch.zeros_like(loss_mask)}

    def failed_state(self, P, y, round_, rounds, end):
        r = random.choice([0] + rounds[:-1])  # after the last round nothing is masked
        x = y.clone()
        x[P:][(round_ < 0) | (round_ > r)] = MASK_ID
        kl_mask = torch.zeros_like(x, dtype=torch.bool)
        kl_mask[P:P + end] = x[P:P + end] == MASK_ID
        return {"input_ids": x, "labels": y, "loss_mask": torch.zeros_like(kl_mask), "kl_mask": kl_mask}


class DataCollator:
    """Right padding to max_length (tokens with EOS, masks with False)"""

    def __init__(self, max_length=512):
        self.max_length = max_length

    def __call__(self, features):
        batch = {}
        for key in ("input_ids", "labels", "loss_mask", "kl_mask"):
            pad = EOS_ID if key in ("input_ids", "labels") else False
            rows = []
            for f in features:
                v = f[key][: self.max_length]
                rows.append(torch.cat([v, torch.full((self.max_length - len(v),), pad, dtype=v.dtype)]))
            batch[key] = torch.stack(rows)
        return batch


class DSparkTrainer(Trainer):
    """Schedule distillation: CE + entropy sharpening on oracle states, KL to the base model on failed states"""

    def __init__(self, lambda_ent=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ent = lambda_ent
        self.beta = beta
        self.log_sums = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits = shift_logits(model(input_ids=inputs["input_ids"]).logits)
        loss = 0.0 * logits.sum()
        loss_mask, kl_mask = inputs["loss_mask"], inputs["kl_mask"]
        stats = {}

        # Cross-entropy on the masked positions + entropy of the positions already predicted correctly
        if loss_mask.any():
            rows = logits[loss_mask].float()
            labels = inputs["labels"][loss_mask]
            ce = F.cross_entropy(rows, labels)
            logp = F.log_softmax(rows, dim=-1)
            correct = logp.argmax(dim=-1) == labels
            sharp = 0.0 * ce
            if correct.any():
                lp = logp[correct]
                sharp = -(lp.exp() * lp).sum(dim=-1).mean()
            loss = loss + ce + self.lambda_ent * sharp
            stats.update(ce=ce.item(), sharp=sharp.item(), acc=correct.float().mean().item())

        # KL(base || student) on the masked positions of failed trajectories; the base is the model without LoRA
        if self.beta > 0 and kl_mask.any():
            base = model
            while hasattr(base, "module"):
                base = base.module
            frozen = base.disable_adapter() if hasattr(base, "disable_adapter") else contextlib.nullcontext()
            with torch.no_grad(), frozen:
                teacher = shift_logits(model(input_ids=inputs["input_ids"]).logits)
            kl = F.kl_div(F.log_softmax(logits[kl_mask].float(), dim=-1),
                          F.log_softmax(teacher[kl_mask].float(), dim=-1), log_target=True, reduction="batchmean")
            loss = loss + self.beta * kl
            stats["kl"] = kl.item()

        for k, v in stats.items():
            s = self.log_sums.setdefault(k, [0.0, 0])
            s[0] += v
            s[1] += 1
        return (loss, logits) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        for k, (s, n) in self.log_sums.items():
            logs[f"train/{k}"] = round(s / n, 4)
        self.log_sums = {}
        super().log(logs, *args, **kwargs)


def main():
    # 1. Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "config_lora_dream.yaml")
    config = load_config(config_path)
    config = override_config(config, sys.argv[1:])
    print(json.dumps(config, indent=2))
    set_seed(config["training"].get("seed", 42))

    # 2. Model and data
    model, tokenizer = prepare_model(config)
    data = config["data"]
    correct, failed = load_data(data["path"], data["gen_length"], data["max_length"], data["neg_ratio"])
    dataset = DSparkDataset(correct, failed, data["block_length"], data["pipelined_window"])

    # 3. Train
    training_args = TrainingArguments(
        **config["training"],
        deepspeed=get_deepspeed_config(),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        label_names=["input_ids", "labels", "loss_mask", "kl_mask"],
    )
    trainer = DSparkTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollator(data["max_length"]),
        lambda_ent=config["loss"]["lambda_ent"],
        beta=config["loss"]["beta"],
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
