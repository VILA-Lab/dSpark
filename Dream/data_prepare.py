import argparse
import ast
import glob
import itertools
import json
import os
import random
import re
import subprocess
import sys
import tempfile

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MASK_ID = 151666
EOS_ID = 151643
H_MAX = 12.0  # cap on the surprisal -log p(y_j) used by the oracle
CODE_SUFFIX = "\n\nWrite a complete Python solution. Put your code in a ```python code block."


def get_logits(model, x):
    """Full-attention forward; Dream's logits at position i - 1 predict token i, so shift them by one"""
    logits = model(x, "full", None).logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def encode_prompt(tokenizer, task, question):
    # math problems use the zero-shot chain-of-thought format of lm-eval's gsm8k_cot_zeroshot
    content = question + CODE_SUFFIX if task == "code" else f"Q: {question}\nA: Let's think step by step."
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def decode(tokenizer, ids):
    vocab_size = len(tokenizer)
    return tokenizer.decode([t for t in ids if t < vocab_size], skip_special_tokens=True)


# ---------------------------------------------------------------------------------------------- answer checking
def extract_boxed_answer(text):
    """Content of the last \\boxed{} in text"""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


REPLACEMENTS = [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""), ("\\ ", " "),
                ("\\$", ""), ("$", ""), ("\\%", ""), ("%", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                ("^{\\circ}", ""), ("^\\circ", ""), ("\\cdot", "*"), ("\\times", "*"), ("dollars", ""),
                ("units", ""), ("\\text{ }", " ")]


def normalize_answer(s):
    """Normalize a math answer for comparison"""
    if s is None:
        return None
    s = re.sub(r"\\\\", r"\\", s.strip())
    for a, b in REPLACEMENTS:
        s = s.replace(a, b)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\s*(\d|[a-zA-Z])\s*(\d|[a-zA-Z])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]+)\}(\d|[a-zA-Z])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d|[a-zA-Z])\{([^{}]+)\}", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt\s*(\d|[a-zA-Z])", r"\\sqrt{\1}", s)
    s = re.sub(r"\s+", "", s).rstrip(".")
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    m = re.fullmatch(r"(-?)(\d+)/(\d+)", s)
    if m:
        s = f"{m.group(1)}\\frac{{{m.group(2)}}}{{{m.group(3)}}}"
    return re.sub(r"(?<![\d.])0\.(\d)", r".\1", s)


def to_number(text):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", "").replace("$", "").replace("%", ""))
    return float(numbers[-1]) if numbers else None


def numeric_value(s):
    try:
        m = re.fullmatch(r"(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
        if m:
            v = float(m.group(2)) / float(m.group(3))
            return -v if m.group(1) == "-" else v
        return float(s.replace(",", ""))
    except Exception:
        return None


def math_equal(pred, gold):
    a, b = normalize_answer(pred), normalize_answer(gold)
    if a is None or b is None:
        return False
    if a == b:
        return True
    na, nb = numeric_value(a), numeric_value(b)
    return na is not None and nb is not None and abs(na - nb) < 1e-6


def passes_tests(response, tests, timeout=10):
    """Run the definitions of the first ```python block of the response together with the tests"""
    code = response.split("```python", 1)[1].split("```")[0] if "```python" in response else response
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    # keep imports and definitions, drop top-level example calls
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign)
    body = [n for n in tree.body if isinstance(n, keep)]
    program = ast.unparse(ast.Module(body=body, type_ignores=[])) + "\n\n" + tests + "\n\ntest_acecode()\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
    try:
        result = subprocess.run([sys.executable, f.name], capture_output=True, timeout=timeout,
                                stdin=subprocess.DEVNULL, cwd=tempfile.gettempdir())
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(f.name)


def is_correct(task, response, answer):
    if task == "code":
        return passes_tests(response, answer)
    if task == "gsm8k":
        boxed = re.findall(r"\\boxed\{([^}]*)\}", response)
        pred = to_number(boxed[-1]) if boxed else None
        if pred is None:
            pred = to_number(response)
        gold = to_number(answer)
        return pred is not None and gold is not None and abs(pred - gold) < 1e-4
    pred = extract_boxed_answer(response)
    if not pred:
        # "The answer is ..." when there is no \boxed{}
        pattern = r"[Tt]he\s+(?:final\s+)?answer(?:\s+is)?\s*:?\s*\$?(.+?)\$?\s*\.?\s*$"
        lines = re.findall(pattern, response.strip(), re.M)
        pred = lines[-1].strip().rstrip(".") if lines else None
    return math_equal(pred, answer)


# ---------------------------------------------------------------------------------------------- problems
def load_problems(tokenizer, num_numina=30000, num_omi=110000, num_acecode=10000, max_prompt_len=250, seed=0):
    """Training problems as (task, question, answer) from GSM8K, MATH, NuminaMath-CoT, OpenMathInstruct-2 and
    AceCode-87K.

    NuminaMath and OpenMathInstruct-2 problems that also appear in GSM8K or MATH are dropped and only short
    answers are kept; AceCode asserts become one test function. Prompts longer than max_prompt_len tokens are
    skipped so that prompt + response fit the training length.
    """
    def fits(task, question):
        return len(encode_prompt(tokenizer, task, question)) <= max_prompt_len

    gsm8k_splits = {split: load_dataset("openai/gsm8k", "main", split=split) for split in ("train", "test")}
    math_splits = {split: load_dataset("nlile/hendrycks-MATH-benchmark", split=split) for split in ("train", "test")}
    gsm8k, math = gsm8k_splits["train"], math_splits["train"]
    problems = [("gsm8k", q, a.split("####")[-1].strip()) for q, a in zip(gsm8k["question"], gsm8k["answer"])]
    problems += [("math", q, a) for q, a in zip(math["problem"], math["answer"])]
    problems = [p for p in problems if fits(p[0], p[1])]

    # NuminaMath-CoT, sampled round-robin over its sources
    seen = {q.strip() for d in gsm8k_splits.values() for q in d["question"]}
    seen |= {q.strip() for d in math_splits.values() for q in d["problem"]}
    by_source = {}
    numina = load_dataset("AI-MO/NuminaMath-CoT", split="train").select_columns(["source", "problem", "solution"])
    for batch in numina.iter(batch_size=10000):
        for source, q, solution in zip(batch["source"], batch["problem"], batch["solution"]):
            q, answer = q.strip(), extract_boxed_answer(solution)
            if source in ("gsm8k", "math") or q in seen:
                continue
            if not answer or len(answer) > 60 or not normalize_answer(answer):
                continue
            seen.add(q)
            by_source.setdefault(source, []).append(("math", q, answer))
    rng = random.Random(seed)
    for queue in by_source.values():
        rng.shuffle(queue)
    queues = sorted(by_source.values(), key=len, reverse=True)
    picked = []
    for k in range(max(map(len, queues))):
        picked += [queue[k] for queue in queues if k < len(queue)]
        if len(picked) >= 2 * num_numina:
            break
    problems += list(itertools.islice((p for p in picked if fits(p[0], p[1])), num_numina))

    # OpenMathInstruct-2: synthesized problems of the 1M subset (27% from GSM8K, 73% from MATH)
    root = snapshot_download("nvidia/OpenMathInstruct-2", repo_type="dataset",
                             allow_patterns=["data/train_1M-*.parquet"])
    omi = pd.concat([pd.read_parquet(f, columns=["problem", "expected_answer", "problem_source"])
                     for f in sorted(glob.glob(f"{root}/data/train_1M-*.parquet"))]).drop_duplicates("problem")
    for source, weight in (("augmented_gsm8k", 0.27), ("augmented_math", 0.73)):
        target, kept = round(num_omi * weight), []
        subset = omi[omi.problem_source == source].sample(frac=1.0, random_state=seed)
        for q, answer in zip(subset.problem, subset.expected_answer):
            q, answer = str(q).strip(), str(answer).strip()
            if q in seen or not answer or len(answer) > 60 or not normalize_answer(answer) or not fits("math", q):
                continue
            kept.append(("math", q, answer))
            if len(kept) == target:
                break
        problems += kept

    # AceCode-87K: problems with tests that a reference solution passes
    acecode = load_dataset("TIGER-Lab/AceCode-87K", split="train")
    order = list(range(len(acecode)))
    random.Random(seed).shuffle(order)
    num_code = 0
    for i in order:
        row = acecode[i]
        q, tests = (row["question"] or "").strip(), row["test_cases"] or []
        if not 40 <= len(q) <= 1500 or not tests:
            continue
        if not any((x.get("pass_rate") or 0) >= 0.999 for x in row["inferences"] or []):
            continue
        test_fn = "def test_acecode():\n" + "\n".join("    " + line for t in tests for line in t.splitlines())
        if len(test_fn) <= 2800 and fits("code", q):
            problems.append(("code", q, test_fn))
            num_code += 1
            if num_code == num_acecode:
                break

    # every question enters once
    unique, taken = [], set()
    for p in problems:
        if p[1].strip() not in taken:
            taken.add(p[1].strip())
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------------------------- decoding
@torch.no_grad()
def generate_with_trajectory(model, prompt_ids, gen_length=256, block_length=32, threshold=0.9, temperature=0.0,
                             generator=None):
    """Semi-autoregressive threshold decoding of the base model.

    In every forward, each masked token of the current block with confidence >= threshold is committed (else the
    lowest-entropy one). With temperature > 0 the tokens are sampled.

    Returns:
        The response (gen_length tokens, EOS-filled after the first EOS) and, per response position, the forward
        (counted from 1) in which it was committed (-1 if never decoded).
    """
    P = len(prompt_ids)
    x = torch.full((1, P + gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[0, :P] = torch.tensor(prompt_ids, device=model.device)
    rounds = [-1] * gen_length
    nfe = 0
    for b in range(gen_length // block_length):
        lo, hi = P + b * block_length, P + (b + 1) * block_length
        steps = 0
        while (x[0, lo:hi] == MASK_ID).any() and steps < block_length + 2:
            steps += 1
            nfe += 1
            logits = get_logits(model, x)[0]
            masked = torch.arange(lo, hi, device=x.device)[x[0, lo:hi] == MASK_ID]
            p = F.softmax(logits[masked].float(), dim=-1)
            if temperature > 0:
                probs = F.softmax(logits[masked].float() / temperature, dim=-1)
                tokens = torch.multinomial(probs, 1, generator=generator)[:, 0]
                conf = p.gather(1, tokens[:, None])[:, 0]
            else:
                conf, tokens = p.max(dim=-1)
            commit = conf >= threshold
            if not commit.any():
                entropy = -(p * torch.log(p + 1e-12)).sum(dim=-1)
                commit[entropy.argmin()] = True
            x[0, masked[commit]] = tokens[commit]
            for q in masked[commit].tolist():
                rounds[q - P] = nfe
        eos = (x[0, P:hi] == EOS_ID).nonzero()
        if eos.numel():
            x[0, P + int(eos[0, 0]) + 1:] = EOS_ID
            break
    return x[0, P:], rounds


# ---------------------------------------------------------------------------------------------- oracle schedule
@torch.no_grad()
def promotion_gains(model, x, y, masked, candidates, batch_size=8):
    """Surprisal reduction at the other masked positions when each candidate is revealed alone.

    The current state is re-evaluated in the first row of every batch, so that the surprisals before and after a
    reveal come from the same forward.
    """
    gains = []
    for s in range(0, len(candidates), batch_size - 1):
        chunk = candidates[s:s + batch_size - 1]
        xb = x[0].repeat(len(chunk) + 1, 1)
        for row, k in enumerate(chunk, start=1):
            xb[row, masked[k]] = y[masked[k]]
        logp = F.log_softmax(get_logits(model, xb)[:, masked].float(), dim=-1)
        h = -logp.gather(2, y[masked][None, :, None].expand(xb.shape[0], -1, 1))[..., 0]
        h = h.clamp(max=H_MAX)
        for row, k in enumerate(chunk, start=1):
            others = torch.ones(len(masked), dtype=torch.bool, device=x.device)
            others[k] = False
            gains.append(float(h[0, others].sum() - h[row, others].sum()))
    return gains


@torch.no_grad()
def oracle_schedule(model, prompt_ids, response, block_length=32, p_min=0.02, max_rank=10, batch_size=8):
    """Uncertainty-guided unmasking order of a correct response, built block by block with the frozen model.

    Each round commits every ready position (the model's argmax is the reference token) and at most one promoted
    candidate: among masked positions with p(y_i) >= p_min and rank(y_i) <= max_rank, the one whose reveal most
    reduces the surprisal of the other masked positions, if that reduction is positive. A round with neither
    commits the most probable reference token.

    Returns:
        Per response position, its block and its commit round within the block (-1 after the EOS block).
    """
    device = model.device
    P = len(prompt_ids)
    y = torch.tensor(list(prompt_ids) + list(response), device=device)
    block_of, round_of = [-1] * len(response), [-1] * len(response)
    for b in range(len(response) // block_length):
        lo, hi = P + b * block_length, P + (b + 1) * block_length
        x = y.clone()[None]
        x[0, lo:] = MASK_ID
        r = 0
        while (x[0, lo:hi] == MASK_ID).any() and r < block_length + 4:
            masked = torch.arange(lo, hi, device=device)[x[0, lo:hi] == MASK_ID]
            logp = F.log_softmax(get_logits(model, x)[0, masked].float(), dim=-1)
            logp_y = logp.gather(1, y[masked][:, None])[:, 0]
            p_y = logp_y.exp()
            rank = (logp > logp_y[:, None]).sum(-1) + 1
            ready = logp.argmax(-1) == y[masked]
            commit = masked[ready].tolist()
            candidates = [k for k in range(len(masked))
                          if not ready[k] and int(rank[k]) <= max_rank and float(p_y[k]) >= p_min]
            if candidates:
                gains = promotion_gains(model, x, y, masked, candidates, batch_size)
                best = max(range(len(candidates)), key=lambda n: gains[n])
                if gains[best] > 0:
                    commit.append(int(masked[candidates[best]]))
            if not commit:
                commit = [int(masked[p_y.argmax()])]
            for q in commit:
                block_of[q - P], round_of[q - P] = b, r
            commit = torch.tensor(commit, device=device)
            x[0, commit] = y[commit]
            r += 1
        if (y[P:hi] == EOS_ID).any():
            break
    return block_of, round_of


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()

    problems = load_problems(tokenizer, args.num_numina, args.num_omi, args.num_acecode, args.max_prompt_len,
                             args.seed)
    per_shard = (len(problems) + args.num_shards - 1) // args.num_shards
    start, end = args.shard_id * per_shard, min((args.shard_id + 1) * per_shard, len(problems))
    print(f"{len(problems)} problems, shard {args.shard_id}: [{start}, {end})")

    # resume: keep the finished records (and drop a partially written last line)
    records = []
    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        with open(args.output_file, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    done = {r["idx"] for r in records}
    num_correct = sum(r["is_correct"] for r in records)

    temperatures = [float(t) for t in args.temperatures.split(",") if t]
    with open(args.output_file, "a") as f:
        for idx in tqdm(range(start, end), desc="dSpark data"):
            if idx in done:
                continue
            task, question, answer = problems[idx]
            prompt_ids = encode_prompt(tokenizer, task, question)
            record = {"idx": idx, "task": task, "prompt_ids": prompt_ids, "is_correct": False}

            response, rounds = generate_with_trajectory(model, prompt_ids, args.gen_length, args.block_length,
                                                        args.threshold)
            if is_correct(task, decode(tokenizer, response.tolist()), answer):
                record.update(is_correct=True, response=response.tolist())
            else:
                # keep the failed trajectory for the KL term, then retry with sampling
                record.update(failed_response=response.tolist(), failed_round=rounds)
                for attempt, temperature in enumerate(temperatures, start=1):
                    generator = torch.Generator(device=device).manual_seed((args.seed * 1000003 + idx) * 64 + attempt)
                    response, _ = generate_with_trajectory(model, prompt_ids, args.gen_length, args.block_length,
                                                           args.threshold, temperature, generator)
                    if is_correct(task, decode(tokenizer, response.tolist()), answer):
                        record.update(is_correct=True, response=response.tolist())
                        break

            # oracle schedule of the correct response
            if record["is_correct"]:
                record["block"], record["round"] = oracle_schedule(model, prompt_ids, record["response"],
                                                                   args.block_length, args.p_min, args.max_rank)
                num_correct += 1
            f.write(json.dumps(record) + "\n")
            f.flush()
            done.add(idx)
            if len(done) % 20 == 0:
                print(f"[idx {idx}] correct: {num_correct}/{len(done)}", flush=True)
    print(f"Saved {len(done)} records to {args.output_file} ({num_correct} correct)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_file", type=str, default="data/shard_0.jsonl")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperatures", type=str, default="0.5,0.6,0.5,0.6,0.7,0.5,0.6",
                        help="one sampled retry per temperature when the greedy response is wrong")
    parser.add_argument("--p_min", type=float, default=0.02)
    parser.add_argument("--max_rank", type=int, default=10)
    parser.add_argument("--num_numina", type=int, default=30000)
    parser.add_argument("--num_omi", type=int, default=110000)
    parser.add_argument("--num_acecode", type=int, default=10000)
    parser.add_argument("--max_prompt_len", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    main(args)
