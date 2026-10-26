import torch
from accelerate import Accelerator
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from generate import EOS_ID, MASK_ID, generate_multi_block


BOXED_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


@register_model("dspark_llada")
class DSparkLLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        gen_length=256,
        block_length=32,
        factor=0.95,
        block_add_threshold=0.5,
        decoded_token_threshold=0.5,
        boxed=False,
        batch_size=1,
        **kwargs,
    ):
        """
        Args:
            boxed: Append the boxed-answer instruction to the prompt (GSM8K, MATH).
        """
        super().__init__()
        assert int(batch_size) == 1, "the multi-block decoder generates one request at a time"
        self.accelerator = Accelerator()
        self._device = self.accelerator.device
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model = self.model.eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.decode_kwargs = dict(gen_length=int(gen_length), block_length=int(block_length), factor=float(factor),
                                  block_add_threshold=float(block_add_threshold),
                                  decoded_token_threshold=float(decoded_token_threshold))
        self.suffix = BOXED_SUFFIX if str(boxed).lower() in ("true", "1") else ""

    @property
    def rank(self):
        return self.accelerator.process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes

    def generate_until(self, requests, disable_tqdm=False):
        outputs, num_tokens, num_forwards = [], 0, 0
        for req in tqdm(requests, disable=disable_tqdm or self.rank != 0, desc="Generating"):
            context, gen_kwargs = req.args
            # every request is one chat turn: the harness context plus the optional instruction
            messages = [{"role": "user", "content": context + self.suffix}]
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt = torch.tensor(self.tokenizer(text, add_special_tokens=False)["input_ids"], device=self.device)[None]
            seq, nfe = generate_multi_block(self.model, prompt, **self.decode_kwargs)

            # cut at the first EOS, then at the stop strings
            answer = seq[0, prompt.shape[1]:]
            eos = (answer == EOS_ID).nonzero()
            answer = answer[: int(eos[0, 0])] if eos.numel() else answer[answer != MASK_ID]
            num_tokens += answer.numel()
            num_forwards += nfe
            text = self.tokenizer.decode(answer, skip_special_tokens=False)
            for stop in gen_kwargs.get("until") or []:
                text = text.split(stop)[0]
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            outputs.append(self.tokenizer.decode(ids, skip_special_tokens=True))

        # tokens per forward over all processes
        totals = torch.tensor([num_tokens, num_forwards, len(requests)], device=self.device)
        if self.world_size > 1:
            totals = self.accelerator.reduce(totals, reduction="sum")
        if self.rank == 0 and totals[1] > 0:
            tokens, forwards, n = totals.tolist()
            print(f"[speed] {n} requests | tokens per forward (TPF) {tokens / forwards:.2f} | "
                  f"{forwards / n:.1f} forwards and {tokens / n:.1f} tokens per request")
        return outputs

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


if __name__ == "__main__":
    cli_evaluate()
