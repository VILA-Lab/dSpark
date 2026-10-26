import torch
from accelerate import Accelerator
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from generate import EOS_ID, MASK_ID, generate_multi_block


@register_model("dspark_dream")
class DSparkDreamEvalHarness(LM):
    def __init__(
        self,
        model_path="Dream-org/Dream-v0-Instruct-7B",
        gen_length=256,
        block_length=32,
        factor=0.95,
        block_add_threshold=0.1,
        decoded_token_threshold=0.95,
        max_prompt_length=1024,
        batch_size=1,
        **kwargs,
    ):
        """
        Args:
            max_prompt_length: Prompts are truncated to their last max_prompt_length tokens.
        """
        super().__init__()
        assert int(batch_size) == 1, "the multi-block decoder generates one request at a time"
        self.accelerator = Accelerator()
        self._device = self.accelerator.device
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model = self.model.eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.vocab_size = len(self.tokenizer)  # the embedding has padding rows beyond the tokenizer
        self.decode_kwargs = dict(gen_length=int(gen_length), block_length=int(block_length), factor=float(factor),
                                  block_add_threshold=float(block_add_threshold),
                                  decoded_token_threshold=float(decoded_token_threshold))
        self.max_prompt_length = int(max_prompt_length)

    @property
    def rank(self):
        return self.accelerator.process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(chat_history, tokenize=False,
                                                  add_generation_prompt=add_generation_prompt,
                                                  continue_final_message=not add_generation_prompt)

    def generate_until(self, requests, disable_tqdm=False):
        outputs, num_tokens, num_forwards = [], 0, 0
        for req in tqdm(requests, disable=disable_tqdm or self.rank != 0, desc="Generating"):
            context, gen_kwargs = req.args
            prompt = self.tokenizer(context, return_tensors="pt").input_ids[:, -self.max_prompt_length:]
            seq, nfe = generate_multi_block(self.model, prompt.to(self.device), **self.decode_kwargs)

            # cut at the first EOS, then at the stop strings
            answer = seq[0, prompt.shape[1]:]
            eos = (answer == EOS_ID).nonzero()
            num_tokens += int(eos[0, 0]) if eos.numel() else int((answer != MASK_ID).sum())
            num_forwards += nfe
            ids = [t for t in answer.tolist() if t < self.vocab_size]
            text = self.tokenizer.decode(ids).split(self.tokenizer.eos_token)[0]
            for stop in gen_kwargs.get("until") or []:
                text = text.split(stop)[0]
            outputs.append(text)

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
