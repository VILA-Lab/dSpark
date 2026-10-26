import argparse
import glob
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def main(args):
    base_model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    merged_model = PeftModel.from_pretrained(base_model, args.lora_path).merge_and_unload()
    merged_model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)

    # copy the model code and generation config of the base checkpoint (needed for trust_remote_code)
    base_dir = args.base_model if os.path.isdir(args.base_model) else snapshot_download(args.base_model)
    for f in glob.glob(os.path.join(base_dir, "*.py")) + glob.glob(os.path.join(base_dir, "generation_config.json")):
        shutil.copy(f, args.output_path)
    print(f"Merged model saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--lora_path", type=str, default="output_model/dspark_llada_lora")
    parser.add_argument("--output_path", type=str, default="output_model/dspark_llada")
    main(parser.parse_args())
