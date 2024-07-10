from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from llm_lsp.constants import GLOBAL_CONFIGURATION, MODEL


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_config = GLOBAL_CONFIGURATION
    p = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_config,
        return_full_text=False,
    )
    with open(args.file) as f:
        content = f.read()
    result = p(content)[0]["generated_text"]
    print(result)


def parse_args():
    parser = ArgumentParser(
        prog="Pseudo code meta strategy",
        description="Stuff",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-f", "--file", default="tests/rich.py")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
