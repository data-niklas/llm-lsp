import argparse
import asyncio

import nest_asyncio
import torch
from logzero import logger
from lsprotocol.types import *
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_lsp.constants import *
from llm_lsp.generator import Generator
from llm_lsp.config import LspGenerationConfig

nest_asyncio.apply()


# 1. meta-strategy part-by-part generation
# -> pipeline -> add information to prompt -> pipeline -> add information to prompt -> pipeline -> done
# custom token in vocab
# add comment so that first part is not

# 2. token by token
# -> pipeline -> done
# (using LLM to rerank output completion list)


def highlight_code(code):
    return highlight(code, PythonLexer(), Terminal256Formatter())


async def generate_test(model, tokenizer, generation_config, code, repo_root, file_name):
    generator = Generator(model, tokenizer, generation_config)

    return await generator.complete(code, repo_root, file_name)


async def test(args):
    from llm_lsp.generation_utils.user_model import (HumanConfig, HumanModel,
                                                     HumanTokenizer)

    config = HumanConfig("/nfs/home/nloeser_msc2023/llm-lsp-typing-demo/watch.py")
    tokenizer = HumanTokenizer()
    model = HumanModel(config, tokenizer)
    generation_config = {
        "num_return_sequences": 1,
        "do_sample": False,
        "max_length": 5000,
    }
    generator = Generator(model, tokenizer, generation_config)
    if args.strategy == "COMPLETE":
        with open(args.file, "r") as f:
            code = f.read()
        repo_root = args.directory
        file_name = args.file
        completed_code = await generator.complete(code, repo_root, file_name)
        code += completed_code
        hl = highlight_code(code)
        logger.info("Code:\n##########\n" + hl + "\n##########")

async def main(args):
    # if True:
    #    await test(args)
    #    return
    config = LspGenerationConfig(
        chat_history_log_file=args.chat_history_log_file,
        use_completion_context=False,
        masked_gen=False,
        predict_correct_completion_symbol=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    generation_config = GLOBAL_CONFIGURATION
    generator = Generator(model, tokenizer, generation_config, config=config)
    with open(args.file, "r") as f:
        code = f.read()
    repo_root = args.directory
    file_name = args.file
    completed_code = await generator.complete(code, repo_root, file_name)
    code += completed_code
    hl = highlight_code(code)
    logger.info("Code:\n##########\n" + hl + "\n##########")

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Pseudo code meta strategy",
        description="Stuff",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-f", "--file", default="tests/bidict_2.py")
    parser.add_argument("-d", "--directory", default=".")
    parser.add_argument("-l", "--level", default="DEBUG")
    parser.add_argument("-v", "--chat_history_log_file", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
