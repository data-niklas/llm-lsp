from evalplus.data import get_human_eval_plus, write_jsonl
from logzero import logger
import asyncio
from llm_lsp.__main__ import create_lsp, initialize_generation
import os
import nest_asyncio
from llm_lsp.lsp_processor import LspLogitsProcessor
from llm_lsp.lsp_client import LspClient
from llm_lsp.constants import *
from llm_lsp.generator import LspGenerator
from llm_lsp.interrupts import InterruptStoppingCriteria, Interrupt, handle_deprecation_interrupt, handle_signature_interrupt

nest_asyncio.apply()

async def main():
    # Assume that it is run from the project root, where the venv is located
    root_directory = os.getcwd()
    lsp = await create_lsp(root_directory)
    interrupts = [
        Interrupt(token=DEPRECATION_INTERRUPT_TOKEN, callable=handle_deprecation_interrupt),
        Interrupt(token=SIGNATURE_INTERRUPT_TOKEN, callable=handle_signature_interrupt)
    ]
    pipeline, tokenizer = initialize_generation(interrupts)
    generator = LspGenerator(pipeline, tokenizer, lsp, interrupts, LspLogitsProcessor)
    samples = []
    for task_id, problem in get_human_eval_plus().items():
        code = problem["prompt"]
        with open("tmp_code2.py", "w") as f:
            f.write(code)
        generated_code = generator(code, "tmp_code2.py")
        logger.info(generated_code)
        samples.append(dict(task_id=task_id, solution=generated_code))
        write_jsonl("samples2.jsonl", samples)       
if __name__ == "__main__":
    asyncio.run(main())