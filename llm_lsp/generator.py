from llm_lsp.constants import *
from llm_lsp.interrupts import get_new_prompt_or_finish, InterruptStoppingCriteria
from transformers import AutoTokenizer, Pipeline, LogitsProcessorList, StoppingCriteriaList, StoppingCriteria


class LspGenerator:
    def __init__(self, pipeline, tokenizer, lsp, interrupts, logits_processor_cls):
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.lsp = lsp
        self.interrupts = interrupts
        self.logits_processor_cls = logits_processor_cls

    def __call__(self, code: str, file: str) -> str:
        for interrupt in self.interrupts:
            interrupt.input_id = self.tokenizer.convert_tokens_to_ids(interrupt.token)
        interrupt_input_ids = [interrupt.input_id for interrupt in self.interrupts]
        prompt = "[INST] " + PROMPT_TEMPLATE + "\n[/INST]\n" + code
        text_len_prompt_with_initial_code = len(prompt)
        processor = self.logits_processor_cls(self.tokenizer, self.lsp, len(prompt), file, code, self.pipeline)
        while True:
            sequences = self.pipeline(
                prompt,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([processor]),
                stopping_criteria=StoppingCriteriaList([InterruptStoppingCriteria(interrupt_input_ids)]),
                return_tensors=True,
                **GLOBAL_CONFIGURATION,
            )
            last = sequences[-1]
            last_token_ids = last["generated_token_ids"]
            finished, text = get_new_prompt_or_finish(self.tokenizer, self.interrupts, last_token_ids, text_len_prompt_with_initial_code, processor, code)
            if finished:
                return text
            prompt = text