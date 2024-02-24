from llm_lsp.constants import *
from llm_lsp.interrupts import get_new_prompt_or_finish, InterruptStoppingCriteria
from transformers import AutoTokenizer, Pipeline, LogitsProcessorList, StoppingCriteriaList, StoppingCriteria
from llm_lsp.prompt import Prompt, make_prompt
from llm_lsp.strategy import GenerationStrategy

class LspGenerator:
    def __init__(self, pipeline, tokenizer, lsp, interrupts, logits_processor_cls, strategy: GenerationStrategy):
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.lsp = lsp
        self.interrupts = interrupts
        self.logits_processor_cls = logits_processor_cls
        self.prompt_util = Prompt(make_prompt, tokenizer)
        self.strategy = strategy

    def __call__(self, code: str, file: str) -> str:
        """Kwargs are different depending on the strategy"""
        for interrupt in self.interrupts:
            interrupt.input_id = self.tokenizer.convert_tokens_to_ids(interrupt.token)
        interrupt_input_ids = [interrupt.input_id for interrupt in self.interrupts]
        # Last \n needed such that it will be cut off later by the +1
        # code is the initial instruction for the generation
        prompt = self.prompt_util.format(code, "", self.strategy)

        processor = self.logits_processor_cls(self.tokenizer, self.lsp, self.prompt_util, self.pipeline, self.strategy, file)
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
            finished, text = get_new_prompt_or_finish(self.tokenizer, self.interrupts, last_token_ids, self.prompt_util, processor, self.prompt_util.code, self.strategy)
            if finished:
                return text
            prompt = text