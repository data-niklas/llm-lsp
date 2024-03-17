from dataclasses import dataclass
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod
from llm_lsp.prompt import Prompt
from transformers import StoppingCriteria


class InterruptStoppingCriteria(StoppingCriteria):
    def __init__(self, interrupt_token_ids):
        self.interrupt_token_ids = interrupt_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Any batch ends with the interrupt token
        for i in range(input_ids.shape[0]):
            if len(input_ids[i]) > 0 and input_ids[i][-1] in self.interrupt_token_ids:
                return True
        return False

class Interrupt(ABC):
    def __init__(self, token: str):
        self.token = token
        self.input_id = None

    def init_input_id(self, tokenizer):
        self.input_id = tokenizer.convert_tokens_to_ids(self.token)

    @abstractmethod
    def edit_generated_code_for_completion(self, generated_code: str, context: Any) -> str:
        pass




def decode_tokens_with_maybe_interrupt(tokenizer, interrupt_token_ids, tokens):
    if tokens[-1] in interrupt_token_ids:
        return tokens[-1], tokenizer.decode(tokens[:-1])
    # Remove eos token
    tokens = tokens[:-1]
    if tokens[-1] in interrupt_token_ids:
        return tokens[-1], tokenizer.decode(tokens[:-1])
    return None, tokenizer.decode(tokens)

