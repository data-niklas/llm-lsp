from dataclasses import dataclass
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod
from llm_lsp.prompt import Prompt, Comment
from transformers import StoppingCriteria
from torch import Tensor
from llm_lsp.code_utils import CodeUtil

class InterruptStoppingCriteria(StoppingCriteria):
    def __init__(self, interrupt_token_id):
        self.interrupt_token_id = interrupt_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Any batch ends with the interrupt token
        for i in range(input_ids.shape[0]):
            if len(input_ids[i]) > 0 and input_ids[i][-1] == self.interrupt_token_id:
                return True
        return False

@dataclass
class Interrupt:
    # Tensor of (return_count * beams * batched_items_count) x currently_generated_tokens
    input_ids: Tensor
    interrupt_context: Any
    interrupt_type_name: str

class InterruptType(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def type_name(self) -> str:
        pass

    @abstractmethod
    def create_comment(self, context: Any, code_util: CodeUtil) -> Optional[Comment]:
        pass




def decode_tokens_with_maybe_interrupt(tokenizer, interrupt_token_id, tokens):
    if tokens[-1] == interrupt_token_id:
        return tokens[-1], tokenizer.decode(tokens[:-1])
    # Remove eos token
    tokens = tokens[:-1]
    if tokens[-1] == interrupt_token_id:
        return tokens[-1], tokenizer.decode(tokens[:-1])
    return None, tokenizer.decode(tokens)

