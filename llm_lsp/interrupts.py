from transformers import StoppingCriteria

class InterruptStoppingCriteria(StoppingCriteria):
    def __init__(self, interrupt_token_id):
        self.interrupt_token_id = interrupt_token_id
    def __call__(self, input_ids, scores, **kwargs):
        # Any batch ends with the interrupt token
        for i in range(input_ids.shape[0]):
            if len(input_ids[i]) > 0 and input_ids[i][-1] == self.interrupt_token_id:
                return True
        return False

def decode_tokens_with_maybe_interrupt(tokenizer, interrupt_token_id, tokens):
    # Remove eos token
    tokens = tokens[:-1]
    if tokens[-1] == interrupt_token_id:
        return True, tokenizer.decode(tokens[:-1])
    return False, tokenizer.decode(tokens)