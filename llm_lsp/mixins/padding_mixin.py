# from llm_lsp.lsp_client import LspClient

import torch.nn.functional as F


class PaddingMixin:
    """
    Encapsulates all functionality which works with padded tokens.
    Requires a tokenizer
    """

    def pad_input_ids(self, input_ids, edited_input_id):
        pad_token_id = self.tokenizer.pad_token_id
        edited_len = edited_input_id.shape[1]
        inputs_len = input_ids.shape[1]
        pad_to_len = max(edited_len, inputs_len)
        edited_input_id = F.pad(
            edited_input_id, (pad_to_len - edited_len, 0), value=pad_token_id
        )
        input_ids = F.pad(input_ids, (pad_to_len - inputs_len, 0), value=pad_token_id)
        return input_ids, edited_input_id

    def remove_padding(self, tokens):
        token_id = self.tokenizer.pad_token_id
        index = (tokens != token_id).nonzero()[0].item()
        return tokens[index:]

    def remove_nd_padding(self, tokens):
        token_id = self.tokenizer.pad_token_id
        index = (tokens != token_id).nonzero()[:, -1].min().item()
        return tokens[:, index:]
