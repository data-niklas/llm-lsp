from torch import FloatTensor, LongTensor
from transformers import LogitsProcessor


class CommentsLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, disabled=False):
        vocab = tokenizer.get_vocab()
        comment_tokens = [token for token in vocab.keys() if "#" in token]
        self.comment_token_ids = [
            vocab[comment_token] for comment_token in comment_tokens
        ]
        self.disabled = disabled

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        """Returns a 2d FloatTensor which has scores for every batch"""
        if self.disabled:
            return scores
        for i in range(input_ids.shape[0]):
            score = scores[i]
            for comment_token_id in self.comment_token_ids:
                # Was -inf but caused errors?
                score[comment_token_id] = -100.0
            scores[i] = score
        return scores
