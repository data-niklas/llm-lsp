from marisa_trie import Trie
from transformers import LogitsProcessor, PreTrainedTokenizer
from torch import LongTensor, FloatTensor, full_like, full

class MapLogitsProcessor(LogitsProcessor):
    def __init__(self, input_id_map, disabled=False):
        self.input_id_map = input_id_map
        self.disabled = disabled

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        """Returns a 2d FloatTensor which has scores for every batch"""
        if self.disabled:
            return scores
        for i in range(input_ids.shape[0]):
            score = scores[i]
            for (k,v) in self.input_id_map.items():
                a,b = score[k], score[v]
                if a > b:
                    score[k] = b
                    score[v] = a
            scores[i] = score            
        return scores

class BoundaryLogitsProcessor(MapLogitsProcessor):
    def __init__(self, tokenizer, token_boundaries, disabled=False):
        vocab = tokenizer.get_vocab()
        trie = Trie(vocab.keys())
        input_id_map = {}
        for boundary in token_boundaries:
            boundary_tokens = [token for token in vocab.keys() if boundary in token]
            tokens_to_map = [token for token in boundary_tokens if not token.endswith(boundary) and token.count(boundary) == 1]
            def last_allowed_token(prefixes):
                return [prefix for prefix in prefixes if prefix.endswith(boundary) or boundary not in prefix][-1]
            token_map = {token: last_allowed_token(trie.prefixes(token)) for token in tokens_to_map}
            boundary_input_id_map = {vocab[k]: vocab[v] for (k,v) in token_map.items()}
            input_id_map.update(boundary_input_id_map)
        super().__init__(input_id_map, disabled)