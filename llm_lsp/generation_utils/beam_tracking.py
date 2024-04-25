import torch
from typing import Union, List, Optional, Dict
from transformers.generation import BeamSearchScorer
from copy import deepcopy

class BeamIndexStoringSearchScores(BeamSearchScorer):
    def __init__(self, *args, **kwargs):
        if "beam_tracker" in kwargs:
            self.beam_tracker = kwargs["beam_tracker"]
            del kwargs["beam_tracker"]
        else:
            self.beam_tracker = None
        super().__init__(*args, **kwargs)

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        group_index: Optional[int] = 0,
        decoder_prompt_len: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        if self.beam_tracker is not None:
            self.beam_tracker.track_beam_indices(beam_indices)
        return super().process(
            input_ids,
            next_scores,
            next_tokens,
            next_indices,
            pad_token_id,
            eos_token_id,
            beam_indices,
            group_index,
            decoder_prompt_len,
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        beam_indices: Optional[torch.LongTensor] = None,
        decoder_prompt_len: Optional[int] = 0,
    ):
        self.input_ids = input_ids  # NOTE: to be read later on to track beam indices
        if self.beam_tracker is not None:  # NOTE: to be read later on to track beam indices
            self.beam_tracker.track_beam_indices(beam_indices)
        return super().finalize(
            input_ids,
            final_beam_scores,
            final_beam_tokens,
            final_beam_indices,
            max_length,
            pad_token_id,
            eos_token_id,
            beam_indices,
            decoder_prompt_len,
        )

class BeamTracker:
    def __init__(self):
        self.indices = None

    def track_beam_indices(self, beam_indices):
        # Could be optimized by caching old indices calculated from n beam_i
        change_count = len(beam_indices[0])
        beam_batch_count = len(beam_indices)
        indices = list(range(beam_batch_count))
        for i in range(change_count):
            new_indices = deepcopy(indices)
            for j in range(beam_batch_count):
                tuple_index = beam_indices[j]
                torch_index = tuple_index[i]
                index = torch_index.item()
                new_indices[j] = indices[index]
            indices = new_indices
        self.indices = indices

    def get_final_beam_indices(self):
        return self.indices

    def rearrange_according_to_beams(self, items):
        indices = self.get_final_beam_indices()
        if indices is None:
            indices = list(range(len(items)))
        return [items[indices[i]] for i in range(len(items))]

    def is_beam_search(self) -> bool:
        return self.indices is not None