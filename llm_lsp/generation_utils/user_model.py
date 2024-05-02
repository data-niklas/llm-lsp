import os
import time
import warnings

from typing import Optional, Union, List

import torch
import torch.distributed as dist

from transformers import PreTrainedTokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateNonBeamOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.streamers import BaseStreamer

PATH = "watch/to_watch.py"

EOS = "ß"
PAD = "@"

def wait_for_change(path):
    last_modified = os.path.getmtime(path)
    while os.path.getmtime(path) == last_modified:
        time.sleep(0.1)
    return


def read_path(path):
    with open(path, "r", encoding="utf8") as f:
        return f.read()


def wait_for_longer(path, length):
    text = ""
    while len(text) <= length:
        wait_for_change(path)
        text = read_path(path)
    return text


def overwrite(path, text):
    error = None
    for _ in range(10):
        try:
            with open(path, "w", encoding="utf8") as f:
                f.write(text)
            return
        except Exception as e:
            error = e
    raise error

class HumanConfig(PretrainedConfig):
    model_type = "human"

    def __init__(
            self,
            path: str = PATH,
            eos_token: str = EOS,
            eos_token_id: int = ord(EOS),
            pad_token: str = PAD,
            pad_token_id: int = ord(PAD),
            **kwargs
        ):
        if not os.path.isfile(path):
            raise ValueError("Provided path must point to a valid (text) file")

        self.path = path
        self.eos_token = eos_token
        self.pad_token = pad_token

        super().__init__(
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )


class HumanTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        self._base_vocab_size = 10000
        self._vocab_size = self._base_vocab_size
        self._special_tokens = []
        super().__init__(
            eos_token=EOS,
            pad_token=PAD,
            eos_token_id=ord(EOS),
            pad_token_id=ord(PAD),
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size # would actually be a sys.maxunicode, but this is too big

    def get_vocab(self):
        return {self._convert_id_to_token(id): id for id in range(self.vocab_size)}

    def add_special_tokens(self, special_tokens, **kwargs):
        ast = special_tokens["additional_special_tokens"]
        self._special_tokens = ast
        self._vocab_size = self._base_vocab_size + len(ast)

    def _convert_id_to_token(self, id):
        if id >= self._base_vocab_size:
            return self._special_tokens[id - self._base_vocab_size]
        return chr(id)

    def _convert_token_to_id(self, tokens):
        if tokens in self._special_tokens:
            return self._special_tokens.index(tokens) + self._base_vocab_size
        return ord(tokens)

    def prepare_for_tokenization(self, text, **kwargs):
        return (text, kwargs)

    def _tokenize(self, text, **kwargs):
        return list(text)

    def _decode(self, token_ids, *args, **kwargs):
        return "".join(self._convert_id_to_token(token_id) for token_id in token_ids)


class HumanModel(PreTrainedModel):
    config_class = HumanConfig

    def __init__(self, config: HumanConfig, tok: HumanTokenizer):
        super().__init__(config)
        self.config = config

        self.dummy_tensor = torch.empty([1])
        self.tok = tok

    def forward(
            self,
            input_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        ):
        assert return_dict and not output_attentions and not output_attentions, "Do not support other options"
        num_tokens = self.tok.vocab_size
        batch_size, seq_length = input_ids.shape
        assert batch_size == 1
        logits = torch.zeros([batch_size, 1, num_tokens])
        text = self.tok.decode(input_ids[0])

        overwrite(self.config.path, text)
        new_text = wait_for_longer(self.config.path, seq_length)
        new_input_ids = self.tok.encode(new_text)
        assert len(new_input_ids) > seq_length
        logits[:, -1, new_input_ids[seq_length]] = 10

        return CausalLMOutput(
            logits=logits
        )

    def prepare_inputs_for_generation(
        self, input_ids, **kwargs
    ):
        return {
            "input_ids": input_ids
        }
    
    def greedy_search(self, *args, **kwargs):
        warnings.warn(
            "Calling `greedy_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._greedy_search(*args, **kwargs)

    def _greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # assertions on unsupported parameters
        assert not synced_gpus
        assert not output_attentions
        assert not output_hidden_states
        assert streamer is None

        batch_size, init_seq_length = input_ids.shape
        assert batch_size == 1

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        from time import sleep
        #sleep(0.1)
        while True:
            next_token_logits = torch.zeros([batch_size, self.tok.vocab_size])
            text = self.tok.decode(input_ids[0])

            overwrite(self.config.path, text)
            wait_for_change(self.config.path)
            new_text = read_path(self.config.path)
            new_input_ids = torch.tensor(self.tok.encode(new_text), device=input_ids.device, dtype=input_ids.dtype)
            next_token_logits[:, new_input_ids[-1]] = 10

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([new_input_ids[None, :-1], next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                CausalLMOutput(logits=next_token_logits), model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                if torch.any(input_ids == eos_token_id_tensor):
                    break

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                break

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=None,
                hidden_states=None,
                past_key_values=None,
            )
        else:
            return input_ids

    def resize_token_embeddings(self, *args, **kwargs):
        pass

HumanConfig.register_for_auto_class()
AutoConfig.register("human", HumanConfig)
HumanModel.register_for_auto_class("AutoModelForCausalLM")
AutoModelForCausalLM.register(HumanConfig, HumanModel)

def main():
    config = HumanConfig()
    tok = HumanTokenizer()
    model = HumanModel(config, tok)

    print("Started generation")
    input_ids = model.generate(torch.tensor([tok("abc").input_ids]), max_new_tokens=10)
    print(tok.decode(input_ids[0]))

if __name__ == "__main__":
    main()
