MODEL = "codellama/CodeLlama-7b-Instruct-hf"
MODEL = "ise-uiuc/Magicoder-S-DS-6.7B"
#MODEL = "m-a-p/OpenCodeInterpreter-DS-6.7B"
#MODEL = "deepseek-ai/deepseek-coder-33b-instruct"
#MODEL = "google/codegemma-7b-it"

# . Do not generate more code than necessary. Do not generate comments with further tasks.
GLOBAL_CONFIGURATION = {
    # "num_beam_groups": 0,
    "num_beams": 1,
    # "diversity_penalty": 1.0,
    "do_sample": False,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 1,
    #"return_full_text": False,
    "max_new_tokens": 2048,
    "repetition_penalty": 1.18,
    #"no_repeat_ngram_size": 3
}

MAXIMUM_DOCUMENTATION_LENGTH = 500