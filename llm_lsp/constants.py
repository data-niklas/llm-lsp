MODEL = "codellama/CodeLlama-7b-Instruct-hf"
#MODEL = "ise-uiuc/Magicoder-S-DS-6.7B"
MODEL = "m-a-p/OpenCodeInterpreter-DS-6.7B"
#MODEL = "deepseek-ai/deepseek-coder-33b-instruct"
#MODEL = "google/codegemma-7b-it"
#MODEL = "microsoft/Phi-3-mini-4k-instruct"

# . Do not generate more code than necessary. Do not generate comments with further tasks.
GLOBAL_CONFIGURATION = {
    # "num_beam_groups": 0,
    "num_beams": 1,
    # "diversity_penalty": 1.0,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 1,
    #"return_full_text": False,
    #"max_new_tokens": 2048,
    "max_new_tokens": 5048,
    #"repetition_penalty": 1.3
    "repetition_penalty": 1.18

}

MAXIMUM_DOCUMENTATION_LENGTH = 500