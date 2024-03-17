#MODEL = "codellama/CodeLlama-7b-Instruct-hf"
#MODEL = "ise-uiuc/Magicoder-S-DS-6.7B"
MODEL = "m-a-p/OpenCodeInterpreter-DS-6.7B"
#MODEL = "deepseek-ai/deepseek-coder-33b-instruct"

# . Do not generate more code than necessary. Do not generate comments with further tasks.
GLOBAL_CONFIGURATION = {
    # "num_beam_groups": 0,
    "num_beams": 4,
    # "diversity_penalty": 1.0,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 1,
    #"return_full_text": False,
    #"temperature": 0.7,
    "max_new_tokens": 2048,
    "repetition_penalty": 1.3
}
COMPLETION_INTERRUPT_TOKEN = '[COMPLETION_INTERRUPT]'
SIGNATURE_INTERRUPT_TOKEN = '[SIGNATURE_INTERRUPT]'

MAXIMUM_DOCUMENTATION_LENGTH = 500