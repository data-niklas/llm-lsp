MODEL = "codellama/CodeLlama-7b-Instruct-hf"
#MODEL = "ise-uiuc/Magicoder-S-DS-6.7B"

PROMPT_TEMPLATE = "You are a code completion tool. Complete the following Python code. Only provide the completed code. Do not return descriptions of your actions"
# . Do not generate more code than necessary. Do not generate comments with further tasks.
GLOBAL_CONFIGURATION = {
    # "num_beam_groups": 0,
    "num_beams": 1,
    # "diversity_penalty": 1.0,
    "do_sample": False,
    #"top_k": 50,
    #"top_p": 0.95,
    "num_return_sequences": 1,
    #"return_full_text": False,
    #"temperature": 0.7,
    "max_new_tokens": 2048,
    "repetition_penalty": 1.3
}
DEPRECATION_INTERRUPT_TOKEN = '[DEPRECATION_INTERRUPT]'
SIGNATURE_INTERRUPT_TOKEN = '[SIGNATURE_INTERRUPT]'

MAXIMUM_DOCUMENTATION_LENGTH = 500