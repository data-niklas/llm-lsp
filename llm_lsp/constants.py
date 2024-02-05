MODEL = "codellama/CodeLlama-7b-Instruct-hf"

PROMPT_TEMPLATE = "You are a code completion tool. Complete the following Python tool. Only provide the completed code. Do not return descriptions of your actions. Do not generate more code than necessary. Do not generate comments with further tasks."
GLOBAL_CONFIGURATION = {
    # "num_beam_groups": 0,
    "num_beams": 2,
    # "diversity_penalty": 1.0,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 1,
    #"return_full_text": False,
    "temperature": 0.7,
    "max_new_tokens": 2048,
}
DOCUMENTATION_INTERRUPT_TOKEN = '[DOCUMENTATION_INTERRUPT]'