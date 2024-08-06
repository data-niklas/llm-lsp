# LLM_LSP
Enhance neural code completion using static analysis!
This repository enhances the code completion of instruction based LLMs running on the Transformers runtime.
It uses static analysis provided by a language server to provide exact information on dependencies, such as function signatures, code completions and deprecated items.

## Installation
The code exposes the Python package `llm_lsp`, which can be installed as any other Python package:
```sh
git clone https://github.com/data-niklas/llm-lsp/
cd llm_lsp
# install dependencies in a virtual environment
# alternatively use conda
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run `llm_lsp/__main__.py` with the necessary arguments, to complete a given code file:
```sh
python llm_lsp -f path/to/code
```
For further testing, the configuration values in `llm_lsp/constants.py` and `llm_lsp/__main__.py` -> `LspGenerationConfig` can be changed.

For serious use it is recommended to use the package as a library programmatically:
```sh
git clone https://github.com/data-niklas/llm-lsp/
cd llm_lsp
# install dependencies in a virtual environment
# alternatively use conda
python -m venv venv
source venv/bin/activate
pip install .
```
Now import and use it inside other code. For an example refer to https://github.com/data-niklas/DependencyEval/blob/main/dependency_eval/eval_prompt.py .


## Configuration
The library exposes two methods of configuring the generation:
1. The generation config of the model / the used model
2. Toggling features of the approach

The generation config may be changed directly in `llm_lsp/constants.py`, or by passing the variable `generation_config` into the `Generator`.

Toggling features requires a `LspGenerationConfig` (look at `llm_lsp/config.py`), which is passed into the `Generator` through the parameter `config`.

Default toggled features of the approach:
```py
    comments_processor: bool = True
    boundary_processor: bool = True
    lsp_processor: bool = True
    chat_history_log_file: Optional[str] = None
    predict_correct_completion_symbol: bool = True
    force_custom_pad: bool = False
    masked_gen: bool = True
    use_completion_context: bool = True
    use_deprecation_context: bool = True
    use_signature_context: bool = True
    enabled: bool = (
        True  # quick setting to disable all processors, overrides other settings
    )
```

Setting the chat history file will log the prompts during interrupts, before and aftre generation. Masked generation may be toggled, predicting the correct completion item, using different Logits Processors, and the use of different information sources for the interrupts.