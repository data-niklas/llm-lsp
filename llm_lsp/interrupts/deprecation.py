import importlib
import json
import sys
from functools import lru_cache
from typing import Any, List, Optional
import subprocess
from os import environ, path

from llm_lsp.interrupts import InterruptType
from llm_lsp.prompt_state import Comment

VENV_LSP_FEATURES = path.join(path.dirname(__file__), "venv_lsp_features.py")


@lru_cache
def get_deprecation_message(item):
    venv_dir = environ["VIRTUAL_ENV"]
    environ["TOKENIZERS_PARALLELISM"] = "true"
    python = path.join(venv_dir, "bin", "python")
    cmd = [python, VENV_LSP_FEATURES, "get_deprecation_message", item]
    handle = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    #output = subprocess.check_output(cmd)
    output, _ = handle.communicate()
    return json.loads(output)


@lru_cache
def is_deprecated(item):
    venv_dir = environ["VIRTUAL_ENV"]
    environ["TOKENIZERS_PARALLELISM"] = "true"
    python = path.join(venv_dir, "bin", "python")
    cmd = [python, VENV_LSP_FEATURES, "is_deprecated", item]
    handle = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    #output = subprocess.check_output(cmd)
    output, _ = handle.communicate()
    return json.loads(output)


DEPRECATION_COMMENT_TYPE = "deprecation"


class DeprecationInterrupt(InterruptType):
    def __init__(self):
        super().__init__()

    def type_name(self) -> str:
        return DEPRECATION_COMMENT_TYPE

    def create_comment(self, context: Any, _code_util) -> Optional[Comment]:
        if len(context) == 0:
            return None
        used_context = [a for a in context]
        used_context.sort(key=lambda x: x.sort_text)
        # context = context[-3:]
        notes = [
            # "The following variable is deprecated, use an alternative: " +
            "Hint: "
            + get_deprecation_message(
                completion_item.detail + "." + completion_item.insert_text
            ).strip()
            for completion_item in used_context
        ]
        return Comment(
            comment="\n".join(notes),
            interrupt=DEPRECATION_COMMENT_TYPE,
            context=context,
        )


if __name__ == "__main__":
    items = json.loads(sys.stdin.read())
    print(json.dumps([get_deprecation_message(item) for item in items]))
