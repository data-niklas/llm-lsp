from llm_lsp.interrupts import InterruptType
from llm_lsp.prompt import Prompt, Comment
from llm_lsp.code_utils import CodeUtil
from typing import Any, Optional
import importlib
import sys
import json
from typing import List
from functools import lru_cache



def module_name_and_variable_parts(item):
    parts: List[str] = item.split(".")
    module_name_parts = []
    variable_parts = []
    for i, part in enumerate(parts):
        if part.islower():
            module_name_parts.append(part)
        else:
            variable_parts.extend(parts[i:])
            break
    module_name = ".".join(module_name_parts)
    return module_name, variable_parts


def get_deprecation_message(item):
    module_name, variable_parts = module_name_and_variable_parts(item)
    variable_parts.append("__deprecated__")
    module = importlib.import_module(module_name)
    variable = module
    for variable_part in variable_parts:
        if not hasattr(variable, variable_part):
            return None
        variable = getattr(variable, variable_part)
    return variable


@lru_cache
def is_deprecated(item):
    module_name, variable_parts = module_name_and_variable_parts(item)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return False
    except ValueError:
        # empty module name
        return False
    variable = module
    for variable_part in variable_parts:
        if not hasattr(variable, variable_part):
            return False
        variable = getattr(variable, variable_part)
    return hasattr(variable, "__deprecated__")


TOKEN_ID = "[DEPRECATION_INTERRUPT]"
DEPRECATION_COMMENT_TYPE = "deprecation"



class DeprecationInterrupt(InterruptType):
    def __init__(self):
        super().__init__(TOKEN_ID)

    def type_name(self) -> str:
        return DEPRECATION_COMMENT_TYPE

    def create_comment(self, context: Any, _code_util) -> Optional[Comment]:
        if len(context) == 0:
            return None
        used_context = [a for a in context]
        used_context.sort(key=lambda x: x.sort_text, reverse=True)
        #context = context[-3:]
        notes = [
            #"The following variable is deprecated, use an alternative: " +
            "Hint: " + get_deprecation_message(
                completion_item.detail + "." + completion_item.insert_text
            ).strip()
            for completion_item in used_context
        ]
        return Comment(
            comment="\n".join(notes),
            interrupt=DEPRECATION_COMMENT_TYPE,
            context=context
        )


if __name__ == "__main__":
    items = json.loads(sys.stdin.read())
    print(json.dumps([get_deprecation_message(item) for item in items]))
