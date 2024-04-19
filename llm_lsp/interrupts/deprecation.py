from llm_lsp.interrupts import InterruptType
from llm_lsp.prompt import Prompt
from llm_lsp.code_utils import CodeUtil
from typing import Any
import importlib
import sys
import json
from typing import List
from functools import lru_cache
from llm_lsp.commentor import Comment, Lifetime, InsertedComment
from llm_lsp.generator import PY_LANGUAGE
from tree_sitter import Tree

PY_IDENTIFIER_QUERY = PY_LANGUAGE.query("(identifier) @element")


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


def next_sibling(node):
    if node.next_sibling is not None:
        return node.next_sibling
    if node.parent is None:
        return None
    return next_sibling(node.parent)


class DeprecationInterrupt(InterruptType):
    def __init__(self):
        super().__init__(TOKEN_ID)

    def is_comment_old(code: str, tree: Tree, comment: InsertedComment) -> bool:
        captures = PY_IDENTIFIER_QUERY.captures(tree.root_node)
        all_identifiers = [capture[0] for capture in captures]
        identifiers_after_comment = [
            identifier
            for identifier in all_identifiers
            if identifier.start_point[0] >= comment.end_line
        ]
        if len(identifiers_after_comment) == 0:
            return False
        last_identifier = identifiers_after_comment[-1]
        identifier_sibling = next_sibling(last_identifier)
        return identifier_sibling is not None and identifier_sibling.type != "."

    def create_comment(self, context: Any, _code_util) -> Comment:
        context.sort(key=lambda x: x.sort_text, reverse=True)
        #context = context[-3:]
        notes = [
            "Deprecation note: "
            + get_deprecation_message(
                completion_item.detail + "." + completion_item.insert_text
            ).strip()
            for completion_item in context
        ]
        return Comment(
            is_old=DeprecationInterrupt.is_comment_old,
            comment="\n".join(notes),
            interrupt=DEPRECATION_COMMENT_TYPE,
        )


if __name__ == "__main__":
    items = json.loads(sys.stdin.read())
    print(json.dumps([get_deprecation_message(item) for item in items]))
