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
from lsprotocol.types import CompletionItemKind
from llm_lsp.code_utils import CodeUtil

PY_IDENTIFIER_QUERY = PY_LANGUAGE.query("(identifier) @element")


TOKEN_ID = "[COMPLETION_INTERRUPT]"
COMPLETION_COMMENT_TYPE = "completion"


def next_sibling(node):
    if node.next_sibling is not None:
        return node.next_sibling
    if node.parent is None:
        return None
    return next_sibling(node.parent)


KINDS = {
    CompletionItemKind.Variable: "variable",
    CompletionItemKind.Class: "class",
    CompletionItemKind.Field: "field",
    CompletionItemKind.Property: "property",
    CompletionItemKind.Method: "method",
    CompletionItemKind.Function: "function",
}


def kind_text(kind):
    if kind in KINDS:
        return KINDS[kind]
    else:
        return "unknown"


def extract_doc(completion_item):
    doc = completion_item.documentation
    if doc is None:
        return ""
    if not isinstance(doc, str):
        doc = doc.value
    doc = doc.strip()
    stripped_doc = doc.strip("`").strip()
    if len(stripped_doc) == 0:
        return ""
    return doc


def get_completion_message(completion_item, code_util) -> str:
    if completion_item.kind is None:
        kind = ""
    else:
        kind = kind_text(completion_item.kind) + " "
    text = kind + completion_item.label
    documentation = code_util.shorten_docstring(extract_doc(completion_item))
    if len(documentation) > 0:
        text += ": " + documentation
    return text


class CompletionInterrupt(InterruptType):
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
        # Prevent direct removal when triggering autocompletion with '.'
        # TODO: attach comment to lifetime of node
        return identifier_sibling is not None and identifier_sibling.type != "."

    def create_comment(self, context: Any, code_util: CodeUtil) -> Comment:
        context.sort(key=lambda x: x.sort_text, reverse=True)
        context = context[-3:]
        notes = [
            "Completion note: "
            + get_completion_message(completion_item, code_util).replace(
                "\n", "\nCompletion note: "
            )
            for completion_item in context
        ]
        return Comment(
            is_old=CompletionInterrupt.is_comment_old,
            comment="\n".join(notes),
            interrupt=COMPLETION_COMMENT_TYPE,
        )


if __name__ == "__main__":
    items = json.loads(sys.stdin.read())
    print(json.dumps([get_deprecation_message(item) for item in items]))
