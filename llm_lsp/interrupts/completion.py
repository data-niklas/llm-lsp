from llm_lsp.interrupts import InterruptType
from llm_lsp.prompt import Prompt
from llm_lsp.code_utils import CodeUtil
from typing import Any, Optional
import importlib
import sys
import json
from typing import List
from functools import lru_cache
from llm_lsp.commentor import Comment, Lifetime, InsertedComment
from lsprotocol.types import CompletionItemKind
from llm_lsp.code_utils import CodeUtil



TOKEN_ID = "[COMPLETION_INTERRUPT]"
COMPLETION_COMMENT_TYPE = "completion"




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


    def type_name(self) -> str:
        return COMPLETION_COMMENT_TYPE

    def create_comment(self, context: Any, code_util: CodeUtil) -> Optional[Comment]:
        if len(context) == 0:
            return None
        used_context = [item for item in context]
        used_context.sort(key=lambda x: x.sort_text, reverse=True)
        notes = ["Hint: Use one of the following to complete the variable: " + ", ".join([item.label for item in used_context])]
        return Comment(
            comment="\n".join(notes),
            interrupt=COMPLETION_COMMENT_TYPE,
            context=context
        )


if __name__ == "__main__":
    items = json.loads(sys.stdin.read())
    print(json.dumps([get_deprecation_message(item) for item in items]))
