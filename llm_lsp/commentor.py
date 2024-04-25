from llm_lsp.code_utils import CodeUtil
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Optional, Any
from tree_sitter import Tree, Parser


class Lifetime(str, Enum):
    EPHEMERAL = "EPHEMERAL"  # Removes the comment at the next possible instance


@dataclass
class InsertedComment:
    start_line: int
    end_line: int
    interrupt: str
    target_column: int
    context: Any


@dataclass
class Comment:
    comment: str
    interrupt: str
    context: Any

class Commentor:
    def __init__(self, code_util: CodeUtil, parser: Parser):
        self.code_util = code_util
        self.comments: List[InsertedComment] = []  # Always sorted by start_line
        self.parser = parser
        self.tree = None

    def insert_comment(self, code: str, comment: Comment) -> str:
        code_lines = code.splitlines()
        last_line = code_lines.pop()
        start_index = len(code_lines)
        end_index = start_index + comment.comment.count("\n") + 1

        indent = self.code_util.get_indentation_prefix(last_line)

        comment_lines = [
            indent + self.code_util.make_single_line_comment(comment)
            for comment in comment.comment.splitlines()
        ]

        code_lines += comment_lines
        code_lines.append(last_line)

        self.comments.append(
            InsertedComment(
                start_line=start_index,
                end_line=end_index,
                interrupt=comment.interrupt,
                target_column=len(code_lines[-1]),
                context=comment.context
            )
        )
        return "\n".join(code_lines)

    def get_comment_of_interrupt(self, interrupt: str) -> Optional[InsertedComment]:
        for comment in self.comments:
            if comment.interrupt == interrupt:
                return comment
        return None

    def remove_old_comments(self, code: str, interrupt: str) -> str:
        if self.tree is not None:
            self.tree = self.parser.parse(bytes(code, "utf-8"), self.tree)
        else:
            self.tree = self.parser.parse(bytes(code, "utf-8"))
        code_lines = code.splitlines()
        if len(code_lines) == 0:
            return code
        old_comments_with_index = [
            (i, comment)
            for (i, comment) in enumerate(self.comments)
            if comment.interrupt == interrupt or interrupt == "signature"
        ]
        for i, comment in reversed(old_comments_with_index):
            line_count = comment.end_line - comment.start_line
            code_lines = (
                code_lines[: comment.start_line] + code_lines[comment.end_line :]
            )
            for j in range(i + 1, len(self.comments)):
                self.comments[j].start_line -= line_count
                self.comments[j].end_line -= line_count
        self.comments = [
            comment
            for comment in self.comments
            if comment.interrupt != interrupt and interrupt != "signature"
        ]
        return "\n".join(code_lines)

    def remove_all_comments(self, code: str) -> str:
        code_lines = code.splitlines()
        if len(code_lines) == 0:
            return code
        old_comments_with_index = self.comments
        for comment in reversed(old_comments_with_index):
            code_lines = (
                code_lines[: comment.start_line] + code_lines[comment.end_line :]
            )
        self.comments = []
        return "\n".join(code_lines)
