from transformers import StoppingCriteria
from dataclasses import dataclass
from typing import Callable, Optional, Any
from logzero import logger
import re
from llm_lsp.deprecation_messages import get_deprecation_message
from llm_lsp.constants import *


class InterruptStoppingCriteria(StoppingCriteria):
    def __init__(self, interrupt_token_ids):
        self.interrupt_token_ids = interrupt_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Any batch ends with the interrupt token
        for i in range(input_ids.shape[0]):
            if len(input_ids[i]) > 0 and input_ids[i][-1] in self.interrupt_token_ids:
                return True
        return False


def determine_indentation(line: str):
    non_whitespace_match = re.search(r"\S", line)
    if non_whitespace_match is None:
        return ""
    non_whitespace_index = non_whitespace_match.start()
    return line[:non_whitespace_index]

def line_is_note(line):
    line = line.strip()
    return line.startswith("# ") and "note:" in line

def remove_notes(code: str) -> str:
    text = "\n".join([line for line in code.splitlines() if not line_is_note(line)])
    return text


def remove_old_notes(code: str) -> str:
    """Removes all notes which are not part of the last comment block"""
    lines = []
    code_lines = code.splitlines()
    code_lines.reverse()
    lines.append(code_lines.pop(0))
    for i, line in enumerate(code_lines):
        lines.append(line)
        if not line_is_note(line):
            lines.extend([line for line in code_lines[i+1:] if not line_is_note(line)])
            break
    lines.reverse()
    return "\n".join(lines)


def add_deprecation_notes(completion_items, code):
    first_lines, last_line = code.rsplit("\n", 1)
    indentation = determine_indentation(last_line)
    comments = [
        indentation
        + "# Deprecation note: "
        + get_deprecation_message(
            completion_item.detail + "." + completion_item.insert_text
        ).strip()
        for completion_item in completion_items
    ]
    comments_text = "\n".join(comments)
    return first_lines + "\n" + comments_text + "\n" + last_line


def add_signature_notes(signature_help, code):
    try:
        first_lines, last_line = code.rsplit("\n", 1)
    except ValueError:
        first_lines, last_line = "", code
    indentation = determine_indentation(last_line)

    active_signature = signature_help.signatures[signature_help.active_signature]
    documentation = active_signature.documentation.value
    comments_text = indentation + "# Signature note: " + active_signature.label.strip()
    if len(documentation) > 0 and len(documentation) < MAXIMUM_DOCUMENTATION_LENGTH:
        comments_text += (
            "\n"
            + indentation
            + '# Signature note: Documentation is: """'
            + documentation.replace("\n", "\n" + indentation + "# Signature note: ")
            + '"""'
        )
    return first_lines + "\n" + comments_text + "\n" + last_line


def decode_tokens_with_maybe_interrupt(tokenizer, interrupt_token_ids, tokens):
    if tokens[-1] in interrupt_token_ids:
        return tokens[-1], tokenizer.decode(tokens[:-1])
    # Remove eos token
    tokens = tokens[:-1]
    if tokens[-1] in interrupt_token_ids:
        return tokens[-1], tokenizer.decode(tokens[:-1])
    return None, tokenizer.decode(tokens)


def handle_deprecation_interrupt(deprecated_items, only_generated_code, code):
    generated_code_with_notes = add_deprecation_notes(
        deprecated_items, only_generated_code
    )
    prompt = (
        "[INST] "
        + PROMPT_TEMPLATE
        + "\n[/INST]\n"
        + code
        + "\n"
        + generated_code_with_notes
    )
    logger.debug("New prompt is:")
    logger.debug(prompt)
    return prompt


def handle_signature_interrupt(signature_help, only_generated_code, code):
    generated_code_with_notes = add_signature_notes(signature_help, only_generated_code)
    prompt = (
        "[INST] "
        + PROMPT_TEMPLATE
        + "\n[/INST]\n"
        + code
        + "\n"
        + generated_code_with_notes
    )
    logger.debug("New prompt is:")
    logger.debug(prompt)
    return prompt


def get_new_prompt_or_finish(
    tokenizer,
    interrupts,
    last_token_ids,
    text_len_prompt_with_initial_code,
    processor,
    code,
):
    """Returns if it is finished and the text (either finished text or new prompt)"""
    interrupt_token_ids = [interrupt.input_id for interrupt in interrupts]
    interrupt_id, text = decode_tokens_with_maybe_interrupt(
        tokenizer, interrupt_token_ids, last_token_ids
    )
    # + 1 is for newline added in the prompt creation
    only_generated_code = text[text_len_prompt_with_initial_code:]
    if interrupt_id is None:
        only_generated_code = remove_notes(only_generated_code)
        return True, code + "\n" + only_generated_code
    only_generated_code = remove_old_notes(only_generated_code)
    logger.warn(only_generated_code)
    interrupt = [
        interrupt for interrupt in interrupts if interrupt.input_id == interrupt_id
    ][0]
    logger.info(f"Interrupt {interrupt.token}")
    interrupt_callable = interrupt.callable
    prompt = interrupt_callable(processor.interrupt_data, only_generated_code, code)
    return False, prompt


@dataclass
class Interrupt:
    token: str
    callable: Callable[[Any, str, str], str]
    input_id: Optional[int] = None
