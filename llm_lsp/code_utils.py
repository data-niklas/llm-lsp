import re

def determine_indentation(line: str) -> str:
    non_whitespace_match = re.search(r"\S", line)
    if non_whitespace_match is None:
        return ""
    non_whitespace_index = non_whitespace_match.start()
    return line[:non_whitespace_index]

def line_is_note(line) -> bool:
    line = line.strip()
    return line.startswith("# ") and "note:" in line


def remove_notes(code: str) -> str:
    text = "\n".join([line for line in code.splitlines() if not line_is_note(line)])
    return text


def remove_old_notes(code: str) -> str:
    """Removes all notes which are not part of the last comment block"""
    lines = []
    code_lines = code.splitlines()
    if len(code_lines) == 0:
        return code
    code_lines.reverse()
    lines.append(code_lines.pop(0))
    for i, line in enumerate(code_lines):
        lines.append(line)
        if not line_is_note(line):
            lines.extend(
                [line for line in code_lines[i + 1 :] if not line_is_note(line)]
            )
            break
    lines.reverse()
    return "\n".join(lines)