import importlib
import json
import sys
from typing import Any, List, Optional
import logging

logging.disable(logging.INFO)
try:
    import transformers
    transformers.logging.set_verbosity_error()
except:
    pass

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

if __name__ == "__main__":
    cmd = sys.argv[1]
    item = sys.argv[2]
    if cmd == "is_deprecated":
        result = is_deprecated(item)
    elif cmd == "get_deprecation_message":
        result = get_deprecation_message(item)
    else:
        result = "unknown"
    print(json.dumps(result))