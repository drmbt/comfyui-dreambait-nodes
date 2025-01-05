import torch
import numpy as np
import scipy
import os
#import re
from pathlib import Path
import folder_paths

FONTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")

SCRIPT_DIR = Path(__file__).parent


# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

def parse_string_to_list(s):
    elements = s.split(',')
    result = []

    def parse_number(s):
        try:
            if '.' in s:
                return float(s)
            else:
                return int(s)
        except ValueError:
            return 0

    def decimal_places(s):
        if '.' in s:
            return len(s.split('.')[1])
        return 0

    for element in elements:
        element = element.strip()
        if '...' in element:
            start, rest = element.split('...')
            end, step = rest.split('+')
            decimals = decimal_places(step)
            start = parse_number(start)
            end = parse_number(end)
            step = parse_number(step)
            current = start
            if (start > end and step > 0) or (start < end and step < 0):
                step = -step
            while current <= end:
                result.append(round(current, decimals))
                current += step
        else:
            result.append(round(parse_number(element), decimal_places(element)))

    return result

def parse_keys(key_string):
    """
    Parse a string of keys into a list, handling various separators.
    Handles space-separated, comma-separated, and comma-space separated formats.
    Returns None if input is empty or '*', indicating all keys should be included.
    """
    if not key_string or key_string.strip() == '*':
        return None
    # First replace comma-space with comma, then replace remaining spaces with commas
    normalized = key_string.replace(', ', ',').replace(' ', ',')
    # Split by comma and filter out empty strings
    return [k.strip() for k in normalized.split(',') if k.strip()]


class DynamicDictionary:
    """
    A node that combines multiple inputs into a dictionary and can filter by keys.
    Flattens incoming dictionaries to keep all values accessible at the top level.
    """
    RETURN_TYPES = ("DICT", "DICT", "STRING", any_typ, "STRING", any_typ,)
    RETURN_NAMES = ("complete_dictionary", "filtered_dictionary", "keys", "values", "info", "types",)
    FUNCTION = "combine"
    CATEGORY = "utils"
    DESCRIPTION = "Creates a dictionary from multiple inputs. Flattens nested dictionaries. Filter by keys using comma or space-separated list. Get values as list or single value."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "keys": ("STRING", {
                    "default": "*",
                    "multiline": False,
                    "placeholder": "Filter keys (space or comma-separated). Use * for all",
                    "tooltip": "List of keys to include, separated by spaces or commas. Use * or leave empty for all keys. Examples: 'image prompt seed' or 'image, prompt, seed'"
                }),
                "delimiter": ("STRING", {
                    "default": ", ",
                    "multiline": False,
                    "placeholder": "Delimiter for info string (e.g., ', ' or '\\n')",
                    "tooltip": "Character(s) to separate key-value pairs in info string. Use \\n for new lines. Examples: ', ' or ' | ' or '\\n'"
                })
            }
        }

    RETURN_TYPES_HINTS = {
        "complete_dictionary": "Complete dictionary with type information. Nested dictionaries are flattened with underscore-separated keys",
        "filtered_dictionary": "Filtered dictionary based on specified keys",
        "keys": "Space-separated list of available keys",
        "values": "Single value if one specific key requested, otherwise list of values in key order",
        "info": "Formatted string of key-value pairs with specified delimiter",
        "types": "Single type string if one specific key requested, otherwise list of type strings in key order"
    }

    OUTPUT_IS_LIST = (False, False, False, True, False, True)

    OUTPUT_NODE_COLORS = [
        (0.84, 0.84, 0.84),  # Gray for complete dictionary
        (0.9, 0.9, 0.9),     # Light gray for filtered dictionary
        (0.7, 0.7, 1.0),     # Light blue for keys string
        (1.0, 0.7, 0.7),     # Light red for values
        (0.7, 1.0, 0.7),     # Light green for info string
        (1.0, 0.85, 0.7)     # Light orange for types
    ]

    def _format_value_for_info(self, value, type_name):
        """Format value for info string based on its type"""
        if type_name in ['tensor', 'Tensor'] or isinstance(value, torch.Tensor):
            return "<image>"
        elif type_name == "audio" or hasattr(value, 'audio_data'):
            return "<audio>"
        elif isinstance(value, (list, tuple)):
            return f"[{len(value)} items]"
        elif isinstance(value, dict):
            return "{...}"
        elif isinstance(value, float):
            # Format floats with reasonable precision
            return f"{value:.4f}"
        elif isinstance(value, (int, bool)):
            return str(value)
        return str(value)

    def _get_clean_type(self, value):
        """Get clean type name for a value"""
        if isinstance(value, torch.Tensor):
            return "tensor"
        elif hasattr(value, 'audio_data'):  # Check for LazyAudioMap
            return "audio"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "dict"
        return type(value).__name__

    def _flatten_dict(self, d, parent_key=''):
        """
        Recursively flatten a dictionary, handling both regular dicts and typed dicts.
        Nested keys are joined with underscores.
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{k}" if parent_key else k
            
            # If it's our special dict format with type/value
            if isinstance(v, dict) and "type" in v and "value" in v:
                if isinstance(v["value"], dict):
                    # Recursively flatten the inner dictionary
                    items.update(self._flatten_dict(v["value"], f"{new_key}_"))
                else:
                    items[new_key] = {
                        "type": self._get_clean_type(v["value"]),
                        "value": v["value"]
                    }
            
            # If it's a regular dictionary
            elif isinstance(v, dict):
                items.update(self._flatten_dict(v, f"{new_key}_"))
            
            # If it's a regular value
            else:
                items[new_key] = {
                    "type": self._get_clean_type(v),
                    "value": v
                }
                
        return items

    def combine(self, keys="*", delimiter=", ", **kwargs):
        # Create flattened dictionary with type information
        complete_dict = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                # Flatten incoming dictionaries
                complete_dict.update(self._flatten_dict(v))
            else:
                complete_dict[k] = {
                    "type": self._get_clean_type(v),
                    "value": v
                }
        
        # Create filtered dictionary based on keys
        filtered_dict = complete_dict.copy()
        if parsed_keys := parse_keys(keys):
            filtered_dict = {k: v for k, v in complete_dict.items() if k in parsed_keys}
        
        # Create string representation of keys
        keys_str = " ".join(filtered_dict.keys())
        
        # Get values and types
        values = [v["value"] for v in filtered_dict.values()]
        types = [v["type"] for v in filtered_dict.values()]
        
        # Always wrap values and types in lists, even for single items
        if len(filtered_dict) == 1:
            values_out = values  # Already a single-item list
            types_out = types   # Already a single-item list
        else:
            values_out = values
            types_out = types
            
        # Create formatted info string with key-value pairs
        delimiter = delimiter.replace('\\n', '\n')
        info_pairs = [
            f"{k}: {self._format_value_for_info(v['value'], v['type'])}" 
            for k, v in filtered_dict.items()
        ]
        info_str = delimiter.join(info_pairs)
        
        return (complete_dict, filtered_dict, keys_str, values_out, info_str, types_out,)