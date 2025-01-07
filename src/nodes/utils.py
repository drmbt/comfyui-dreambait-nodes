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
                    "tooltip": "List of keys to include, separated by spaces or commas. Use * or leave empty for all keys."
                }),
                "delimiter": ("STRING", {
                    "default": ", ",
                    "multiline": False,
                    "placeholder": "Delimiter for info string (e.g., ', ' or '\\n')",
                    "tooltip": "Character(s) to separate key-value pairs in info string."
                })
            }
        }

    def _format_value_for_info(self, value, type_name):
        """Format value for info string based on its type"""
        if isinstance(value, torch.Tensor):
            shape = list(value.shape)
            if len(shape) == 3:  # Typical image tensor
                return f"<image tensor {shape[1]}x{shape[2]}>"
            return f"<tensor {shape}>"
        elif hasattr(value, 'audio_data'):  # Audio data
            return f"<audio {value.audio_data.shape[1]} samples>"
        elif isinstance(value, (list, tuple)):
            return f"[{len(value)} items]"
        elif isinstance(value, dict):
            return "{...}"
        elif isinstance(value, float):
            return f"{value:.4f}"
        elif isinstance(value, (int, bool)):
            return str(value)
        return str(value)

    def _format_value_for_output(self, value, is_string_output=False):
        """Format value based on whether it's going to a string output"""
        if is_string_output:
            return self._format_value_for_info(value, self._get_clean_type(value))
        return value

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
        
        # Handle values output based on context
        if len(filtered_dict) == 1:
            values_out = values[0]  # Return single value directly
        else:
            # For multiple values, format tensors/audio if any output is connected to a string
            values_out = []
            for v in values:
                if isinstance(v, (torch.Tensor, dict)) or hasattr(v, 'audio_data'):
                    values_out.append(self._format_value_for_info(v, self._get_clean_type(v)))
                else:
                    values_out.append(v)
            
        # Create formatted info string with key-value pairs
        delimiter = delimiter.replace('\\n', '\n')
        info_pairs = [
            f"{k}: {self._format_value_for_info(v['value'], v['type'])}" 
            for k, v in filtered_dict.items()
        ]
        info_str = delimiter.join(info_pairs)
        
        return (complete_dict, filtered_dict, keys_str, values_out, info_str, types,)

def smart_parse_string_to_dict(input_string):
    """
    Intelligently parse a string of key-value pairs into a dictionary.
    Handles various formats and delimiters.
    """
    # Remove curly braces if present
    cleaned = input_string.strip()
    if cleaned.startswith('{') and cleaned.endswith('}'):
        cleaned = cleaned[1:-1]
    
    # Try different common delimiters
    for pair_delimiter in [',', '|', ';', '\n']:
        # Skip if delimiter isn't in string
        if pair_delimiter not in cleaned:
            continue
            
        pairs = cleaned.split(pair_delimiter)
        result = {}
        
        for pair in pairs:
            # Skip empty pairs
            if not pair.strip():
                continue
                
            # Try different key-value separators
            for kv_separator in [':', '=', '=>', '->']:
                if kv_separator in pair:
                    key, value = pair.split(kv_separator, 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to parse the value
                    try:
                        # Handle lists
                        if value.startswith('[') and value.endswith(']'):
                            # Parse simple lists like ['4', 0] or [1, 2, 3]
                            value = eval(value)
                        # Handle numbers
                        elif value.replace('.', '').isdigit():
                            value = float(value) if '.' in value else int(value)
                        # Handle booleans
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                    except:
                        # Keep as string if parsing fails
                        pass
                    
                    result[key] = value
                    break
        
        # If we found at least one valid key-value pair, return the result
        if result:
            return result
    
    # If no common delimiters worked, try to parse as a single key-value pair
    for kv_separator in [':', '=', '=>', '->']:
        if kv_separator in cleaned:
            key, value = cleaned.split(kv_separator, 1)
            return {key.strip(): value.strip()}
    
    # If all parsing attempts fail, return empty dict
    return {}

class StringToDict:
    """
    A node that converts a string containing key-value pairs into a dictionary.
    Handles various formats and delimiters intelligently.
    """
    RETURN_TYPES = ("DICT",)
    FUNCTION = "parse"
    CATEGORY = "utils"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "key1: value1, key2: value2",
                    "tooltip": "String containing key-value pairs. Supports various formats and delimiters."
                }),
            }
        }
    
    def parse(self, input_string):
        result = smart_parse_string_to_dict(input_string)
        return (result,)

class DictToOutputs:
    """
    A node that outputs dictionary values through configurable outputs.
    Keys can be specified to select specific dictionary values, or left blank for sequential values.
    """
    DEFAULT_COUNT = 4
    RETURN_TYPES = ("DICT",) + ("*",) * DEFAULT_COUNT
    RETURN_NAMES = ("dictionary",) + tuple(f"value_{i+1}" for i in range(DEFAULT_COUNT))
    FUNCTION = "process"
    CATEGORY = "utils"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "dictionary": ("DICT", {}),
            }
        }
        
        # Add dynamic key inputs
        for i in range(s.DEFAULT_COUNT):
            inputs["required"][f"key_{i+1}"] = ("STRING", {"default": ""})
            
        # Add output count at the end
        inputs["required"]["output_count"] = ("INT", {
            "default": s.DEFAULT_COUNT,
            "min": 1, 
            "max": 50,
            "step": 1,
        })
        
        return inputs
    
    def process(self, dictionary, output_count, **kwargs):
        # Update return types and input types if output count changed
        if output_count != len(self.RETURN_TYPES) - 1:
            self.__class__.RETURN_TYPES = ("DICT",) + ("*",) * output_count
            self.__class__.RETURN_NAMES = ("dictionary",) + tuple(f"value_{i+1}" for i in range(output_count))
            self.__class__.DEFAULT_COUNT = output_count
        
        # Get values based on dictionary type
        if all(isinstance(v, dict) and "type" in v and "value" in v 
               for v in dictionary.values()):
            # For our special dict format
            values_dict = {k: v["value"] for k, v in dictionary.items()}
        else:
            values_dict = dictionary
            
        # Get the values based on keys or sequential keys
        output_values = []
        keys = [kwargs.get(f"key_{i+1}", "") for i in range(output_count)]
        dict_keys = list(values_dict.keys())
        
        for i in range(output_count):
            if keys[i]:  # If we have a key specified
                value = values_dict.get(keys[i], None)  # Get named value or None
            else:  # Otherwise get sequential value
                value = values_dict[dict_keys[i]] if i < len(dict_keys) else None
            output_values.append(value)
        
        # Return dictionary first, then all values
        return (dictionary,) + tuple(output_values)