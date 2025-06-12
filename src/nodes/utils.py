import torch
import numpy as np
import scipy
import os
#import re
from pathlib import Path
import folder_paths
import ast  # Add this at the top with other imports

FONTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")

SCRIPT_DIR = Path(__file__).parent

class MultiInput(str):
    def __new__(cls, string, allowed_types="*"):
        res = super().__new__(cls, string)
        res.allowed_types = allowed_types
        return res
    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types

floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])

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
    
    def __init__(self):
        self.unique_id = None  # Will be set by ComfyUI
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "key_lookup": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "New key names (space, comma, or newline separated)",
                    "tooltip": "Rename keys in order. Leave empty to keep original names."
                }),
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

    def _flatten_dict(self, d, parent_key='', node_id=None):
        """
        Recursively flatten a dictionary, handling both regular dicts and typed dicts.
        Now includes node_id for key uniqueness.
        """
        items = {}
        key_counts = {}
        
        def get_unique_key(base_key):
            full_key = self._get_input_key(base_key, node_id)
            if full_key not in key_counts:
                key_counts[full_key] = 1
                return base_key  # Return clean key for first instance
            key_counts[full_key] += 1
            return f"{base_key}{key_counts[full_key]}"  # Add number for duplicates
        
        for k, v in d.items():
            new_key = f"{parent_key}{k}" if parent_key else k
            new_key = get_unique_key(new_key)
            
            # If it's our special dict format with type/value
            if isinstance(v, dict) and "type" in v and "value" in v:
                if isinstance(v["value"], dict):
                    # Recursively flatten the inner dictionary
                    items.update(self._flatten_dict(v["value"], f"{new_key}_", node_id))
                else:
                    items[new_key] = {
                        "type": self._get_clean_type(v["value"]),
                        "value": v["value"]
                    }
            
            # If it's a regular dictionary
            elif isinstance(v, dict):
                items.update(self._flatten_dict(v, f"{new_key}_", node_id))
            
            # If it's a regular value
            else:
                items[new_key] = {
                    "type": self._get_clean_type(v),
                    "value": v
                }
                
        return items

    def _get_input_key(self, base_key, node_id):
        """Create a unique key based on input name and source node ID"""
        return f"{base_key}_{node_id}" if node_id else base_key

    def combine(self, key_lookup="", keys="*", delimiter=", ", **kwargs):
        complete_dict = {}
        key_counts = {}
        
        # Extract node IDs from input metadata if available
        input_ids = getattr(self, 'input_ids', {})
        
        def get_unique_key(base_key, node_id=None):
            full_key = self._get_input_key(base_key, node_id)
            if full_key not in key_counts:
                key_counts[full_key] = 1
                return base_key  # Return clean key for first instance
            key_counts[full_key] += 1
            return f"{base_key}{key_counts[full_key]}"  # Add number for duplicates
        
        for k, v in kwargs.items():
            node_id = input_ids.get(k)  # Get source node ID for this input
            if isinstance(v, dict):
                # Pass node ID to flatten_dict
                flattened = self._flatten_dict(v, node_id=node_id)
                complete_dict.update(flattened)
            else:
                unique_k = get_unique_key(k, node_id)
                complete_dict[unique_k] = {
                    "type": self._get_clean_type(v),
                    "value": v
                }

        # Handle key renaming with similar numbering scheme
        if key_lookup.strip():
            new_keys = parse_keys(key_lookup)
            if new_keys:
                original_keys = list(complete_dict.keys())
                renamed_dict = {}
                rename_counts = {}  # Track count of each new key for uniqueness
                
                def get_unique_renamed_key(base_key):
                    if base_key not in rename_counts:
                        rename_counts[base_key] = 1
                        return base_key
                    rename_counts[base_key] += 1
                    return f"{base_key}{rename_counts[base_key]}"  # Now using number without underscore
                
                # Handle explicitly renamed keys
                for i, new_key in enumerate(new_keys):
                    if i < len(original_keys):
                        unique_new_key = get_unique_renamed_key(new_key)
                        renamed_dict[unique_new_key] = complete_dict[original_keys[i]]
                
                # Add remaining unmapped keys
                for old_key in original_keys[len(new_keys):]:
                    renamed_dict[old_key] = complete_dict[old_key]
                
                complete_dict = renamed_dict

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
    # First try to evaluate as a proper dictionary if it looks like one
    cleaned = input_string.strip()
    if cleaned.startswith('{') and cleaned.endswith('}'):
        try:
            # Use ast.literal_eval which safely evaluates strings containing Python expressions
            return ast.literal_eval(cleaned)
        except:
            # If literal_eval fails, remove braces and continue with string parsing
            cleaned = cleaned[1:-1]
    
    # Try different common delimiters for string format
    for pair_delimiter in [',', '|', ';', '\n']:
        # Skip if delimiter isn't in string
        if pair_delimiter not in cleaned:
            continue
            
        try:
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
                                value = eval(value, {"__builtins__": {}}, {})
                            # Handle quoted strings
                            elif (value.startswith('"') and value.endswith('"')) or \
                                 (value.startswith("'") and value.endswith("'")):
                                value = value[1:-1]
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
        except:
            # If parsing with this delimiter fails, try the next one
            continue
    
    # If no common delimiters worked, try to parse as a single key-value pair
    for kv_separator in [':', '=', '=>', '->']:
        if kv_separator in cleaned:
            try:
                key, value = cleaned.split(kv_separator, 1)
                return {key.strip(): value.strip()}
            except:
                continue
    
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

class DynamicStringConcatenate:
    """
    A node that concatenates multiple string inputs using a configurable delimiter.
    Accepts an arbitrary number of string inputs and combines them with the specified delimiter.
    """
    OUTPUT_IS_LIST = (False, True)
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("concatenated_string", "list_strings",)
    FUNCTION = "concatenate"
    CATEGORY = "utils"
    DESCRIPTION = "Concatenates multiple string inputs with a configurable delimiter. Supports dynamic input count and smart delimiter parsing."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "delimiter": ("STRING", {
                    "default": ", ",
                    "multiline": False,
                    "placeholder": "Delimiter for concatenation (e.g., ', ' or '\\n')",
                    "tooltip": "Character(s) to separate the input strings. Use \\n for newlines. Leave empty for newlines. Default is ', '."
                }),
                "skip_empty": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, empty or whitespace-only strings will be skipped during concatenation."
                }),
                "trim_whitespace": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, leading and trailing whitespace will be removed from each input string before concatenation."
                })
            }
        }

    def concatenate(self, delimiter=", ", skip_empty=True, trim_whitespace=True, **kwargs):
        import re
        print(f"[DynamicStringConcatenate] kwargs: {kwargs}")
        # Accept both STRING1, STRING2, ... and string, string_1, string_2, ...
        def extract_index(key):
            if re.match(r"^STRING\d+$", key):
                return int(key[6:])
            elif key == "string":
                return 1
            elif re.match(r"^string_\d+$", key):
                return int(key[7:]) + 1
            return 9999
        # Collect all valid keys
        string_keys = [k for k in kwargs.keys() if re.match(r"^STRING\d+$", k) or k == "string" or re.match(r"^string_\d+$", k)]
        string_keys.sort(key=extract_index)
        string_inputs = []
        for key in string_keys:
            value = kwargs[key]
            if value is not None:
                str_value = str(value)
                if trim_whitespace:
                    str_value = str_value.strip()
                if skip_empty and not str_value:
                    continue
                string_inputs.append(str_value)
        parsed_delimiter = delimiter.replace('\\n', '\n') if delimiter else '\n'
        result = parsed_delimiter.join(string_inputs)
        print(f"[DynamicStringConcatenate] result: {result!r}, string_inputs: {string_inputs!r}")
        return (result, string_inputs)