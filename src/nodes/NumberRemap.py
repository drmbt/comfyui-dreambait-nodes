import math
from .utils import any_typ
import hashlib
class NumberRemap:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min":-1000000000000000,
                        "step": .001,
                        "tooltip": "Main number parameter"
                    },
                ),
                "from_range_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "step": .001
                    },
                ),
                "from_range_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": .001
                    },
                ),
                "to_range_min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "step": .001
                    },
                ),
                "to_range_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "step": .001
                    },
                ),
            },
             "optional": {
                "override_number": (
                    any_typ, 
                    {
                        "default": None,
                        "defaultInput": True,
                        "tooltip": "override any type float int or legal float wrappable string value to number"
                    },
                    
                ),
                "pre_multiply": (
                    any_typ, 
                    {
                        "default": 1.000,
                        "defaultInput": True,
                        "tooltip": "multiply by an optional number"
                    },
                    
                ),
                "clamp_min": (
                    any_typ, 
                    {
                        "default": None,
                        "defaultInput": True,
                        "tooltip": "optional_clamp"
                    },
                    
                ),
                "clamp_max": (
                    any_typ, 
                    {
                        "default": None,
                        "defaultInput": True,
                        "tooltip": "multiply by an optional number"
                    },
                    
                ),
            }
        }
    RETURN_TYPES = ("FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("output_number", "float_string", "bool")
    OUTPUT_NODE = True
    FUNCTION = "number_operation"
    CATEGORY = "DRMBT nodes"
    DESCRIPTION = """A node that remaps a number from one range to another.
    Supports optional clamping, overriding the number, and pre-multiplying the number."""
    
    @classmethod
    def IS_CHANGED(cls, value, old_min, old_max, new_min, new_max):
        # Use default ComfyUI behavior for input change detection
        return hash((value, old_min, old_max, new_min, new_max))

    def number_operation(self, number=0.0, pre_multiply=1.000, override_number=None, from_range_min=0.0, from_range_max=1.0, to_range_min=0.0, to_range_max=1.0, clamp_min=None, clamp_max=None):
        if override_number is not None:
            number = float(override_number)
        
        number *= float(pre_multiply)
        
        output_number = (number - from_range_min) / (from_range_max - from_range_min) * (to_range_max - to_range_min) + to_range_min
        
        if clamp_min is not None:
            output_number = max(float(clamp_min), output_number)
        if clamp_max is not None:
            output_number = min(float(clamp_max), output_number)
            
        float_string = f"{output_number:.4f}"
        bool_value = output_number > 0
        
        return {
            "ui": {
                "output_number": [output_number],
                "float_string": [float_string],   
                "bool": [bool_value]
            },        
            "result": (output_number, float_string, bool_value)
        }
