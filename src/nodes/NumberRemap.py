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
    def IS_CHANGED(cls, number=0.0, pre_multiply=1.000, override_number=None, from_range_min=0.0, from_range_max=1.0, to_range_min=0.0, to_range_max=1.0, clamp_min=None, clamp_max=None):
        # If override_number is set, only use that and ignore the main number
        main_value = str(override_number) if override_number is not None else str(round(float(number) * float(pre_multiply), 6))
        
        # Round the range values to reduce unnecessary updates
        ranges = [
            round(float(from_range_min), 6),
            round(float(from_range_max), 6),
            round(float(to_range_min), 6),
            round(float(to_range_max), 6)
        ]
        
        # Only include clamp values if they're set
        clamps = []
        if clamp_min is not None:
            clamps.append(round(float(clamp_min), 6))
        if clamp_max is not None:
            clamps.append(round(float(clamp_max), 6))
            
        # Combine all relevant values
        values = [main_value] + [str(x) for x in ranges] + [str(x) for x in clamps]
        combined = "_".join(values)
        
        # Create a hash of the inputs that matter
        m = hashlib.sha256()
        m.update(combined.encode('utf-8'))
        return m.digest().hex()

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
