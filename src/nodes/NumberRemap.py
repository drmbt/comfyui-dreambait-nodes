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
    
    @classmethod
    def IS_CHANGED(self, number=0.0, pre_multiply=1.000, override_number=None, from_range_min=0.0, from_range_max=1.0, to_range_min=0.0, to_range_max=1.0, clamp_min=None, clamp_max=None):
        m = hashlib.sha256()
        dummy = str(float(pre_multiply))
        m.update(dummy.encode("utf-8"))
        return {"pre_multiply": pre_multiply, "override_number": override_number, "dummy": m.digest().hex()}

    def number_operation(self, number=0.0, pre_multiply=1.000, override_number=None, from_range_min=0.0, from_range_max=1.0, to_range_min=0.0, to_range_max=1.0, clamp_min=None, clamp_max=None):
        if override_number or pre_multiply is not None:
            change_dict = self.IS_CHANGED(number, pre_multiply, override_number, from_range_min, from_range_max, to_range_min, to_range_max)
            override = change_dict['override_number']
            mult = float(change_dict['pre_multiply'])
            if override is not None:
                number = float(override) * mult

        output_number = (number*pre_multiply - from_range_min) / (from_range_max - from_range_min) * (to_range_max - to_range_min) + to_range_min
        if clamp_min is not None:
            output_number = max(output_number, clamp_min)
        if clamp_max is not None:
            output_number = min(output_number, clamp_max)
        float_string = f"{output_number:.4f}"
        bool = output_number>0
        return {
            "ui": {
                "output_number": [output_number],
                "float_string": [float_string],   
                "bool": [bool]
            },        
            "result": (output_number, float_string, bool)
        }
