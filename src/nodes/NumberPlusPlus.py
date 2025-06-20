import math
from .utils import any_typ
class NumberPlusPlus:
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
                        "step": .01,
                        "forceInput": False,
                        "tooltip": "Main number parameter"
                    },
                ),
                "integer": (
                    ["round", "floor", "ceiling"], 
                    {
                        "default": "round",
                        "tooltip": "rounding method"
                    }
                )
            },
            "optional": {
                "pre_add": (
                    any_typ,
                    {
                        "default": 0.0,
                        "step": .001,
                        "forceInput": True,
                        "tooltip": "pre-add any type float int or legal float wrappable string value to number"
                    },
                ),
                "override_number": (
                    any_typ, 
                    {
                        "default": None,
                        "step": .001,
                        "forceInput": True,
                        "tooltip": "override any type float int or legal float wrappable string value to number"
                    },
                ),
                "multiplier": (
                    any_typ,
                    {
                        "default": 1.0,
                        "step": .001,
                        "forceInput": True,
                        "tooltip": "multiply the main number"
                    },
                ),
                "post_add": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "step": .001,
                        "forceInput": True,
                        "tooltip": "add to value after multiplying"
                    },
                )
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    RETURN_TYPES = ("FLOAT", "INT", "BOOLEAN", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("output_number", "output_int", "bool>0", "float_string", "int_string", "pre_add", "multiplier", "post_add")
    OUTPUT_NODE = True
    FUNCTION = "number_operation"
    CATEGORY = "DRMBT nodes"
    DESCRIPTION = "A node to perform arithmetic operations on a number, including pre-addition, multiplication, and post-addition. It also provides formatted string outputs for the number, its integer representation, and a boolean indicating if the number is greater than 0."
    def number_operation(self, number=0.0, integer="round", pre_add=0.0, multiplier=1.0, post_add=0.0, override_number=None, unique_id=None, extra_pnginfo=None, display_number="0.00"):
        number = float(number)
        if override_number is not None:
            number = float(override_number)
        try:
            pre_add = float(pre_add)
        except ValueError:
            pre_add = 0.0
        try:
            multiplier = float(multiplier)
        except ValueError:
            multiplier = 1.0
        try:
            post_add = float(post_add)
        except ValueError:
            post_add = 0.0
        output_number = number
        output_number = float((pre_add + number) * multiplier + post_add)
        if integer == "round":
            output_int = int(round(output_number))
        elif integer == "floor":
            output_int = int(math.floor(output_number))
        elif integer == "ceiling":
            output_int = int(math.ceil(output_number))
        float_string = f"{output_number:.3f}"
        int_string = f"{output_int:.0f}"
        bool = output_number>0
        return (output_number, output_int, bool, float_string, int_string, pre_add, multiplier, post_add)
