import math
from .utils import any_typ
import hashlib
class BoolPlusPlus:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "on_false": (
                    "STRING",
                    {
                        "default": "0.0",
                        "tooltip": "return value if false"
                    },
                ),
                "on_true": (
                    "STRING",
                    {
                        "default": "1.0",
                        "tooltip": "return value if true"
                    },
                )
            },
            "optional": {
                "override_bool": (
                    any_typ, 
                    {
                        "default": None,
                        "defaultInput": True,
                        "tooltip": "override any type float int or legal float wrappable string value to number"
                    },
                ),
                "override_on_false": (
                    any_typ, 
                    {
                        "default": None,
                        "defaultInput": True,
                        "tooltip": "override any type float int or legal float wrappable string value"
                    },
                ),
                "override_on_true": (
                    any_typ, 
                    {
                        "default": None,
                        "defaultInput": True,
                        "tooltip": "override any type float int or legal float wrappable string value"
                    },
                ),
                
            }
        }
    RETURN_TYPES = ("BOOLEAN", "STRING", "INT", "FLOAT", "STRING", any_typ)
    RETURN_NAMES = ("BOOL", "BOOL_STRING", "INT", "FLOAT", "STRING", "ANY")
    OUTPUT_NODE = True
    FUNCTION = "boolean_operation"
    CATEGORY = "DRMBT nodes"
    DESCRIPTION = """
    Returns a boolean value based on the input value and optional overrides, as well as type conversions and boolean conditional outputs.
    """
    def boolean_operation(self, value=False, on_false="0.0", on_true="1.0", override_bool=None, override_on_false=None, override_on_true=None):
        boolean = value
        
        if override_bool is not None:
            if isinstance(override_bool, str):
                if override_bool in ["False", "false", "0", "0.0", ""]:
                    boolean = False
                else:
                    try:
                        boolean = bool(override_bool)
                    except:
                        try:
                            boolean = bool(float(override_bool))
                        except:
                            boolean = True
            else:
                try:
                    boolean = bool(override_bool)
                except:
                    boolean = value

        if override_on_false is not None:
            on_false = override_on_false
            
        if override_on_true is not None:
            on_true = override_on_true

        output_any = boolean
        output_bool = boolean
        if override_bool is not None and isinstance(override_bool, (bool, int, float)):
            boolean = bool(override_bool)
        elif isinstance(boolean, str):
            if boolean in ["False", "false", "0", "0.0", ""]:
                boolean = False
            else:
                boolean = True

        output_bool = boolean
                
        if boolean is False:
            output_bool = False  # Ensure bool output is always False when boolean is False
            if isinstance(on_false, (bool, int, float)):
                output_int = int(on_false)
                output_any = bool(on_false)
                try:
                    output_float = float(on_false)
                    output_string = str(bool(output_int))
                except:
                    output_float = float(output_int)
                    output_string = str(bool(output_int))
            elif isinstance(on_false, str):
                # Keep output_bool as False, but process other outputs normally
                output_int = 0 if on_false in ["False", "false", "0", "0.0", ""] else 1
                output_float = float(output_int)
                output_string = str(on_false)
            else:
                output_int = 0  # Default to 0 for unsupported types
                output_float = 0.0
                output_string = str(output_bool)
                output_any = on_false
        elif boolean is True:
            if isinstance(on_true, (bool, int, float)) and value is False:
                output_int = int(on_true)
                output_any = bool(on_false)
                try:
                    output_float = float(on_true)
                    output_string = str(bool(output_int))
                except:
                    output_float = float(output_int)
                    output_string = str(bool(output_int))
            elif isinstance(on_true, str):
                if on_true in ["False", "false", "0", "0.0", ""]:
                    output_bool = False
                    output_int = 0
                    output_float = 0.0
                    output_string = str(output_bool)
                else:
                    output_bool = True
                    try: 
                        output_float = int(on_true) 
                    except: 
                        output_float = 1.0
                    output_int = int(output_float)
                    output_string = str(on_true)

            else:
                output_int = int(output_bool)
                output_float = float(output_int)
                output_string = str(on_true)
                output_any = on_true

        return {
            "ui": {
                "BOOL": [output_bool],
                "BOOL_STRING": [str(output_bool)],  # Ensure these are lists
                "INT": [output_int],
                "FLOAT": [output_float],                  # Ensure these are lists
                "STRING": [output_string],    # Ensure these are lists
                "ANY": [output_any]
            },
            "result": (output_bool, str(output_bool), output_int, output_float, output_string, output_any)
        }
