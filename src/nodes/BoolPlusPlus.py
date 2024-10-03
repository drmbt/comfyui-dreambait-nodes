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
    @classmethod
    def IS_CHANGED(self, value=False, on_false="0.0", on_true="1.0", override_bool= None, override_on_false= None, override_on_true= None):
        m = hashlib.sha256()
        dummy = "dummy"
        m.update(dummy.encode("utf-8"))
        return {"override_bool": override_bool, "override_on_false":override_on_false, "override_on_true": override_on_true,  "dummy": m.digest().hex()}
    
    def boolean_operation(self, value=False, on_false="0.0", on_true="1.0", override_bool= None, override_on_false= None, override_on_true= None):
        boolean = value 
        
        if override_bool is not None or override_on_false is not None or override_on_true is not None:
            change_dict = self.IS_CHANGED(value, on_false, on_true, override_bool, override_on_false, override_on_true)
            
            if change_dict['override_bool'] is not None:

                if isinstance(change_dict['override_bool'], str):
                    if change_dict['override_bool'] in ["False", "false", "0", "0.0", ""]:
                        boolean = False
                    else: 
                        try:
                            boolean = bool(change_dict['override_bool'])
                        except:
                            try:
                                boolean = bool(float(change_dict['override_bool']))
                            except:
                                boolean = True
                else:
                    try:
                        boolean = bool(change_dict['override_bool'])
                    except:
                        boolean = value
                
            if change_dict['override_on_false'] is not None:
                try:
                    on_false = change_dict['override_on_false']
                except:
                    pass
            if change_dict['override_on_true'] is not None:
                try:
                    on_true = change_dict['override_on_true']
                except:
                    pass
        output_any = boolean
        output_bool = boolean
        if override_bool is not None and isinstance(override_bool, (bool, int, float)):
            boolean = bool(override_bool)
        elif isinstance(boolean, str):
            if boolean in ["False", "false", "0", "0.0", ""]:
                boolean = False
            else:
                boolean = True
        if boolean is False:
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
                if on_false in ["False", "false", "0", "0.0", ""]:
                    output_bool = False
                    output_int = 0
                    output_float = 0.0
                    output_string = str(output_bool)
                else:
                    output_bool = True
                    try: 
                        output_float = int(on_false) 
                    except: 
                        output_float = 1.0
                    output_int = int(output_float)
                    output_string = str(on_false)
            else:
                output_int = int(output_bool)
                output_float = float(output_int)
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
