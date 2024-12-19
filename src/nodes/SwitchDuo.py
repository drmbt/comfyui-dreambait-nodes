from .utils import any_typ

class SwitchDuo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
                "on_false1": (any_typ, {"defaultInput": True}),
                "on_true1": (any_typ, {"defaultInput": True}),
            },
            "optional": {
                "on_false2": (any_typ, {"defaultInput": True}),
                "on_true2": (any_typ, {"defaultInput": True}),
                "override_switch": (
                    any_typ,
                    {
                        "default": None,
                        "defaultInput": True,
                        "tooltip": "override any type bool, float int or legal float wrappable string value to number"
                    },
                )
            }
        }

    RETURN_TYPES = (any_typ, any_typ)
    RETURN_NAMES = ("OUTPUT1", "OUTPUT2")
    FUNCTION = "switch"
    CATEGORY = "DRMBT nodes"
    DESCRIPTION = """A node that switches between two outputs based on a boolean value.
    Supports overriding the boolean value with an optional input."""
    def switch(self, value=False, on_false1=None, on_true1=None, override_switch=None, on_false2=None, on_true2=None):
        if override_switch is not None:
            if isinstance(override_switch, str):
                value = override_switch.lower() not in ["false", "0", "0.0", "", "none"]
            elif isinstance(override_switch, (bool, int, float)):
                value = bool(override_switch)

        OUTPUT1 = on_true1 if value else on_false1
        OUTPUT2 = on_true2 if value else on_false2

        return OUTPUT1, OUTPUT2