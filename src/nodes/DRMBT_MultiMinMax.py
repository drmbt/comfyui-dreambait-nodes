from .utils import any_typ

class DRMBT_MultiMinMax:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ("BOOLEAN", {"default": True, "label_on": "max", "label_off": "min"}),
                "a": (any_typ, {}),
                "b": (any_typ, {}),
            },
            "optional": {
                "c": (any_typ, {"optional": True}),
                "d": (any_typ, {"optional": True}),
                "e": (any_typ, {"optional": True}),
                "f": (any_typ, {"optional": True}),
                "g": (any_typ, {"optional": True}),
                "h": (any_typ, {"optional": True}),
            }
        }

    FUNCTION = "doit"
    CATEGORY = "Math"
    DESCRIPTION = """A node that finds the minimum or maximum value among multiple inputs.
    Forked from Impact Pack's MinMax node, extended to support up to 8 inputs.
    Accepts any comparable type and returns the min/max value of the same type."""
    RETURN_TYPES = (any_typ, )

    def doit(self, mode, a, b, **optional):
        inputs = [x for x in [a, b] + list(optional.values()) if x is not None]
        if mode:
            return (max(inputs), )
        else:
            return (min(inputs), )