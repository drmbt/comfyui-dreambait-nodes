from .utils import any_typ

class DRMBT_MultiMinMax:
    ### forked from impact pack###
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

    RETURN_TYPES = (any_typ, )

    def doit(self, mode, a, b, **optional):
        inputs = [x for x in [a, b] + list(optional.values()) if x is not None]
        if mode:
            return (max(inputs), )
        else:
            return (min(inputs), )