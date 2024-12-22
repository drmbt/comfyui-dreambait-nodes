class TextPlusPlus:
    """A node for text manipulation with prepend/append functionality"""
    
    DESCRIPTION = """Combines text with optional prepend, append and override operations."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "print_to_screen": True,
                    },
                ),
                "delimiter": (
                    "STRING",
                    {
                        "default": " ",
                        "multiline": False,
                        "forceInput": False,
                        "tooltip": "split the text fields by this delimiter"
                    },
                )
            },
            "optional": {
                "prepend": (
                    "STRING",
                    {
                        "default": "", 
                        "multiline": True, 
                        "defaultInput": True,
                        "tooltip": "prepend this text to the body parameter separated by the delimiter"
                    },
                ),
                "override_body": (
                    "STRING",
                    {
                        "default": "", 
                        "multiline": True, 
                        "defaultInput": True,
                        "tooltip": "replace all text in the body parameter with text from this input"
                    },
                ),
                "append": (
                    "STRING",
                    {
                        "default": "", 
                        "multiline": True, 
                        "defaultInput": True,
                        "tooltip": "append this text to the body parameter separated by the delimiter"
                    }
                )
            },
        }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "prepend", "body", "append")
    OUTPUT_NODE = True
    FUNCTION = "text_assemble"
    CATEGORY = "DRMBT nodes"

    @classmethod
    def IS_CHANGED(cls, prepend, body, override_body, append, delimiter):
        # All text inputs and the delimiter can affect the output
        return (prepend, body, override_body, append, delimiter)

    def text_assemble(self, prepend="", body="", override_body=None, append="", delimiter=" "):
        # Add handling of escape sequences
        delimiter = bytes(delimiter, "utf-8").decode("unicode_escape")
        
        if override_body is not None:
            body = override_body
        if prepend != "":
            prepend = f"{prepend}{delimiter}"
        if append != "":
            append = f"{delimiter}{append}" 
        text = f"{prepend}{body}{append}"
        return {"ui": {"text": text, "prepend": prepend, "body": body, "append": append}, "result": (text, prepend, body, append)}