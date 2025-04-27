class TextPlusPlus:
    """A node for text manipulation with prepend/append functionality"""
    
    DESCRIPTION = """Combines text with optional prepend, append and override operations."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepend": (
                    "STRING",
                    {
                        "default": "", 
                        "multiline": False, 
                        "tooltip": "prepend this text to the body parameter separated by the delimiter"
                    },
                ),
                "body": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "print_to_screen": True,
                    },
                ),
                "append": (
                    "STRING",
                    {
                        "default": "", 
                        "multiline": False, 
                        "tooltip": "append this text to the body parameter separated by the delimiter"
                    }
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
        }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "prepend", "body", "append")
    OUTPUT_NODE = True
    FUNCTION = "text_assemble"
    CATEGORY = "DRMBT nodes"

    @classmethod
    def IS_CHANGED(cls, body, prepend, append, delimiter):
        # Use a more reliable change detection method than hash
        return (prepend, body, append, delimiter)

    def text_assemble(self, body="", prepend="", append="", delimiter=" "):
        try:
            # Add handling of escape sequences
            delimiter = bytes(delimiter, "utf-8").decode("unicode_escape")
            
            if prepend != "":
                prepend = f"{prepend}{delimiter}"
            if append != "":
                append = f"{delimiter}{append}" 
            text = f"{prepend}{body}{append}"
            return {"ui": {"text": text, "prepend": prepend, "body": body, "append": append}, 
                   "result": (text, prepend, body, append)}
        except UnicodeError as e:
            raise ValueError(f"Invalid text or delimiter encoding: {str(e)}")