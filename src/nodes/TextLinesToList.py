import logging
from typing import Optional

# Initialize logger
logger = logging.getLogger(__name__)

class TextLinesToList:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {        
                "text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_text",)
    FUNCTION = "process_text"
    CATEGORY = "Text Tools"
    DESCRIPTION = "Converts multi-line text into a string formatted as a Python list"
    
    def process_text(self, text: Optional[str]):
        if text is None:
            logger.error("Received None for text input in process_text.")
            return ("[]",)

        # Split text into lines and filter out empty lines
        text_list = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Format the list manually to ensure single quotes
        formatted_text = "[" + ", ".join(f'"{line}"' for line in text_list) + "]"
        
        return (formatted_text,) 