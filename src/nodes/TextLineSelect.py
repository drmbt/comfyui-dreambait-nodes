import sys
import logging
import random
from typing import Optional

# Initialize logger
logger = logging.getLogger(__name__)

class TextLineSelect:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {        
                "text": ("STRING", {"forceInput": True}),
                "select": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": sys.maxsize,  # No practical upper limit
                    "step": 1,
                    "tooltip": "Select which line to output (cycles through available lines). This number loops with a modulo operator."
                }),
                "random": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, selects a random line instead of using the select index"
                }),     
            },
            "hidden": {},
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("text", "text_list", "count", "selected")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "display_text"
    OUTPUT_NODE = True
    CATEGORY = "Text Tools"
    DESCRIPTION = "Display text with optional line selection"
    
    @classmethod
    def IS_CHANGED(cls, text: Optional[str], select: int, random: bool) -> bool:
        return True  # Always update when random is enabled to ensure new random selections
    
    def display_text(self, text: Optional[str], select: int, random: bool):
        if text is None:
            logger.error("Received None for text input in display_text.")
            return {
                "ui": {"text": [""]},
                "result": ("", [], 0, "")
            }
            
        # Split text into lines and filter out empty lines
        text_list = [line.strip() for line in text.split('\n') if line.strip()]
        count = len(text_list)
        
        # Handle empty text or no valid lines
        if count == 0:
            return {
                "ui": {"text": [text]},
                "result": (text, [], 0, text)
            }
        
        # Select either random or indexed line
        if random:
            select_index = random.randrange(count)
        else:
            select_index = select % count if count > 0 else 0
            
        selected = text_list[select_index]
        
        # Return both UI update and results
        return {
            "ui": {"text": [selected]},  # Show the selected line in the UI
            "result": (
                text,        # complete text
                text_list,   # list of individual lines as separate string outputs
                count,       # number of lines
                selected    # selected line based on select input
            )
        }



