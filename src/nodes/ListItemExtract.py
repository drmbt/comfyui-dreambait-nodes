import logging
import ast
from typing import Any, Optional, Tuple, Union
import torch

# Initialize logger
logger = logging.getLogger(__name__)

class ListItemExtract:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("any",),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("any",)
    FUNCTION = "extract_item"
    CATEGORY = "List Operations"
    DESCRIPTION = "Extracts an item from a list at the specified index. Works with text, numbers, and images."

    def extract_item(self, list_input: Any, index: int) -> Tuple[Any]:
        try:
            # Handle tensor batch (multiple images)
            if isinstance(list_input, torch.Tensor) and len(list_input.shape) == 4:
                if index >= list_input.shape[0]:
                    logger.warning(f"Index {index} is out of range. Using last item.")
                    index = list_input.shape[0] - 1
                return (list_input[index:index+1],)  # Keep the batch dimension

            # If input is a string representation of a list, parse it
            if isinstance(list_input, str):
                try:
                    list_input = ast.literal_eval(list_input)
                except (ValueError, SyntaxError):
                    # If parsing fails, treat the string as a single item
                    list_input = [list_input]

            # Convert to list if not already
            if not isinstance(list_input, (list, tuple)):
                list_input = [list_input]

            # Check if index is valid
            if index >= len(list_input):
                logger.warning(f"Index {index} is out of range. Using last item.")
                index = len(list_input) - 1
            
            # Extract the item
            result = list_input[index]
            
            return (result,)

        except Exception as e:
            logger.error(f"Error extracting item from list: {str(e)}")
            return (None,) 