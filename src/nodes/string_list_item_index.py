class StringListItemIndex:
    def __init__(self):
        pass

    """
    A node to parse a string input, split the text by a delimiter, and find the index of the input string in the list.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING",),
                "delimiter": ("STRING", {"default": ","}),
                "text": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_string", "index")
    FUNCTION = "parse_and_find_index"
    CATEGORY = "String Operations"
    
    TOOLTIPS = {
        "input": { 
            "input_string": "The string to be matched against the list items.",
            "delimiter": "The delimiter used to split the text into a list.",
            "text": "The text to be split into a list by the delimiter.",
        },
        "output": ("The original input string and the index of the matching item in the list.",)
    }

    def parse_and_find_index(self, input_string, delimiter, text):
        # Split the text by the delimiter
        items = text.split(delimiter)
        
        # Attempt to find the index of the input_string in the list
        try:
            index = items.index(input_string)
        except ValueError:
            index = -1  # If the input_string is not found, return -1

        return (input_string, index)