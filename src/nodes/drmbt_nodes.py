import torch
import os
import configparser

###########################################################################


class AspectPadImageForOutpainting:
    def __init__(self):
        pass
    
    """
    A node to calculate args for default comfy node 'Pad Image For Outpainting'
    """
    ASPECT_RATIO_MAP = {
        "1-1_square_1024x1024": (1024, 1024),
        "4-3_landscape_1152x896": (1152, 896),
        "3-2_landscape_1216x832": (1216, 832),
        "16-9_landscape_1344x768": (1344, 768),
        "21-9_landscape_1536x640": (1536, 640),
        "3-4_portrait_896x1152": (896, 1152),
        "2-3_portrait_832x1216": (832, 1216),
        "9-16_portrait_768x1344": (768, 1344),
        "9-21_portrait_640x1536": (640, 1536),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(s.ASPECT_RATIO_MAP.keys()), {"default": "16-9_landscape_1344x768"}),
                "justification": (["top-left", "center", "bottom-right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE","LEFT","TOP","RIGHT","BOTTOM",)
    FUNCTION = "fit_and_calculate_padding"
    CATEGORY = "DRMBT 💭🎣"

    def fit_and_calculate_padding(self, image, aspect_ratio, justification):
        bs, h, w, c = image.shape

        # Get the canvas dimensions from the aspect ratio map
        canvas_width, canvas_height = self.ASPECT_RATIO_MAP[aspect_ratio]

        # Calculate the aspect ratios
        image_aspect_ratio = w / h
        canvas_aspect_ratio = canvas_width / canvas_height

        # Determine the new dimensions
        if image_aspect_ratio > canvas_aspect_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / image_aspect_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * image_aspect_ratio)

        # Resize the image
        resized_image = torch.nn.functional.interpolate(image.permute(0, 3, 1, 2), size=(new_height, new_width), mode='bicubic', align_corners=False)
        resized_image = resized_image.permute(0, 2, 3, 1)

        # Calculate padding
        if justification == "center":
            left = (canvas_width - new_width) // 2
            right = canvas_width - new_width - left
            top = (canvas_height - new_height) // 2
            bottom = canvas_height - new_height - top
        elif justification == "top-left":
            left = 0
            right = canvas_width - new_width
            top = 0
            bottom = canvas_height - new_height
        elif justification == "bottom-right":
            left = canvas_width - new_width
            right = 0
            top = canvas_height - new_height
            bottom = 0

        return (resized_image, left, top, right, bottom)
    
class DRMBT_Model_Context_Selector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.ini')
        model_context_list = []

        if os.path.exists(config_path):
            config.read(config_path)
            if 'Model Context List Settings' in config and 'model_context' in config['Model Context List Settings']:
                model_context_list = config['Model Context List Settings']['model_context'].split(', ')
        
        if not model_context_list:
            model_context_list.append("None")
        
        return {
            "required": {
                "model_context": (model_context_list,),
            }
        }

    RETURN_TYPES = (str, int)
    RETURN_NAMES = ("selected_model_context", "selected_index")
    FUNCTION = "load_model_context"

    CATEGORY = "DRMBT Suite/Model"

    def load_model_context(self, model_context):
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.ini')
        model_context_list = []

        if os.path.exists(config_path):
            config.read(config_path)
            if 'Model Context List Settings' in config and 'model_context' in config['Model Context List Settings']:
                model_context_list = config['Model Context List Settings']['model_context'].split(', ')

        if model_context in model_context_list:
            index = model_context_list.index(model_context)
        else:
            model_context = ''
            index = -1

        return (model_context, index)
    
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