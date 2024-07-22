import torch
from PIL import Image
import sys, os, time
import torch
import cv2
import numpy as np
import random
import gc
import torch
from torchvision import transforms
import imghdr

###########################################################################

# Import comfyUI modules:
from cli_args import args
import folder_paths

###########################################################################


class AspectPadImageForOutpainting:
    """
    A node to calculate args for default comfy node 'Pad Image For Outpainting'
    """
    ASPECT_RATIO_MAP = {
        "SDXL_1-1_square_1024x1024": (1024, 1024),
        "SDXL_4-3_landscape_1152x896": (1152, 896),
        "SDXL_3-2_landscape_1216x832": (1216, 832),
        "SDXL_16-9_landscape_1344x768": (1344, 768),
        "SDXL_21-9_landscape_1536x640": (1536, 640),
        "SDXL_3-4_portrait_896x1152": (896, 1152),
        "SDXL_5-8_portrait_832x1216": (832, 1216),
        "SDXL_9-16_portrait_768x1344": (768, 1344),
        "SDXL_9-21_portrait_640x1536": (640, 1536),
        "SD15_1-1_square_512x512": (512, 512),
        "SD15_2-3_portrait_512x768": (512, 768),
        "SD15_3-4_portrait_512x682": (512, 682),
        "SD15_3-2_landscape_768x512": (768, 512),
        "SD15_4-3_landscape_682x512": (682, 512),
        "SD15_16-9_cinema_910x512": (910, 512),
        "SD15_37-20_cinema_952x512": (952, 512),
        "SD15_2-1_cinema_1024x512": (1024, 512),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(s.ASPECT_RATIO_MAP.keys()), {"default": "SDXL_16-9_landscape_1344x768"}),
                "justification": (["top-left", "center", "bottom-right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE","LEFT","TOP","RIGHT","BOTTOM",)
    FUNCTION = "fit_and_calculate_padding"
    CATEGORY = "Eden 🌱/Image"

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