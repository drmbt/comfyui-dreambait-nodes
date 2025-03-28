"""
DreamBait Draw Mana Node

A specialized version of the Draw Text node that supports MTG mana symbols.
Forked from the Draw Text node to add mana symbol mapping functionality.

Features:
- All features from Draw Text node
- Automatic conversion of mana cost strings to proper unicode symbols
- Support for common MTG mana symbols (WUBRG, numbers, X, Y, Z, etc.)
- Maintains all text styling and positioning options
"""

import torch
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import torchvision.transforms.v2 as T
import pyphen
import numpy as np
import re
from .draw_text_drmbt import DrawText

# Mana symbol mapping dictionary
MANA_SYMBOLS = {
    # Basic mana colors
    'W': chr(0xe600),  # White
    'U': chr(0xe601),  # Blue
    'B': chr(0xe602),  # Black
    'R': chr(0xe603),  # Red
    'G': chr(0xe604),  # Green
    
    # Special white variants
    'W-ORIGINAL': chr(0xe997),  # White (Original)
    'W-LIST': chr(0xe998),      # White (List)
    
    # Other mana types
    'C': chr(0xe904),  # Colorless
    'H': chr(0xe618),  # Phyrexian
    'S': chr(0xe619),  # Snow
    'S-MTGA': chr(0xe996),  # Snow (MTGA)
    
    # Variable mana
    'X': chr(0xe615),  # X
    'Y': chr(0xe616),  # Y
    'Z': chr(0xe617),  # Z
    
    # Numbers 0-20
    '0': chr(0xe605), '1': chr(0xe606), '2': chr(0xe607),
    '3': chr(0xe608), '4': chr(0xe609), '5': chr(0xe60a),
    '6': chr(0xe60b), '7': chr(0xe60c), '8': chr(0xe60d),
    '9': chr(0xe60e), '10': chr(0xe60f), '11': chr(0xe610),
    '12': chr(0xe611), '13': chr(0xe612), '14': chr(0xe613),
    '15': chr(0xe614), '16': chr(0xe62a), '17': chr(0xe62b),
    '18': chr(0xe62c), '19': chr(0xe62d), '20': chr(0xe62e),
}

# Mana color mapping dictionary
MANA_COLORS = {
    # Basic mana colors
    'W': '#FFFFFF',  # White
    'U': '#0066CC',  # Blue
    'B': '#000000',  # Black
    'R': '#CC0000',  # Red
    'G': '#00CC00',  # Green
    
    # Special white variants
    'W-ORIGINAL': '#FFFFFF',  # White (Original)
    'W-LIST': '#FFFFFF',      # White (List)
    
    # Other mana types
    'C': '#CCCCCC',  # Colorless
    'H': '#CC0000',  # Phyrexian (Red)
    'S': '#FFFFFF',  # Snow (White)
    'S-MTGA': '#FFFFFF',  # Snow (MTGA)
    
    # Variable mana
    'X': '#CCCCCC',  # X (Colorless)
    'Y': '#CCCCCC',  # Y (Colorless)
    'Z': '#CCCCCC',  # Z (Colorless)
    
    # Numbers 0-20 (Colorless)
    '0': '#CCCCCC', '1': '#CCCCCC', '2': '#CCCCCC',
    '3': '#CCCCCC', '4': '#CCCCCC', '5': '#CCCCCC',
    '6': '#CCCCCC', '7': '#CCCCCC', '8': '#CCCCCC',
    '9': '#CCCCCC', '10': '#CCCCCC', '11': '#CCCCCC',
    '12': '#CCCCCC', '13': '#CCCCCC', '14': '#CCCCCC',
    '15': '#CCCCCC', '16': '#CCCCCC', '17': '#CCCCCC',
    '18': '#CCCCCC', '19': '#CCCCCC', '20': '#CCCCCC',
}

# Regular expression patterns for mana costs
MANA_PATTERNS = [
    (r'\{(\d+)\}', lambda m: MANA_SYMBOLS[m.group(1)]),  # {1}, {2}, etc.
    (r'\{([WUBRG])\}', lambda m: MANA_SYMBOLS[m.group(1)]),  # {W}, {U}, etc.
    (r'\{([WUBRG])\-ORIGINAL\}', lambda m: MANA_SYMBOLS[f"{m.group(1)}-ORIGINAL"]),  # {W-ORIGINAL}
    (r'\{([WUBRG])\-LIST\}', lambda m: MANA_SYMBOLS[f"{m.group(1)}-LIST"]),  # {W-LIST}
    (r'\{C\}', lambda m: MANA_SYMBOLS['C']),  # {C}
    (r'\{H\}', lambda m: MANA_SYMBOLS['H']),  # {H}
    (r'\{S\}', lambda m: MANA_SYMBOLS['S']),  # {S}
    (r'\{S-MTGA\}', lambda m: MANA_SYMBOLS['S-MTGA']),  # {S-MTGA}
    (r'\{X\}', lambda m: MANA_SYMBOLS['X']),  # {X}
    (r'\{Y\}', lambda m: MANA_SYMBOLS['Y']),  # {Y}
    (r'\{Z\}', lambda m: MANA_SYMBOLS['Z']),  # {Z}
]

def convert_mana_cost(text):
    """Convert mana cost string to unicode symbols."""
    result = text
    for pattern, replacement in MANA_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result

# Import all the necessary functions and classes from draw_text_drmbt
from .draw_text_drmbt import (
    COLORS, color_mapping, VERTICAL_ALIGN_OPTIONS, HORIZONTAL_ALIGN_OPTIONS,
    ROTATE_OPTIONS, LANGUAGES, POSITION_MODES, tensor2pil, pil2tensor,
    get_text_size, align_text, justify_text, hex_to_rgb, get_color_values,
    try_hyphenate_word, wrap_text, draw_masked_text, draw_rounded_rectangle,
    TextMargins, TextBoxStyle, TextShadow
)

class DrawMana:
    @classmethod
    def INPUT_TYPES(s):
        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]
                      
        return {"required": {
                    "text": ("STRING", {
                        "multiline": True, 
                        "default": "{W}{U}{B}{R}{G}",
                        "tooltip": "Text to render. Supports mana symbols in {W}{U}{B}{R}{G} format"
                    }),
                    "image_width": ("INT", {
                        "default": 512, "min": 64, "max": 2048,
                        "tooltip": "Width of output image. Ignored if img_composite is provided"
                    }),
                    "image_height": ("INT", {
                        "default": 512, "min": 64, "max": 2048,
                        "tooltip": "Height of output image. Ignored if img_composite is provided"
                    }),
                    "font_name": (file_list, {
                        "default": "mana.ttf",
                        "tooltip": "TTF font file to use from the fonts directory"
                    }),
                    "font_size": ("INT", {
                        "default": 54, "min": 1, "max": 1024,
                        "tooltip": "Font size in pixels"
                    }),
                    "font_color": (COLORS, {
                        "default": "white",
                        "tooltip": "Text color. Alpha controlled by opacity unless using 8-digit hex"
                    }),
                    "font_opacity": ("FLOAT", {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "Text opacity (ignored if using 8-digit hex color)"
                    }),
                    "background_color": (COLORS, {
                        "default": "black",
                        "tooltip": "Background color. Alpha controlled by opacity unless using 8-digit hex"
                    }),
                    "background_opacity": ("FLOAT", {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "Background opacity (ignored if using 8-digit hex color)"
                    }),
                    "vertical_align": (VERTICAL_ALIGN_OPTIONS, {
                        "default": "top"
                    }),
                    "horizontal_align": (HORIZONTAL_ALIGN_OPTIONS, {
                        "default": "left",
                        "tooltip": "Horizontal text alignment"
                    }),
                    "justify": ("BOOLEAN", {
                        "default": False,
                        "tooltip": "Enable full justification (except last line of paragraph)"
                    }),
                    "x_percent": ("FLOAT", {
                        "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                        "tooltip": "Horizontal position as percentage of image width"
                    }),
                    "y_percent": ("FLOAT", {
                        "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                        "tooltip": "Vertical position as percentage of image height"
                    }),
                    "x_offset": ("INT", {
                        "default": 0, "min": -4096, "max": 4096,
                        "tooltip": "Additional horizontal offset in pixels"
                    }),
                    "y_offset": ("INT", {
                        "default": 0, "min": -4096, "max": 4096,
                        "tooltip": "Additional vertical offset in pixels"
                    }),
                    "line_spacing": ("INT", {
                        "default": 6, "min": -1024, "max": 1024,
                        "tooltip": "Additional space between lines in pixels"
                    }),
                    "rotation_angle": ("FLOAT", {
                        "default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1,
                        "tooltip": "Text rotation angle in degrees"
                    }),
                    "rotation_options": (ROTATE_OPTIONS,),
                    "language": (LANGUAGES, {
                        "default": "en_US",
                        "tooltip": "Language for hyphenation rules"
                    }),
                },
                "optional": {
                    "font_color_hex": ("STRING", {
                        "multiline": False, 
                        "default": "#000000",
                        "tooltip": "Custom hex color (6 or 8 digits). 8-digit hex overrides opacity"
                    }),
                    "bg_color_hex": ("STRING", {
                        "multiline": False, 
                        "default": "#000000",
                        "tooltip": "Custom hex color (6 or 8 digits). 8-digit hex overrides opacity"
                    }),
                    "img_composite": ("IMAGE",),
                    "margins": ("MARGINS",),
                    "textbox": ("TEXTBOX",),
                    "shadow": ("SHADOW",),
                }          
    }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "draw_mana"
    CATEGORY = "DreamBait/Text"

    def draw_mana(self, image_width, image_height, text,
                  font_name, font_size, font_color, font_opacity,
                  background_color, background_opacity,
                  vertical_align, horizontal_align,
                  justify, x_percent, y_percent, x_offset, y_offset,
                  line_spacing, rotation_angle, rotation_options,
                  language="en_US",
                  margins=None, textbox=None, shadow=None,
                  font_color_hex='#000000', bg_color_hex='#000000',
                  img_composite=None):
        
        # Convert mana symbols in text
        converted_text = convert_mana_cost(text)
        
        # Call the parent class's draw_text method with converted text
        return DrawText().draw_text(
            image_width, image_height, converted_text,
            font_name, font_size, font_color, font_opacity,
            background_color, background_opacity,
            vertical_align, horizontal_align,
            justify, x_percent, y_percent, x_offset, y_offset,
            line_spacing, rotation_angle, rotation_options,
            language,
            margins, textbox, shadow,
            font_color_hex, bg_color_hex,
            img_composite
        )