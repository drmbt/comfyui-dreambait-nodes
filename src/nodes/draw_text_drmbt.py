import numpy as np
import torch
import os
from PIL import Image, ImageDraw, ImageFont

# Constants from original
COLORS = ["custom", "white", "black", "red", "green", "blue", "yellow",
          "cyan", "magenta", "orange", "purple", "pink", "brown", "gray",
          "lightgray", "darkgray", "olive", "lime", "teal", "navy", "maroon",
          "fuchsia", "aqua", "silver", "gold", "turquoise", "lavender",
          "violet", "coral", "indigo"]

color_mapping = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (160, 85, 15),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (102, 102, 102),
    "olive": (128, 128, 0),
    "lime": (0, 128, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "fuchsia": (255, 0, 128),
    "aqua": (0, 255, 128),
    "silver": (192, 192, 192),
    "gold": (255, 215, 0),
    "turquoise": (64, 224, 208),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "coral": (255, 127, 80),
    "indigo": (75, 0, 130),    
}

VERTICAL_ALIGN_OPTIONS = ["center", "top", "bottom"]
HORIZONTAL_ALIGN_OPTIONS = ["center", "left", "right", "justify"]
PERSPECTIVE_OPTIONS = ["top", "bottom", "left", "right"]

# Helper functions from functions_graphics.py
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height

def align_text(align, img_height, text_height, text_pos_y, margins):
    if align == "center":
        text_plot_y = img_height / 2 - text_height / 2 + text_pos_y
    elif align == "top":
        text_plot_y = text_pos_y + margins                       
    elif align == "bottom":
        text_plot_y = img_height - text_height + text_pos_y - margins 
    return text_plot_y

def justify_text(justify, img_width, line_width, margins, text=None):
    if justify == "left":
        text_plot_x = 0 + margins
    elif justify == "right":
        text_plot_x = img_width - line_width - margins
    elif justify == "center":
        text_plot_x = img_width/2 - line_width/2
    elif justify == "justify" and text:
        text_plot_x = margins  # Return margins as default x position for justify
    else:
        text_plot_x = margins  # Default fallback
    return text_plot_x

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

def get_color_values(color, color_hex, color_mapping):
    if color == "custom":
        color_rgb = hex_to_rgb(color_hex)
    else:
        color_rgb = color_mapping.get(color, (0, 0, 0))
    return color_rgb

def wrap_text(draw, text, font, max_width, margins):
    """Smart word wrap that returns lines of text that fit within max_width."""
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        word_width, _ = get_text_size(draw, word, font)
        space_width, _ = get_text_size(draw, " ", font)
        
        # Check if adding this word would exceed the max width
        if current_line and current_width + word_width + space_width > max_width - (2 * margins):
            # Add the current line to lines and start a new line
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_width
        else:
            # Add word to current line
            current_line.append(word)
            current_width += word_width + (space_width if current_line else 0)
    
    # Add the last line if there's anything left
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines

def draw_masked_text(text_mask, text, font_name, font_size,
                    margins, line_spacing, position_x, position_y,
                    vertical_align, horizontal_align, rotation_angle, rotation_options):
    
    draw = ImageDraw.Draw(text_mask)
    
    font_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fonts")
    font_file = os.path.join(font_folder, font_name)
    font = ImageFont.truetype(str(font_file), size=font_size)

    # Get image dimensions
    image_width, image_height = text_mask.size
    
    # Split text into paragraphs and wrap each paragraph
    paragraphs = text.split('\n')
    wrapped_lines = []
    for paragraph in paragraphs:
        if paragraph.strip():  # Only process non-empty paragraphs
            wrapped = wrap_text(draw, paragraph, font, image_width, margins)
            wrapped_lines.extend(wrapped)
        else:
            wrapped_lines.append("")  # Preserve empty lines

    # Calculate maximum dimensions
    max_text_width = 0
    max_text_height = 0
    for line in wrapped_lines:
        line_width, line_height = get_text_size(draw, line, font)
        line_height = line_height + line_spacing
        max_text_width = max(max_text_width, line_width)
        max_text_height = max(max_text_height, line_height)
    
    image_center_x = image_width / 2
    image_center_y = image_height / 2

    text_pos_y = position_y
    sum_text_plot_y = 0
    text_height = max_text_height * len(wrapped_lines)
    last_text_x = margins

    # Draw each line
    for i, line in enumerate(wrapped_lines):
        words = line.split()
        line_width, _ = get_text_size(draw, line, font)
        
        text_plot_y = align_text(vertical_align, image_height, text_height, text_pos_y, margins)
        
        # Handle justified text differently
        if (horizontal_align == "justify" and 
            i < len(wrapped_lines) - 1 and  # Not last line
            len(words) > 1 and  # More than one word
            line.strip() and    # Not an empty line
            i < len(wrapped_lines) - 1 and  # Not the last line of the paragraph
            wrapped_lines[i + 1].strip()):  # Next line is not empty (paragraph break)
            
            # Calculate total width of words
            word_widths = [get_text_size(draw, word, font)[0] for word in words]
            total_word_width = sum(word_widths)
            
            # Calculate space to distribute
            available_width = image_width - 2 * margins
            total_spacing = available_width - total_word_width
            space_between = total_spacing / (len(words) - 1)
            
            # Draw each word with calculated spacing
            x = margins + position_x
            for j, (word, word_width) in enumerate(zip(words, word_widths)):
                draw.text((x, text_plot_y), word, fill=255, font=font)
                if j < len(words) - 1:  # Don't add space after last word
                    x += word_width + space_between
            last_text_x = x
        else:
            # Handle other alignments normally
            text_plot_x = position_x + justify_text(horizontal_align, image_width, line_width, margins)
            if line.strip():  # Only draw non-empty lines
                draw.text((text_plot_x, text_plot_y), line, fill=255, font=font)
            last_text_x = text_plot_x + line_width
        
        text_pos_y += max_text_height
        sum_text_plot_y += text_plot_y

    # Handle rotation
    text_center_x = image_width / 2
    text_center_y = sum_text_plot_y / len(wrapped_lines)

    if rotation_options == "text center":
        rotated_text_mask = text_mask.rotate(rotation_angle, center=(text_center_x, text_center_y))
    else:  # image center
        rotated_text_mask = text_mask.rotate(rotation_angle, center=(image_center_x, image_center_y))
        
    return rotated_text_mask

class DrawText:
    @classmethod
    def INPUT_TYPES(s):
        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]
                      
        return {"required": {
                    "image_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                    "image_height": ("INT", {"default": 512, "min": 64, "max": 2048}),  
                    "text": ("STRING", {"multiline": True, "default": "text"}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                    "font_color": (COLORS,),
                    "background_color": (COLORS,),
                    "vertical_align": (VERTICAL_ALIGN_OPTIONS,),
                    "horizontal_align": (HORIZONTAL_ALIGN_OPTIONS,),
                    "margins": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "line_spacing": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                    "rotation_options": (PERSPECTIVE_OPTIONS,),            
                },
                "optional": {
                    "font_color_hex": ("STRING", {"multiline": False, "default": "#000000"}),
                    "bg_color_hex": ("STRING", {"multiline": False, "default": "#000000"})
                }          
    }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_text"
    CATEGORY = "DreamBait/Text"

    def draw_text(self, image_width, image_height, text,
                  font_name, font_size, font_color, background_color,
                  margins, line_spacing, position_x, position_y,
                  vertical_align, horizontal_align, rotation_angle, rotation_options,
                  font_color_hex='#000000', bg_color_hex='#000000'):

        text_color = get_color_values(font_color, font_color_hex, color_mapping)
        bg_color = get_color_values(background_color, bg_color_hex, color_mapping)
        
        size = (image_width, image_height)
        text_image = Image.new('RGB', size, text_color)
        back_image = Image.new('RGB', size, bg_color)
        text_mask = Image.new('L', back_image.size)

        rotated_text_mask = draw_masked_text(text_mask, text, font_name, font_size,
                                           margins, line_spacing,
                                           position_x, position_y,
                                           vertical_align, horizontal_align,
                                           rotation_angle, rotation_options)

        image_out = Image.composite(text_image, back_image, rotated_text_mask)
        
        return (pil2tensor(image_out),) 