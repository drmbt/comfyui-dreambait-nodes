"""
DreamBait Draw Text Node

Originally forked from ComfyUI_Comfyroll_CustomNodes Draw Text node to fix alignment labeling
and implement true justified text alignment. Subsequently enhanced with features from other
ComfyUI text nodes (comfyui_essentials, ComfyUI-LayerStyle) to create a comprehensive
text rendering solution.

Features have been broken out into helper nodes to declutter the interface and allow for more
granular control:
- Text Margins: Control individual margin settings (left, right, top, bottom)
- Text Box Style: Configure background box with padding, borders, and corner rounding
- Text Shadow: Set shadow parameters including distance, angle, blur and color

Core Features:
- Proper text justification with word spacing
- Smart hyphenation support for multiple languages
- Percentage-based positioning with pixel offsets
- Drop shadows with angle, distance, and blur
- Full RGBA color support with hex codes
- Text rotation with selectable pivot point
- Transparent backgrounds
- Image compositing with alpha
- Outputs both rendered image and text mask

Credits:
- ComfyUI_Comfyroll_CustomNodes (base implementation)
- comfyui_essentials (mask handling, tensor operations)
- ComfyUI-LayerStyle (shadow implementation concepts)
"""

import torch
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import torchvision.transforms.v2 as T
import pyphen  # Simple import, requirements.txt will handle installation
import numpy as np

'''
this node forked from ComfyUI_Comfyroll_CustomNodes draw_text, but corrects mislabled 
justify and vertical align options, implements justify and properly, adds hyphenation
'''



# Constants from original
COLORS = ["custom", "transparent", "white", "black", "red", "green", "blue", "yellow",
          "cyan", "magenta", "orange", "purple", "pink", "brown", "gray",
          "lightgray", "darkgray", "olive", "lime", "teal", "navy", "maroon",
          "fuchsia", "aqua", "silver", "gold", "turquoise", "lavender",
          "violet", "coral", "indigo"]

color_mapping = {
    "transparent": (0, 0, 0),
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
HORIZONTAL_ALIGN_OPTIONS = ["center", "left", "right"]
ROTATE_OPTIONS = ["text center", "image center"]

# Add to existing constants
LANGUAGES = ["en_US", "en_GB", "de_DE", "fr_FR", "es_ES", "it_IT"]  # Add more as needed

POSITION_MODES = ["pixels", "fraction"]

# Update tensor conversion functions
def tensor2pil(image):
    return T.ToPILImage()(image.permute([0,3,1,2])[0]).convert('RGBA')

def pil2tensor(image):
    return T.ToTensor()(image).unsqueeze(0).permute([0,2,3,1])

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
    if len(hex_color) == 8:  # RGBA
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(hex_color[6:8], 16) / 255.0  # Convert to float 0-1
        return (r, g, b), a
    else:  # RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b), None

def get_color_values(color, color_hex, opacity, color_mapping):
    """Get RGB color and opacity (0-1)"""
    if color == "custom":
        rgb, hex_opacity = hex_to_rgb(color_hex)
        opacity = hex_opacity if hex_opacity is not None else opacity
    else:
        rgb = color_mapping.get(color, (0, 0, 0))
        if color == "transparent":
            opacity = 0
    return rgb, opacity

def try_hyphenate_word(dic, word, draw, font, remaining_width):
    """Attempt to hyphenate a word to fit the remaining width."""
    if not dic:
        return None
        
    # Get possible hyphenation points
    hyphenated = dic.inserted(word)
    if not hyphenated:
        return None
        
    # Split at each hyphenation point and test
    parts = hyphenated.split('=')
    for i in range(len(parts)-1):
        first_part = ''.join(parts[:i+1]) + '-'
        first_width, _ = get_text_size(draw, first_part, font)
        if first_width <= remaining_width:
            second_part = ''.join(parts[i+1:])
            return (first_part, second_part)
    
    return None

def wrap_text(draw, text, font, max_width, margins, language="en_US"):
    """Smart word wrap with hyphenation that returns lines of text that fit within max_width."""
    try:
        dic = pyphen.Pyphen(lang=language)
    except:
        dic = None  # Fallback to no hyphenation if language not supported
        
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    space_width, _ = get_text_size(draw, " ", font)
    
    for word in words:
        word_width, _ = get_text_size(draw, word, font)
        
        # If word fits on current line
        if not current_line or current_width + word_width + space_width <= max_width - (2 * margins):
            current_line.append(word)
            current_width += word_width + (space_width if current_line else 0)
            continue
            
        # Try hyphenation if word doesn't fit
        if word_width > max_width - (2 * margins) - (current_width if current_line else 0):
            remaining_width = max_width - (2 * margins) - (current_width if current_line else 0)
            hyphenated = try_hyphenate_word(dic, word, draw, font, remaining_width)
            
            if hyphenated:
                first_part, second_part = hyphenated
                if current_line:
                    current_line.append(first_part)
                    lines.append(" ".join(current_line))
                else:
                    lines.append(first_part)
                current_line = [second_part]
                current_width = get_text_size(draw, second_part, font)[0]
                continue
        
        # If we get here, add current line and start new one
        if current_line:
            lines.append(" ".join(current_line))
        current_line = [word]
        current_width = word_width
    
    # Add the last line if there's anything left
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines

def draw_masked_text(text_mask, text, font_name, font_size,
                    margins, line_spacing, position_x, position_y,
                    vertical_align, horizontal_align, rotation_angle, rotation_options,
                    text_color=(255,255,255,255), justify=False):
    
    draw = ImageDraw.Draw(text_mask)
    
    # Track text boundaries
    text_bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]
    
    font_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fonts")
    font_file = os.path.join(font_folder, font_name)
    font = ImageFont.truetype(str(font_file), size=font_size)

    # Get image dimensions
    image_width, image_height = text_mask.size
    
    # Calculate effective width for text wrapping (accounting for margins)
    effective_width = image_width - margins["left"] - margins["right"]
    
    # Split text into paragraphs and wrap each paragraph
    paragraphs = text.split('\n')
    wrapped_lines = []
    for paragraph in paragraphs:
        if paragraph.strip():  # Only process non-empty paragraphs
            wrapped = wrap_text(draw, paragraph, font, effective_width, 0)  # Pass 0 for margins since we handle them separately
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
    
    # Calculate text area boundaries including margins
    text_area_x = margins["left"]
    text_area_width = image_width - margins["left"] - margins["right"]
    text_area_y = margins["top"]
    text_area_height = image_height - margins["top"] - margins["bottom"]

    # Draw each line
    text_pos_y = position_y + margins["top"]  # Add top margin to starting position
    sum_text_plot_y = 0
    text_height = max_text_height * len(wrapped_lines)

    # Adjust vertical alignment to respect margins
    if vertical_align == "center":
        text_plot_y = text_area_y + (text_area_height / 2 - text_height / 2) + text_pos_y
    elif vertical_align == "top":
        text_plot_y = text_area_y + text_pos_y
    else:  # bottom
        text_plot_y = text_area_y + text_area_height - text_height + text_pos_y

    # Draw each line
    for i, line in enumerate(wrapped_lines):
        words = line.split()
        line_width, line_height = get_text_size(draw, line, font)
        
        # Handle justified text differently
        if (justify and 
            i < len(wrapped_lines) - 1 and  # Not last line
            len(words) > 1 and
            line.strip()):
            
            # Calculate total width of words
            word_widths = [get_text_size(draw, word, font)[0] for word in words]
            total_word_width = sum(word_widths)
            
            # Calculate space to distribute within margins
            available_width = text_area_width
            total_spacing = available_width - total_word_width
            space_between = total_spacing / (len(words) - 1)
            
            # Draw each word with calculated spacing
            x = text_area_x + position_x
            for j, (word, word_width) in enumerate(zip(words, word_widths)):
                word_x = x
                word_y = text_plot_y
                draw.text((word_x, word_y), word, fill=text_color, font=font)
                
                # Update bounds
                bbox = draw.textbbox((word_x, word_y), word, font=font)
                text_bounds[0] = min(text_bounds[0], bbox[0])
                text_bounds[1] = min(text_bounds[1], bbox[1])
                text_bounds[2] = max(text_bounds[2], bbox[2])
                text_bounds[3] = max(text_bounds[3], bbox[3])
                
                if j < len(words) - 1:
                    x += word_width + space_between
        else:
            # Handle other alignments within margins
            if horizontal_align == "left":
                text_plot_x = text_area_x + position_x
            elif horizontal_align == "right":
                text_plot_x = text_area_x + text_area_width - line_width + position_x
            else:  # center
                text_plot_x = text_area_x + (text_area_width - line_width) / 2 + position_x

            if line.strip():
                draw.text((text_plot_x, text_plot_y), line, fill=text_color, font=font)
                
                # Update bounds
                bbox = draw.textbbox((text_plot_x, text_plot_y), line, font=font)
                text_bounds[0] = min(text_bounds[0], bbox[0])
                text_bounds[1] = min(text_bounds[1], bbox[1])
                text_bounds[2] = max(text_bounds[2], bbox[2])
                text_bounds[3] = max(text_bounds[3], bbox[3])

        text_plot_y += max_text_height
        sum_text_plot_y += text_plot_y

    # Handle rotation
    text_center_x = image_width / 2
    text_center_y = sum_text_plot_y / len(wrapped_lines)

    if rotation_options == "text center":
        rotated_text_mask = text_mask.rotate(rotation_angle, center=(text_center_x, text_center_y))
    else:  # image center
        rotated_text_mask = text_mask.rotate(rotation_angle, center=(image_width / 2, image_height / 2))
        
    text_mask = rotated_text_mask
    
    # Rotate bounds if needed
    if text_bounds != [float('inf'), float('inf'), float('-inf'), float('-inf')]:
        # Convert bounds to points
        points = [
            (text_bounds[0], text_bounds[1]),  # Top-left
            (text_bounds[2], text_bounds[1]),  # Top-right
            (text_bounds[2], text_bounds[3]),  # Bottom-right
            (text_bounds[0], text_bounds[3]),  # Bottom-left
        ]
        
        # Rotate points
        center = (image_width / 2, image_height / 2) if rotation_options == "image center" else (
            (text_bounds[0] + text_bounds[2]) / 2,
            (text_bounds[1] + text_bounds[3]) / 2
        )
        
        angle_rad = np.radians(rotation_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotated_points = []
        for x, y in points:
            dx = x - center[0]
            dy = y - center[1]
            rx = center[0] + dx * cos_a - dy * sin_a
            ry = center[1] + dx * sin_a + dy * cos_a
            rotated_points.append((rx, ry))
        
        # Update bounds to encompass rotated points
        text_bounds = [
            min(p[0] for p in rotated_points),
            min(p[1] for p in rotated_points),
            max(p[0] for p in rotated_points),
            max(p[1] for p in rotated_points)
        ]

    return text_mask, text_bounds

class DrawText:
    @classmethod
    def INPUT_TYPES(s):
        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fonts")
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]
                      
        return {"required": {
                    "text": ("STRING", {
                        "multiline": True, 
                        "default": "text",
                        "tooltip": "Text to render. Supports multiple lines"
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
                        "default": "GothamMedium.ttf",
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
    FUNCTION = "draw_text"
    CATEGORY = "DreamBait/Text"

    def draw_text(self, image_width, image_height, text,
                  font_name, font_size, font_color, font_opacity,
                  background_color, background_opacity,
                  vertical_align, horizontal_align,
                  justify, x_percent, y_percent, x_offset, y_offset,
                  line_spacing, rotation_angle, rotation_options,
                  language="en_US",
                  margins=None, textbox=None, shadow=None,
                  font_color_hex='#000000', bg_color_hex='#000000',
                  img_composite=None):
        
        # Get batch size from input image if provided
        batch_size = 1
        if img_composite is not None:
            batch_size = img_composite.shape[0]
        
        # Calculate final position from percentage and offset
        pixel_x = int((x_percent / 100.0) * image_width) + x_offset
        pixel_y = int((y_percent / 100.0) * image_height) + y_offset

        # Get colors with opacity
        text_rgb, text_opacity = get_color_values(font_color, font_color_hex, font_opacity, color_mapping)
        bg_rgb, bg_opacity = get_color_values(background_color, bg_color_hex, background_opacity, color_mapping)
        
        # Convert RGB + opacity to RGBA for PIL
        text_color = text_rgb + (int(text_opacity * 255),)
        bg_color = bg_rgb + (int(bg_opacity * 255),)

        # Get image size from the first image in batch if img_composite provided
        if img_composite is not None:
            # Convert first image in batch to determine dimensions
            first_img = tensor2pil(img_composite[0:1])
            image_width, image_height = first_img.size
        
        size = (image_width, image_height)
        
        # Get default margins if not provided
        if margins is None:
            margins = {"left": 0, "right": 0, "top": 0, "bottom": 0}

        # Create text and shadow images - only once for the batch
        text_image = Image.new('RGBA', size)
        text_mask = Image.new('RGBA', size)
        
        # Create a temporary transparent background for mask composition
        mask_background = Image.new('RGBA', size, (0,0,0,0))

        # Get text bounds first
        temp_mask = Image.new('RGBA', size)
        _, text_bounds = draw_masked_text(temp_mask, text, font_name, font_size,
                                        margins, line_spacing,
                                        pixel_x, pixel_y,
                                        vertical_align, horizontal_align,
                                        rotation_angle, rotation_options,
                                        text_color, justify)

        # Pre-calculate all the text-related elements (box, shadow, text) only once
        text_elements = Image.new('RGBA', size, (0,0,0,0))
        
        # Handle text box if provided
        if textbox is not None:
            # Get text box colors with opacity
            box_rgb, box_opacity = get_color_values(textbox["color"], textbox["color_hex"], textbox["opacity"], color_mapping)
            border_rgb, border_opacity = get_color_values(textbox["border_color"], textbox["border_hex"], textbox["border_opacity"], color_mapping)
            
            # Convert to RGBA
            box_color = box_rgb + (int(box_opacity * 255),)
            border_color = border_rgb + (int(border_opacity * 255),)
            
            # Add padding to text bounds
            text_bounds = [
                text_bounds[0] - textbox["padding"]["left"],
                text_bounds[1] - textbox["padding"]["top"],
                text_bounds[2] + textbox["padding"]["right"],
                text_bounds[3] + textbox["padding"]["bottom"]
            ]

            # Handle full width option
            if textbox["full_width"]:
                if horizontal_align == "left":
                    text_bounds[2] = image_width - margins["right"]
                elif horizontal_align == "right":
                    text_bounds[0] = margins["left"]
                else:  # center or justify
                    text_bounds[0] = margins["left"]
                    text_bounds[2] = image_width - margins["right"]

            # Draw text box
            if box_opacity > 0 or (textbox["border_width"] > 0 and border_opacity > 0):
                box = draw_rounded_rectangle(
                    text_elements, box_color, border_color,
                    textbox["border_width"], textbox["corner_radius"],
                    text_bounds
                )
                text_elements = Image.alpha_composite(text_elements, box)
                mask_background = Image.alpha_composite(mask_background, box)

        # Handle shadow if provided
        if shadow is not None and shadow["distance"] > 0:
            # Get shadow color with opacity
            shadow_rgb, shadow_opacity = get_color_values(shadow["color"], shadow["color_hex"], shadow["opacity"], color_mapping)
            shadow_color = shadow_rgb + (int(shadow_opacity * 255),)
            
            # Calculate shadow offset
            shadow_x = int(shadow["distance"] * np.cos(np.radians(shadow["angle"] - rotation_angle)))
            shadow_y = int(shadow["distance"] * np.sin(np.radians(shadow["angle"] - rotation_angle)))

            # Create and draw shadow
            shadow_mask = Image.new('RGBA', size)
            shadow_image, _ = draw_masked_text(shadow_mask, text, font_name, font_size,
                                             margins, line_spacing,
                                             pixel_x + shadow_x, 
                                             pixel_y + shadow_y,
                                             vertical_align, horizontal_align,
                                             rotation_angle, rotation_options,
                                             shadow_color, justify)
            
            if shadow["blur"] > 0:
                shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(shadow["blur"]))
            
            text_elements = Image.alpha_composite(text_elements, shadow_image)
            mask_background = Image.alpha_composite(mask_background, shadow_image)

        # Draw main text
        text_mask, _ = draw_masked_text(text_mask, text, font_name, font_size,
                                      margins, line_spacing,
                                      pixel_x, pixel_y,
                                      vertical_align, horizontal_align,
                                      rotation_angle, rotation_options,
                                      text_color, justify)
        
        text_elements = Image.alpha_composite(text_elements, text_mask)
        mask_background = Image.alpha_composite(mask_background, text_mask)
        
        # Extract mask from complete composition
        mask = mask_background.split()[3]
        
        # Convert the pre-calculated text elements to tensor for later compositing
        text_elements_tensor = pil2tensor(text_elements)[0]  # Remove the batch dimension
        
        # Initialize output tensors
        output_images = []
        output_masks = []
        
        # Create base mask tensor
        base_mask_tensor = T.ToTensor()(mask)  # Shape: [1, H, W]
        
        if batch_size > 1:
            # Process each image in the batch
            for i in range(batch_size):
                if img_composite is not None:
                    # Get the current image from the batch
                    current_img = tensor2pil(img_composite[i:i+1])
                else:
                    # Create a background if no image is provided
                    current_img = Image.new('RGBA', size, bg_color)
                
                # Convert to tensor
                current_img_tensor = pil2tensor(current_img)[0]  # Remove batch dimension
                
                # Add the text elements using torch operations for efficiency
                # First ensure consistent format
                if current_img_tensor.shape[-1] == 3:  # RGB
                    # Convert RGB to RGBA by adding alpha channel
                    alpha = torch.ones((current_img_tensor.shape[0], current_img_tensor.shape[1], 1), 
                                      device=current_img_tensor.device)
                    current_img_tensor = torch.cat([current_img_tensor, alpha], dim=-1)
                
                # Alpha composite the tensors
                # Extract alpha channels
                img_rgb = current_img_tensor[..., :3]
                img_a = current_img_tensor[..., 3:4]
                text_rgb = text_elements_tensor[..., :3]
                text_a = text_elements_tensor[..., 3:4]
                
                # Calculate composite
                out_a = text_a + img_a * (1.0 - text_a)
                out_rgb = (text_rgb * text_a + img_rgb * img_a * (1.0 - text_a)) / (out_a + 1e-8)
                
                # Combine channels
                out_rgba = torch.cat([out_rgb, out_a], dim=-1)
                output_images.append(out_rgba)
                
                # Add mask to output masks
                output_masks.append(base_mask_tensor)
            
            # Stack all processed images into a batch - shape [B, H, W, C]
            image_tensor = torch.stack(output_images, dim=0)
            # Only keep RGB channels (remove alpha)
            image_tensor = image_tensor[..., :3]
            # Stack all masks into a batch - shape [B, 1, H, W]
            mask_tensor = torch.stack(output_masks, dim=0).squeeze(1)  # Remove channel dimension
            
        else:
            # Handle single image case
            if img_composite is not None:
                back_image = tensor2pil(img_composite[0])
            else:
                back_image = Image.new('RGBA', size, bg_color)
                
            # Composite text onto background
            image_out = Image.alpha_composite(back_image, text_elements)
            
            # Convert to tensors
            image_tensor = pil2tensor(image_out)  # Shape: [1, H, W, C]
            # Only keep RGB channels (remove alpha)
            image_tensor = image_tensor[..., :3]
            mask_tensor = base_mask_tensor.unsqueeze(0)  # Shape: [1, 1, H, W]
            mask_tensor = mask_tensor.squeeze(1)  # Shape: [1, H, W] - correct MASK format

        return (image_tensor, mask_tensor,)

def draw_rounded_rectangle(image, color, border_color, border_width, corner_radius, bbox, anti_aliasing=2):
    """Draw a rounded rectangle on the image"""
    from PIL import ImageDraw
    
    # Create mask for filled rectangle
    fill_mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(fill_mask)
    
    x1, y1, x2, y2 = bbox
    radius = corner_radius
    
    # Draw filled rounded rectangle
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=255)
    
    # Create box image
    box = Image.new('RGBA', image.size, (0,0,0,0))
    
    # Draw filled box
    box_filled = Image.new('RGBA', image.size, color)
    box.paste(box_filled, mask=fill_mask)
    
    # Draw border if needed
    if border_width > 0 and border_color[3] > 0:
        # Create outer and inner masks for border
        outer_mask = Image.new('L', image.size, 0)
        inner_mask = Image.new('L', image.size, 0)
        
        draw_outer = ImageDraw.Draw(outer_mask)
        draw_inner = ImageDraw.Draw(inner_mask)
        
        # Draw outer and inner rounded rectangles
        draw_outer.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=255)
        draw_inner.rounded_rectangle([x1 + border_width, y1 + border_width, 
                                    x2 - border_width, y2 - border_width], 
                                   radius=max(0, radius - border_width), fill=255)
        
        # Create border mask by subtracting inner from outer
        border_mask = ImageChops.subtract(outer_mask, inner_mask)
        
        # Apply border
        border = Image.new('RGBA', image.size, border_color)
        box = Image.alpha_composite(box, Image.composite(border, box, border_mask))
    
    return box 

class TextMargins:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "left": ("INT", {
                "default": 24, "min": -4096, "max": 4096,
                "tooltip": "Left margin in pixels (negative values move text outside bounds)"
            }),
            "right": ("INT", {
                "default": 24, "min": -4096, "max": 4096,
                "tooltip": "Right margin in pixels (negative values move text outside bounds)"
            }),
            "top": ("INT", {
                "default": 24, "min": -4096, "max": 4096,
                "tooltip": "Top margin in pixels (negative values move text outside bounds)"
            }),
            "bottom": ("INT", {
                "default": 24, "min": -4096, "max": 4096,
                "tooltip": "Bottom margin in pixels (negative values move text outside bounds)"
            }),
        }}
    
    RETURN_TYPES = ("MARGINS",)
    FUNCTION = "get_margins"
    CATEGORY = "DreamBait/Text"

    def get_margins(self, left, right, top, bottom):
        return ({"left": left, "right": right, "top": top, "bottom": bottom},)

class TextBoxStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "color": (COLORS, {
                "default": "black",
                "tooltip": "Box fill color. Alpha controlled by opacity unless using 8-digit hex"
            }),
            "opacity": ("FLOAT", {
                "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Box opacity (ignored if using 8-digit hex color)"
            }),
            "border_color": (COLORS, {
                "default": "black",
                "tooltip": "Border color. Alpha controlled by border_opacity unless using 8-digit hex"
            }),
            "border_opacity": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Border opacity (ignored if using 8-digit hex color)"
            }),
            "border_width": ("INT", {
                "default": 6, "min": 0, "max": 100,
                "tooltip": "Border width in pixels"
            }),
            "corner_radius": ("INT", {
                "default": 0, "min": 0, "max": 100,
                "tooltip": "Corner radius in pixels"
            }),
            "padding_left": ("INT", {
                "default": 24, "min": -4096, "max": 4096,
                "tooltip": "Left padding in pixels (negative values shrink the box)"
            }),
            "padding_right": ("INT", {
                "default": 24, "min": -4096, "max": 4096,
                "tooltip": "Right padding in pixels (negative values shrink the box)"
            }),
            "padding_top": ("INT", {
                "default": 24, "min": -4096, "max": 4096,
                "tooltip": "Top padding in pixels (negative values shrink the box)"
            }),
            "padding_bottom": ("INT", {
                "default": 24, "min": -4096, "max": 4096,
                "tooltip": "Bottom padding in pixels (negative values shrink the box)"
            }),
            "full_width": ("BOOLEAN", {
                "default": False,
                "tooltip": "Extend box to margins based on text alignment"
            }),
        },
        "optional": {
            "color_hex": ("STRING", {
                "multiline": False,
                "default": "#FFFFFF",
                "tooltip": "Custom hex color (6 or 8 digits). 8-digit hex overrides opacity"
            }),
            "border_hex": ("STRING", {
                "multiline": False,
                "default": "#000000",
                "tooltip": "Custom border hex color (6 or 8 digits). 8-digit hex overrides border_opacity"
            }),
        }}
    
    RETURN_TYPES = ("TEXTBOX",)
    FUNCTION = "get_textbox"
    CATEGORY = "DreamBait/Text"

    def get_textbox(self, color, opacity, border_color, border_opacity, border_width,
                   corner_radius, padding_left, padding_right, padding_top, padding_bottom,
                   full_width, color_hex="#FFFFFF", border_hex="#000000"):
        return ({"color": color, "opacity": opacity, 
                "border_color": border_color, "border_opacity": border_opacity,
                "border_width": border_width, "corner_radius": corner_radius,
                "padding": {
                    "left": padding_left, "right": padding_right,
                    "top": padding_top, "bottom": padding_bottom
                },
                "full_width": full_width,
                "color_hex": color_hex, "border_hex": border_hex},)

class TextShadow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "color": (COLORS, {
                "default": "black",
                "tooltip": "Shadow color. Alpha controlled by opacity unless using 8-digit hex"
            }),
            "opacity": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Shadow opacity (ignored if using 8-digit hex color)"
            }),
            "distance": ("INT", {
                "default": 10, "min": 0, "max": 100, "step": 1,
                "tooltip": "Shadow distance in pixels. 0 disables shadow"
            }),
            "angle": ("FLOAT", {
                "default": 45.0, "min": -360.0, "max": 360.0, "step": 0.1,
                "tooltip": "Shadow angle in degrees"
            }),
            "blur": ("INT", {
                "default": 4, "min": 0, "max": 100, "step": 1,
                "tooltip": "Shadow blur radius in pixels"
            }),
        },
        "optional": {
            "color_hex": ("STRING", {
                "multiline": False,
                "default": "#000000",
                "tooltip": "Custom hex color (6 or 8 digits). 8-digit hex overrides opacity"
            }),
        }}
    
    RETURN_TYPES = ("SHADOW",)
    FUNCTION = "get_shadow"
    CATEGORY = "DreamBait/Text"

    def get_shadow(self, color, opacity, distance, angle, blur, color_hex="#000000"):
        return ({"color": color, "opacity": opacity, "distance": distance,
                "angle": angle, "blur": blur, "color_hex": color_hex},) 