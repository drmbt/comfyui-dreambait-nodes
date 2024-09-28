import os
import comfy.samplers
import comfy.sample
import torch
from comfy.utils import ProgressBar
from .utils import FONTS_DIR, parse_string_to_list
from .utils import AnyType
import torchvision.transforms.v2 as T
import torch.nn.functional as F
import logging
import folder_paths

any = AnyType("*")

class PlotParametersDRMBT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),                   
                    "order_by": (["none", "time", "seed", "steps", "denoise", "sampler", "scheduler", "guidance", "max_shift", "base_shift", "lora_strength"], ),
                    "cols_value": (["none", "time", "seed", "steps", "denoise", "sampler", "scheduler", "guidance", "max_shift", "base_shift", "lora_strength"], ),
                    "cols_num": ("INT", {"default": -1, "min": -1, "max": 1024 }),
                    "add_prompt": (["false", "true", "excerpt"], ),
                    "add_params": (["false", "true", "changes only"], {"default": "true"}),
                },
                "optional": {
                    "params": ("SAMPLER_PARAMS", { "default": None }),
                    "text": (any, { "default": "" }),
                },
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, images, order_by, cols_value, cols_num, add_prompt, add_params, text=None, params=None):
        from PIL import Image, ImageDraw, ImageFont
        import math
        import textwrap
        import ast

        if params is None:
            try:
                text_dict = ast.literal_eval(text)
                if isinstance(text_dict, dict):
                    # Remove keys with None values
                    text_dict = {k: v for k, v in text_dict.items() if v is not None}
                    params = [text_dict for _ in range(images.shape[0])]
                else:
                    raise ValueError
            except (ValueError, SyntaxError):
                params = [{"text": str(text)} for _ in range(images.shape[0])]
                for p in params:
                    keys_to_remove = list(p.keys())
                    for key in keys_to_remove:
                        if key != "text":
                            del p[key]

        if images.shape[0] != len(params):
            raise ValueError("Number of images and number of parameters do not match.")

        _params = params.copy()

        if order_by != "none":
            sorted_params = sorted(_params, key=lambda x: x[order_by])
            indices = [_params.index(item) for item in sorted_params]
            images = images[torch.tensor(indices)]
            _params = sorted_params

        if cols_value != "none" and cols_num > -1:
            groups = {}
            for p in _params:
                value = p[cols_value]
                if value not in groups:
                    groups[value] = []
                groups[value].append(p)
            cols_num = len(groups)

            sorted_params = []
            groups = list(groups.values())
            for g in zip(*groups):
                sorted_params.extend(g)

            indices = [_params.index(item) for item in sorted_params]
            images = images[torch.tensor(indices)]
            _params = sorted_params
        elif cols_num == 0:
            cols_num = int(math.sqrt(images.shape[0]))
            cols_num = max(1, min(cols_num, 1024))

        width = images.shape[2]
        out_image = []

        font = ImageFont.truetype(os.path.join(FONTS_DIR, 'ShareTechMono-Regular.ttf'), min(48, int(32*(width/1024))))
        text_padding = 3
        line_height = font.getmask('Q').getbbox()[3] + font.getmetrics()[1] + text_padding*2
        char_width = font.getbbox('M')[2]+1 # using monospace font

        for (image, param) in zip(images, _params):
            image = image.permute(2, 0, 1)

            if isinstance(param, dict):
                text = "\n".join([f"{key}: {value}" for key, value in param.items()])
            else:
                text = str(param['text'])

            lines = text.split("\n")
            text_height = line_height * len(lines)
            text_image = Image.new('RGB', (width, text_height), color=(0, 0, 0))

            for i, line in enumerate(lines):
                draw = ImageDraw.Draw(text_image)
                draw.text((text_padding, i * line_height + text_padding), line, font=font, fill=(255, 255, 255))

            text_image = T.ToTensor()(text_image).to(image.device)
            image = torch.cat([image, text_image], 1)

            # a little cleanup
            image = torch.nan_to_num(image, nan=0.0).clamp(0.0, 1.0)
            out_image.append(image)

        # ensure all images have the same height
        max_height = max([image.shape[1] for image in out_image])
        out_image = [F.pad(image, (0, 0, 0, max_height - image.shape[1])) for image in out_image]

        out_image = torch.stack(out_image, 0).permute(0, 2, 3, 1)

        # merge images
        if cols_num > -1:
            cols = min(cols_num, out_image.shape[0])
            b, h, w, c = out_image.shape
            rows = math.ceil(b / cols)

            # Pad the tensor if necessary
            if b % cols != 0:
                padding = cols - (b % cols)
                out_image = F.pad(out_image, (0, 0, 0, 0, 0, 0, 0, padding))
                b = out_image.shape[0]

            # Reshape and transpose
            out_image = out_image.reshape(rows, cols, h, w, c)
            out_image = out_image.permute(0, 2, 1, 3, 4)
            out_image = out_image.reshape(rows * h, cols * w, c).unsqueeze(0)

            """
            width = out_image.shape[2]
            # add the title and notes on top
            if title and export_labels:
                title_font = ImageFont.truetype(os.path.join(FONTS_DIR, 'ShareTechMono-Regular.ttf'), 48)
                title_width = title_font.getbbox(title)[2]
                title_padding = 6
                title_line_height = title_font.getmask(title).getbbox()[3] + title_font.getmetrics()[1] + title_padding*2
                title_text_height = title_line_height
                title_text_image = Image.new('RGB', (width, title_text_height), color=(0, 0, 0, 0))

                draw = ImageDraw.Draw(title_text_image)
                draw.text((width//2 - title_width//2, title_padding), title, font=title_font, fill=(255, 255, 255))

                title_text_image = T.ToTensor()(title_text_image).unsqueeze(0).permute([0,2,3,1]).to(out_image.device)
                out_image = torch.cat([title_text_image, out_image], 1)
            """

        return (out_image, )