from PIL import Image, ImageOps
import os
import sys
import json
import piexif
import hashlib
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import requests
from io import BytesIO
import hashlib
import shutil
import cv2  # Add OpenCV for video processing
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS, IFD
from PIL.PngImagePlugin import PngImageFile
from PIL.JpegImagePlugin import JpegImageFile


class LoadMedia:
    """
    LoadMedia class for loading images, and videos as image sequences.

    This class provides functionality to load images from a file path, which can be a single image,
    a directory of images, or a video file (.mp4, .mov). It supports extracting frames from video files
    and treating them as image sequences.

    Disclaimer:
    There is a known issue with resizing mismatched resolution images. If using a path to a directory,
    make sure all images are the same size to avoid potential issues. Also not sure why I'm still getting 
    `WARNING: LoadMedia.IS_CHANGED() got an unexpected keyword argument 'image_load_cap'`
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"image_upload": True}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "PROMPT", "NEGATIVE", "WIDTH", "HEIGHT", "COUNT", "IMAGE_PATH")
    FUNCTION = "load_media"

    def load_media(self, path, image_load_cap=0, start_index=0):
        if os.path.isdir(path):
            return self.load_images_from_folder(path, image_load_cap, start_index)
        elif path.lower().endswith(('.mp4', '.mov')):
            return self.load_images_from_movie(path, image_load_cap)
        else:
            return self.load_image(path)

    def load_image(self, image):
        # Removes any quotes from Explorer
        image_path = os.path.normpath(str(image).replace('"', "")).replace("\\", "/")
        i = None
        if image_path.startswith("http"):
            response = requests.get(image_path)
            i = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            i = Image.open(image_path)
        prompt, negative, width, height = "", "", i.width, i.height

        if i.format == "PNG":
            if "parameters" in i.info:
                prompt, negative = handle_auto1111(i.info.get("parameters"))
            elif "negative_prompt" in i.info or "Negative Prompt" in i.info:
                prompt, negative = handle_ezdiff(str(i.info).replace("'", '"'))
            elif "sd-metadata" in i.info:
                prompt, negative = handle_invoke_modern(i.info)
            elif "Dream" in i.info:
                prompt, negative = handle_invoke_legacy(i.info)
            elif i.info.get("Software") == "NovelAI":
                prompt, negative = handle_novelai(i.info)
            elif "XML:com.adobe.xmp" in i.info:
                prompt, negative = handle_drawthings(i.info)

            # New code to handle the metadata
            if "metadata" in i.info:
                metadata = json.loads(i.info["metadata"])
                positive_text = ""
                for node_id, node_data in metadata.items():
                    if "positive" in node_data.get("inputs", {}):
                        positive_node = node_data["inputs"]["positive"][0]
                        if str(positive_node) in metadata:
                            positive_text = metadata[str(positive_node)].get("inputs", {}).get("text", "")
                            break
                if positive_text:
                    prompt = positive_text + " " + prompt

        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        if 'A' in i.getbands():
            mask = 1. - torch.from_numpy(np.array(i.getchannel('A')).astype(np.float32) / 255.0)
        return (image, mask, prompt, negative, width, height, 1, image_path)

    def load_images_from_folder(self, folder, image_load_cap, start_index):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")
        dir_files = os.listdir(folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No valid image files in directory '{folder}'.")

        dir_files = sorted(dir_files)
        dir_files = [os.path.normpath(os.path.join(folder, x)).replace("\\", "/") for x in dir_files][start_index:]

        # Apply image_load_cap
        if image_load_cap > 0:
            dir_files = dir_files[:image_load_cap]

        images, masks, image_path_list = [], [], []
        image_count, has_non_empty_mask = 0, False

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            width, height = i.width, i.height
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            if 'A' in i.getbands():
                mask = 1. - torch.from_numpy(np.array(i.getchannel('A')).astype(np.float32) / 255.0)
                has_non_empty_mask = True
            images.append((image, width, height))
            masks.append(mask)
            image_path_list.append(image_path)
            image_count += 1

        # Check if all images have the same dimensions
        base_width, base_height = images[0][1], images[0][2]
        all_same_size = all(img[1] == base_width and img[2] == base_height for img in images)

        if not all_same_size:
            tmp_folder = os.path.join(folder, "tmp_resized_images")
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            os.makedirs(tmp_folder, exist_ok=True)
            resized_image_paths = []

            for img, path in zip(images, image_path_list):
                resized_image_path = os.path.join(tmp_folder, os.path.basename(path))
                if img[1] != base_width or img[2] != base_height:
                    resized_image = self.resize_and_pad(img[0], base_width, base_height)
                    resized_image_pil = Image.fromarray((resized_image.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    resized_image_pil.save(resized_image_path)
                else:
                    shutil.copy(path, resized_image_path)
                resized_image_paths.append(resized_image_path)

            image_path_list = resized_image_paths
            dir_files = resized_image_paths

        # Reload images from the (possibly new) directory
        images, masks = [], []
        for image_path in dir_files:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            width, height = i.width, i.height
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            if 'A' in i.getbands():
                mask = 1. - torch.from_numpy(np.array(i.getchannel('A')).astype(np.float32) / 255.0)
                has_non_empty_mask = True
            images.append((image, width, height))
            masks.append(mask)

        if len(images) == 1:
            image, width, height = images[0]
            return (image, masks[0], "", "", width, height, 1, image_path_list[0])

        # Get the dimensions of the first image
        base_width, base_height = images[0][1], images[0][2]

        resized_images = []
        for img in images:
            try:
                resized_image = self.resize_and_pad(img[0], base_width, base_height)
                resized_images.append(resized_image)
            except Exception as e:
                print(f"Error resizing image: {img[0].shape}, {e}")
                resized_images.append(img[0])  # Add the original image if resizing fails

        if not resized_images:
            raise ValueError("No images were resized successfully.")

        image1 = torch.cat(resized_images, dim=0)

        mask1 = None
        for mask2 in masks[1:]:
            if has_non_empty_mask:
                if image1.shape[1:3] != mask2.shape:
                    mask2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(image1.shape[2], image1.shape[1]), mode='bilinear', align_corners=False).squeeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)
            else:
                mask2 = mask2.unsqueeze(0)
            mask1 = mask2 if mask1 is None else torch.cat((mask1, mask2), dim=0)

        return (image1, mask1, "", "", base_width, base_height, len(images), image_path_list)

    def load_images_from_movie(self, movie_path, image_load_cap=0):
        images_from_movie = self.extract_images_from_movie(movie_path, image_load_cap)
        if not images_from_movie:
            raise ValueError(f"No images extracted from movie file: {movie_path}")

        images, masks, image_path_list = [], [], []
        for img in images_from_movie:
            width, height = img.width, img.height
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append((image, width, height))
            masks.append(mask)
            image_path_list.append(movie_path)

        if len(images) == 1:
            image, width, height = images[0]
            return (image, masks[0], "", "", width, height, 1, image_path_list[0])

        base_width, base_height = images[0][1], images[0][2]
        image1 = torch.cat([img[0] for img in images], dim=0)
        mask1 = torch.cat(masks, dim=0)

        return (image1, mask1, "", "", base_width, base_height, len(images), image_path_list)

    def extract_images_from_movie(self, movie_path, image_load_cap=0):
        cap = cv2.VideoCapture(movie_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {movie_path}")

        images = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or (image_load_cap > 0 and frame_count >= image_load_cap):
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            images.append(img)
            frame_count += 1

        cap.release()
        return images

    def resize_and_pad(self, image, target_width, target_height):
        img = image.squeeze().permute(1, 2, 0).numpy() * 255.0
        img = Image.fromarray(img.astype(np.uint8))
        img = ImageOps.fit(img, (target_width, target_height), Image.ANTIALIAS)
        new_img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(new_img).permute(2, 0, 1).unsqueeze(0)

    @classmethod
    def IS_CHANGED(s, path):
        path = os.path.normpath(str(path).replace('"', "")).replace("\\", "/")
        m = hashlib.sha256()
        if not path.startswith("http"):
            with open(path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        else:
            m.update(path.encode("utf-8"))
            return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, path):
        path = os.path.normpath(str(path).replace('"', "")).replace("\\", "/")
        if path.startswith("http"):
            return True
        if not os.path.isfile(path) and not os.path.isdir(path):
            return f"No file or directory found: {path}"
        return True