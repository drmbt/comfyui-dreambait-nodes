from PIL import Image, ImageOps, ImageSequence
import os
import subprocess
from io import BytesIO
import sys
import json
import piexif
import hashlib
from datetime import datetime
import torch
import re
import torch.nn.functional as F
import numpy as np
import requests
from io import BytesIO
import tempfile
import shutil 
import cv2
from pathlib import Path
from PIL.ExifTags import TAGS, GPSTAGS, IFD
from PIL.PngImagePlugin import PngImageFile
from PIL.JpegImagePlugin import JpegImageFile
import zipfile
import random
import logging
import tarfile
import py7zr
from fractions import Fraction
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import itertools
import time
import argparse
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from tqdm import tqdm
import subprocess
from .sort_visual_path import main as sort_visual_path_main
import glob
from typing import Union, List
import folder_paths
from .utils import floatOrInt

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# This script steals from all the major image loader nodes, and incorporates @eden_comfy_nodes @aixander's 
# sort_visual_path travelling salesman script for visually sorting an image sequence


# ffmpeg_path setup
def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True, capture_output=True).stdout.decode("utf-8")
    except:
        return 0
    score = 0
    simple_criterion = [("libvpx", 20), ("264", 10), ("265", 3), ("svtav1", 5), ("libopus", 1)]
    for criterion in simple_criterion:
        if version.find(criterion[0]) >= 0:
            score += criterion[1]
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index + 6:copyright_index + 9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score

if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warn("Failed to import imageio_ffmpeg")
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        if os.path.isfile("ffmpeg"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg"))
        if os.path.isfile("ffmpeg.exe"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
        if len(ffmpeg_paths) == 0:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        elif len(ffmpeg_paths) == 1:
            ffmpeg_path = ffmpeg_paths[0]
        else:
            ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)

def resize_images_to_common_size(images, target_size):
    resized_images = []
    for image in images:
        resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
        resized_images.append(resized_image)
    return resized_images

def calculate_image_distances(images):
    # Determine a common size (e.g., the size of the first image)
    target_size = images[0].size
    # Resize all images to the common size
    resized_images = resize_images_to_common_size(images, target_size)
    # Convert each image to a consistent format and flatten it
    features = [np.array(image.convert("RGB")).flatten() for image in resized_images]
    # Ensure the features are in a 2D array format
    features_array = np.array(features)
    # Calculate the distance matrix
    distance_matrix = squareform(pdist(features_array, 'euclidean'))
    return distance_matrix

def sort_visual_path(images, filenames):
    # Calculate the distance matrix
    distance_matrix = calculate_image_distances(images)
    # Solve the TSP problem using linear sum assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    # Return the sorted filenames based on the sorted indices
    sorted_filenames = [filenames[i] for i in col_ind]
    return sorted_filenames

def load_path(path: str) -> Union[str, List[str]]:
    """
    Resolves a path that can be either:
    - An absolute path to a file or directory
    - A relative path from input directory to a file or directory
    - A path with [input]/[output]/[temp] annotations
    - A wildcard pattern for image files
    
    Returns either a single path string or list of paths if wildcards matched multiple images
    """
    path = path.strip('"').strip("'").replace("\\", "/")
    
    # Handle annotated paths
    if "[" in path:
        name, base_dir = folder_paths.annotated_filepath(path)
        if base_dir is not None:
            full_path = os.path.join(base_dir, name)
            # Check for wildcards in image patterns
            if ('*' in name or '?' in name) and any(name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
                matches = glob.glob(full_path, recursive=True)
                if matches:
                    return [os.path.abspath(p) for p in matches]
            elif os.path.exists(full_path):
                return os.path.abspath(full_path)
    
    # Try as absolute path with wildcards for images
    if ('*' in path or '?' in path) and any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
        matches = glob.glob(path, recursive=True)
        if matches:
            return [os.path.abspath(p) for p in matches]
    elif os.path.exists(path):
        return os.path.abspath(path)
        
    # Try in input directory
    input_path = os.path.join(folder_paths.get_input_directory(), path)
    if ('*' in path or '?' in path) and any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
        matches = glob.glob(input_path, recursive=True)
        if matches:
            return [os.path.abspath(p) for p in matches]
    elif os.path.exists(input_path):
        return os.path.abspath(input_path)
            
    raise FileNotFoundError(f"Could not find file or directory at {path} or {input_path}")

class LoadMedia:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {
                    "image_upload": True,
                    "tooltip": "Path to media file(s). Can be an image, directory of images, video file, zip/tar archive, URL, or wildcard pattern for images (e.g. *.png)."
                }),
                "resize_images_to_first": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Resize all images to match the dimensions of the first image in the sequence."
                }),
                "image_load_cap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "Maximum number of images to load. 0 means load all images."
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "Index of the first image to load from the sequence."
                }),
                "start_index_use_seed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use the seed value as the start_index for random or sequential operations."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "tooltip": "Seed value for random operations and optionally start_index if start_index_use_seed is enabled."
                }),
                "error_after_last_frame": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, raise an error when start_index exceeds frame count. If False, wrap around using modulo."
                }),
                "skip_n": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "Skip N frames between each loaded frame. 0 means load consecutive frames."
                }),
                "sort": (["None", "alphabetical", "date_created", "date_modified", "visual_path", "random"], {
                    "default": "None",
                    "tooltip": "Method to sort multiple images: None, alphabetical, by date, visual similarity, or random."
                }),
                "reverse_order": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reverse the order of loaded images after sorting."
                }),
                "loop_first_frame": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Append the first frame to the end of the sequence to create seamless loops."
                }),
                "force_rate": (floatOrInt, {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0})
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "STRING", "STRING", "STRING", "FLOAT", "AUDIO", "STRING", "METADATA_RAW")
    RETURN_NAMES = ("image", "mask", "WIDTH", "HEIGHT", "COUNT", "FILE_NAME", "FILE_PATH", "PARENT_DIRECTORY", "FPS", "AUDIO", "PROMPT", "METADATA_RAW")
    FUNCTION = "load_media"
    DESCRIPTION = """Loads media from various sources with advanced options for sequence handling.

Key Features:
- Supports images, directories, videos, archives (zip/tar/7z), and URLs
- Smart sorting options including visual similarity-based ordering
- Seamless loop creation with loop_first_frame
"""
        
    def load_media(self, path, resize_images_to_first, image_load_cap, start_index, start_index_use_seed, seed, error_after_last_frame, skip_n, sort, reverse_order, loop_first_frame, force_rate):
        if start_index_use_seed:
            start_index = seed

        try:
            resolved_path = load_path(path)
            
            # If we got a list of paths from wildcard matching
            if isinstance(resolved_path, list):
                # Sort the paths if needed
                if sort == "alphabetical":
                    resolved_path.sort(key=lambda x: x.lower())
                elif sort == "date_created":
                    resolved_path.sort(key=lambda x: os.path.getctime(x))
                elif sort == "date_modified":
                    resolved_path.sort(key=lambda x: os.path.getmtime(x))
                elif sort == "random":
                    random.seed(seed)
                    random.shuffle(resolved_path)
                elif sort == "visual_path":
                    # Load images for visual sorting
                    images = []
                    valid_paths = []
                    first_size = None
                    for img_path in resolved_path:
                        try:
                            img = Image.open(img_path)
                            img = ImageOps.exif_transpose(img)
                            if first_size is None:
                                first_size = img.size
                            if resize_images_to_first:
                                if img.size != first_size:
                                    img = self.resize_right(img, first_size)
                            elif img.size != first_size:
                                continue
                            images.append(img)
                            valid_paths.append(img_path)
                        except Exception as e:
                            print(f"Warning: Could not load image {img_path}: {e}")
                            continue
                    
                    if not images:
                        raise ValueError("No valid images found for visual path sorting")
                    
                    # Sort using visual path
                    sorted_indices = sort_visual_path(images, list(range(len(images))))
                    resolved_path = [valid_paths[i] for i in sorted_indices]
                
                if reverse_order:
                    resolved_path.reverse()

                # Apply start_index and skip_n
                if skip_n > 0:
                    resolved_path = resolved_path[start_index::skip_n + 1]
                else:
                    resolved_path = resolved_path[start_index:]

                # Apply image_load_cap
                if image_load_cap > 0:
                    resolved_path = resolved_path[:image_load_cap]

                # Handle loop_first_frame
                if loop_first_frame and resolved_path:
                    resolved_path.append(resolved_path[0])
                    
                # Create a temporary directory to handle the matched files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Load and process all images first to ensure consistent sizing
                    images = []
                    first_size = None
                    for src_path in resolved_path:
                        try:
                            img = Image.open(src_path)
                            img = ImageOps.exif_transpose(img)
                            if first_size is None:
                                first_size = img.size
                            if resize_images_to_first and img.size != first_size:
                                img = self.resize_right(img, first_size)
                            elif not resize_images_to_first and img.size != first_size:
                                raise ValueError(f"Image size mismatch: {src_path} is {img.size}, expected {first_size}. Enable resize_images_to_first to automatically resize.")
                            images.append(img)
                        except Exception as e:
                            print(f"Warning: Could not load image {src_path}: {e}")
                            continue

                    if not images:
                        raise ValueError("No valid images found in wildcard match")

                    # Save processed images to temp directory
                    for i, img in enumerate(images):
                        dst_path = os.path.join(temp_dir, f"{i:08d}.png")
                        img.save(dst_path)
                    
                    # Process as directory using existing function
                    return self.load_images_from_folder(
                        path=temp_dir,
                        image_load_cap=0,  # We've already applied the cap
                        start_index=0,     # We've already applied the start index
                        skip_n=0,          # We've already applied skip_n
                        resize_images_to_first=False,  # Images are already resized
                        seed=seed,
                        sort="None",       # Already sorted
                        reverse_order=False,  # Already reversed if needed
                        parent_directory=os.path.dirname(resolved_path[0]),
                        loop_first_frame=False  # Already handled loop frame
                    )
            
            # Handle single path (file, directory, or archive)
            path = resolved_path
            
            # Rest of your existing code for handling the path...
            if path.startswith(("http://", "https://")):
                local_filename = self.get_local_file_path(path)
                if not os.path.exists(local_filename):
                    self.download_file(path, local_filename)
                path = local_filename
            else:
                path = os.path.normpath(path)

            # Update parent_directory logic
            if os.path.isdir(path):
                parent_directory = path
            else:
                parent_directory = os.path.dirname(path)

            # Define legal extensions
            legal_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.mp4', '.mov', '.zip', '.tar', '.tar.gz', '.tar.bz2', '.7z']

            # Handle directories
            if os.path.isdir(path):
                dir_files = os.listdir(path)
                # Filter files to only include those with legal extensions
                dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in legal_extensions)]
                frame_count = len([f for f in dir_files if os.path.isfile(os.path.join(path, f))])
            # Handle video files
            elif path.lower().endswith(('.mp4', '.mov')):
                return self.load_images_from_movie(path=path, image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n, reverse_order=reverse_order, resize_images_to_first=resize_images_to_first, parent_directory=parent_directory, sort=sort, force_rate=force_rate)
            # Handle archive files
            elif path.lower().endswith(('.zip', '.tar', '.tar.gz', '.tar.bz2', '.7z')):
                tmpdirname = self.create_persistent_temp_dir(os.path.splitext(os.path.basename(path))[0])
                logger.debug(f"Extracting archive to temporary directory: {tmpdirname}")
                if path.lower().endswith('.zip'):
                    with zipfile.ZipFile(path, 'r') as z:
                        z.extractall(tmpdirname)
                elif path.lower().endswith('.7z'):
                    with py7zr.SevenZipFile(path, mode='r') as z:
                        z.extractall(path=tmpdirname)
                else:
                    with tarfile.open(path, 'r') as t:
                        t.extractall(tmpdirname)

                # Verify extraction
                logger.debug(f"Contents of {tmpdirname}: {os.listdir(tmpdirname)}")
                extracted_items = os.listdir(tmpdirname)
                if len(extracted_items) == 1 and os.path.isdir(os.path.join(tmpdirname, extracted_items[0])):
                    # If there's only one directory, update the path to that directory
                    tmpdirname = os.path.join(tmpdirname, extracted_items[0])
                    logger.debug(f"Single directory found, updating path to: {tmpdirname}")
                path = tmpdirname
                parent_directory = path  # Update parent_directory to the extracted directory
                frame_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            else:
                frame_count = 1

            # Check for subfolder if no frames found
            if frame_count == 0:
                logger.debug(f"No frames found in {path}. Checking for subfolders.")
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if len(subdirs) == 1:
                    path = os.path.join(path, subdirs[0])
                    logger.debug(f"Subfolder found, updating path to: {path}")
                    frame_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

            if frame_count == 0:
                raise ValueError(f"No valid frames found in the provided path: {path}")

            if start_index >= frame_count:
                if error_after_last_frame:
                    raise ValueError(f"start_index {start_index} is greater than the number of frames {frame_count}.")
                else:
                    start_index = start_index % frame_count

            if os.path.isdir(path):
                return self.load_images_from_folder(path=path, image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n, resize_images_to_first=resize_images_to_first, seed=seed, sort=sort, reverse_order=reverse_order, parent_directory=parent_directory, loop_first_frame=loop_first_frame)
            elif path.lower().endswith(('.mp4', '.mov')):
                return self.load_images_from_movie(path=path, image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n, reverse_order=reverse_order, resize_images_to_first=resize_images_to_first, parent_directory=parent_directory, sort=sort, force_rate=force_rate)
            elif path.lower().endswith(('.zip', '.tar', '.tar.gz', '.tar.bz2', '.7z')):
                return self.load_images_from_archive(path=path, image_load_cap=image_load_cap, start_index=start_index, resize_images_to_first=resize_images_to_first, seed=seed, sort=sort, reverse_order=reverse_order, skip_n=skip_n, parent_directory=parent_directory, loop_first_frame=loop_first_frame)
            else:
                return self.load_image(image=path, image_load_cap=image_load_cap, start_index=start_index, resize_images_to_first=resize_images_to_first, parent_directory=parent_directory)

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    def get_local_file_path(self, url):
        # Extract the file extension from the URL
        extension = os.path.splitext(url)[-1]
        # Create a local file path with the extension
        return os.path.join(tempfile.gettempdir(), hashlib.sha256(url.encode()).hexdigest() + extension)

    def download_file(self, url, local_filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    def create_persistent_temp_dir(self, name):
        temp_dir = os.path.join(tempfile.gettempdir(), name)
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def load_image(self, image, image_load_cap, start_index, resize_images_to_first, parent_directory):
        image_path = str(image).replace('"', "")
        if not image_path.startswith(("http://", "https://")):
            image_path = os.path.normpath(image_path).replace("\\", "/")
        i = None
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path)
            i = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            i = Image.open(image_path)

        # Handle GIFs as movies
        if i.format == 'GIF' and getattr(i, "is_animated", True):
            frames = []
            for frame in ImageSequence.Iterator(i):
                frame = frame.convert("RGB")
                frames.append(frame)
            if start_index >= len(frames):
                start_index = start_index % len(frames)
            images = [self.pil2tensor(frame) for frame in frames]
            width, height = frames[0].size
            frame_count = len(images)
            images = torch.cat(images, dim=0)
            file_name = os.path.basename(image_path).rsplit('.', 1)[0]
            return (images, torch.zeros((frame_count, 64, 64), dtype=torch.float32), width, height, frame_count, file_name, image_path, parent_directory, 1.0, None, "", {})                  
        i = ImageOps.exif_transpose(i)
        prompt, negative, width, height = "", "", i.width, i.height
        metadata = self.build_metadata(image_path, i)
        prompt = self.extract_positive_prompt(metadata)

        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        if 'A' in i.getbands():
            mask = 1. - torch.from_numpy(np.array(i.getchannel('A')).astype(np.float32) / 255.0)
        
        file_name = os.path.basename(image_path).rsplit('.', 1)[0]
        return (image, mask, width, height, 1, file_name, image_path, parent_directory, 1.0, None, prompt, metadata)

    def load_images_from_folder(self, path, image_load_cap, start_index, skip_n, resize_images_to_first, seed, sort, reverse_order, parent_directory, loop_first_frame):
        # Strip quotes from the path if present
        path = path.strip('"').strip("'")
        
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Folder '{path}' cannot be found.")
        dir_files = os.listdir(path)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{path}'.")

        # Define legal extensions
        legal_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.mp4', '.mov', '.zip', '.tar', '.tar.gz', '.tar.bz2', '.7z']
        
        # Filter files to only include those with legal extensions
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in legal_extensions)]
        
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No valid files in directory '{path}'.")

        # Ensure the first file is checked after filtering
        first_file = dir_files[0]
        if not any(first_file.lower().endswith(ext) for ext in legal_extensions):
            raise ValueError(f"First file '{first_file}' does not have a legal extension.")
        
        # Check if the first file is a movie
        if dir_files and dir_files[0].lower().endswith(('.mp4', '.mov')):
            images, masks, width, height, frame_count, file_name_list, file_path_list, parent_directory, fps, audio, metadata_list = self.load_images_from_movie(
                path=os.path.join(path, dir_files[0]),
                image_load_cap=image_load_cap,
                start_index=start_index,
                skip_n=skip_n,
                reverse_order=reverse_order,
                resize_images_to_first=resize_images_to_first,
                parent_directory=parent_directory,
                sort=sort
            )
            # Update frame_count to reflect the actual number of frames when skip_n is used
            if skip_n > 0:
                frame_count = len(images)
            return images, masks, width, height, frame_count, file_name_list, file_path_list, parent_directory, fps, audio, metadata_list

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No valid image files in directory '{path}'.")

        # Determine width and height for target_n_pixels
        width, height = 1920, 1200  # Example default values; replace with actual logic to determine these
        target_n_pixels = width * height

        # Sorting logic
        if sort == "alphabetical":
            dir_files.sort(key=lambda x: x.lower())
        elif sort == "date_created":
            dir_files.sort(key=lambda x: os.path.getctime(os.path.join(path, x)))
        elif sort == "date_modified":
            dir_files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
        elif sort == "random":
            random.seed(seed)
            random.shuffle(dir_files)
        elif sort == "visual_path":
            # Call the main function from sort_visual_path.py
            dir_files = sort_visual_path_main(path, target_n_pixels=target_n_pixels, list_only=False)
            #print(f"!!!! 367  {dir_files}")
        
        if reverse_order:
            dir_files.reverse()
        
        # Apply start_index only once
        dir_files = [os.path.normpath(os.path.join(path, x)).replace("\\", "/") for x in dir_files]
        
        if skip_n > 0:
            dir_files = dir_files[start_index::skip_n + 1]
        else:
            dir_files = dir_files[start_index:]
        
        if image_load_cap > 0:
            dir_files = dir_files[:image_load_cap]

        images, masks, file_path_list, file_name_list, metadata_list, prompts = [], [], [], [], [], []
        first_image_size = None
        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            metadata = self.build_metadata(image_path, i)
            if first_image_size is None:
                first_image_size = i.size
            if i.size == first_image_size or resize_images_to_first:
                img = i if i.size == first_image_size else self.resize_right(i, first_image_size)
                images.append(img)
                masks.append(torch.zeros((64, 64), dtype=torch.float32, device="cpu"))
                file_path_list.append(image_path)
                file_name_list.append(os.path.basename(image_path).rsplit('.', 1)[0])
                metadata_list.append(metadata)
                prompts.append(self.extract_positive_prompt(metadata))
           
        #print(f"!!!! 375  file_name_list: {file_name_list}")
        # Append the first frame to the end if loop_first_frame is True
        if loop_first_frame and images:
            images.append(images[0])
            masks.append(masks[0])
            file_path_list.append(file_path_list[0])
            file_name_list.append(file_name_list[0])
            metadata_list.append(metadata_list[0])
            prompts.append(prompts[0])

        if not images:
            errmsg = f"No valid images found in directory '{path}'!"
            logger.error(errmsg)
            raise ValueError(errmsg)

        images = [self.pil2tensor(img) for img in images]
        width, height = first_image_size  # Ensure width and height are assigned
        frame_count = len(images)
        if len(images) == 1:
            image = images[0]
            return (image, masks[0], width, height, frame_count, file_name_list[0], file_path_list[0], parent_directory, 1.0, None, prompts[0], metadata_list[0])
        images = torch.cat(images, dim=0)
        return (images, masks, width, height, frame_count, file_name_list, file_path_list, parent_directory, 1.0, None, prompts, metadata_list)
    def load_images_from_movie(self, path, image_load_cap, start_index, skip_n, reverse_order, resize_images_to_first, parent_directory, sort, force_rate):
        images_from_movie, fps, audio, frame_count = self.extract_images_from_movie(movie_path=path, image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n, force_rate=force_rate)
        if not images_from_movie:
            raise ValueError(f"No images extracted from movie file: {path}")

        # Sort images using visual path if specified
        if sort == "visual_path":
            filenames = [f"frame_{i}" for i in range(len(images_from_movie))]
            sorted_filenames = sort_visual_path(images_from_movie, filenames)
            images_from_movie = [images_from_movie[filenames.index(name)] for name in sorted_filenames]
         # Randomly shuffle images if specified
        elif sort == "random":
            random.shuffle(images_from_movie)
        if reverse_order:
            images_from_movie.reverse()

        images, masks, file_path_list, file_name_list, metadata_list, prompts = [], [], [], [], [], []
        first_image_size = None
        for img in images_from_movie:
            if first_image_size is None:
                first_image_size = img.size
            width, height = first_image_size
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            file_path_list.append(path)
            file_name_list.append(os.path.basename(path).rsplit('.', 1)[0])
            metadata = self.build_metadata(path, img)
            metadata_list.append(metadata)
            prompts.append(self.extract_positive_prompt(metadata))

        if not images:
            errmsg = f"No valid images found in movie file '{path}'!"
            logger.error(errmsg)
            raise ValueError(errmsg)

        images = torch.cat(images, dim=0)
        return (images, masks, width, height, len(images), file_name_list, file_path_list, parent_directory, fps, audio, metadata_list)

    def load_images_from_archive(self, path, image_load_cap, start_index, resize_images_to_first, seed, sort, reverse_order, skip_n, parent_directory, loop_first_frame):
        supported_image_formats = ('.png', '.jpg', '.jpeg', '.gif', '.tga', '.tiff', '.webp')
        supported_video_formats = ('.mp4', '.mov')
        valid_extensions = supported_image_formats + supported_video_formats
        images, masks, file_path_list, file_name_list, metadata_list, prompts = [], [], [], [], [], []
        first_image_size = None

        if path.lower().endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as z:
                file_names = [f for f in z.namelist() if f.lower().endswith(valid_extensions)]
        elif path.lower().endswith('.7z'):
            with py7zr.SevenZipFile(path, mode='r') as z:
                file_names = [f for f in z.getnames() if f.lower().endswith(valid_extensions)]
        else:
            with tarfile.open(path, 'r') as t:
                file_names = [f for f in t.getnames() if f.lower().endswith(valid_extensions)]

        # Determine width and height for target_n_pixels
        width, height = 1920, 1200  # Example default values; replace with actual logic to determine these
        target_n_pixels = width * height

        # Sorting logic
        if sort == "alphabetical":
            file_names.sort(key=lambda x: x.lower())
        elif sort == "date_created":
            file_names.sort(key=lambda x: os.path.getctime(os.path.join(path, x)))
        elif sort == "date_modified":
            file_names.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
        elif sort == "random":
            random.seed(seed)
            random.shuffle(file_names)
        elif sort == "visual_path":
            # Call the main function from sort_visual_path.py
            file_names = sort_visual_path_main(parent_directory, target_n_pixels=target_n_pixels, list_only=False)
        #print(f"!!!! {file_names}")
        if reverse_order:
            file_names.reverse()

        if skip_n > 0:
            file_names = file_names[start_index::skip_n + 1]
        else:
            file_names = file_names[start_index:]

        if image_load_cap > 0:
            file_names = file_names[:image_load_cap]

        # Check if the first file is a movie
        if file_names and file_names[0].lower().endswith(supported_video_formats):
            #print(f"!!! {file_names[0]}")
            return self.load_images_from_movie(path=os.path.join(parent_directory, file_names[0]), image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n, reverse_order=reverse_order, resize_images_to_first=resize_images_to_first, parent_directory=parent_directory, sort=sort)

        for file_name in file_names:
            if file_name.lower().endswith(supported_image_formats):
                if path.lower().endswith('.zip'):
                    with zipfile.ZipFile(path, 'r') as z:
                        with z.open(file_name) as f:
                            i = Image.open(f)
                elif path.lower().endswith('.7z'):
                    with py7zr.SevenZipFile(path, mode='r') as z:
                        with z.open(file_name) as f:
                            i = Image.open(f)
                else:
                    with tarfile.open(path, 'r') as t:
                        with t.extractfile(file_name) as f:
                            i = Image.open(f)

                i = ImageOps.exif_transpose(i)
                metadata = self.build_metadata(file_name, i)
                if first_image_size is None:
                    first_image_size = i.size
                if i.size == first_image_size or resize_images_to_first:
                    img = i if i.size == first_image_size else self.resize_right(i, first_image_size)
                    images.append(img)
                    masks.append(torch.zeros((64, 64), dtype=torch.float32, device="cpu"))
                    file_path_list.append(file_name)
                    file_name_list.append(os.path.basename(file_name).rsplit('.', 1)[0])
                    metadata_list.append(metadata)
                    prompts.append(self.extract_positive_prompt(metadata))

        # Append the first frame to the end if loop_first_frame is True
        if loop_first_frame and images:
            images.append(images[0])
            masks.append(masks[0])
            file_path_list.append(file_path_list[0])
            file_name_list.append(file_name_list[0])
            metadata_list.append(metadata_list[0])
            prompts.append(prompts[0])

        if not images:
            errmsg = f"No valid images found in archive '{path}'!"
            logger.error(errmsg)
            raise ValueError(errmsg)

        images = [self.pil2tensor(img) for img in images]
        width, height = first_image_size
        frame_count = len(images)
        if len(images) == 1:
            image = images[0]
            return (image, masks[0], width, height, frame_count, file_name_list[0], file_path_list[0], parent_directory, 1.0, None, metadata_list[0])
        images = torch.cat(images, dim=0)
        return (images, masks, width, height, frame_count, file_name_list, file_path_list, parent_directory, 1.0, None, metadata_list)

    def extract_images_from_movie(self, movie_path, image_load_cap, start_index, skip_n, force_rate=0):
        # Check if ffmpeg is available in the system's PATH
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")

        if not ffmpeg_path:
            raise FileNotFoundError("ffmpeg executable not found. Please ensure it is installed and the path is correct.")
        if not ffprobe_path:
            raise FileNotFoundError("ffprobe executable not found. Please ensure it is installed and the path is correct.")

        try:
            # Get frame count and FPS using ffprobe
            probe_args = [
                ffprobe_path, 
                "-v", "error", 
                "-select_streams", "v:0", 
                "-count_frames", 
                "-show_entries", "stream=nb_read_frames,r_frame_rate", 
                "-of", "default=nokey=1:noprint_wrappers=1", 
                movie_path
            ]
            probe_result = subprocess.run(probe_args, capture_output=True, text=True, check=True)
            output = probe_result.stdout.strip().split('\n')
            frame_count = int(output[1])  # Convert frame count to int
            fps = float(Fraction(output[0]))  # Convert fps string to float
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to probe video file: {e.stderr}")
            # Fallback to old method
            frame_count = 0
            fps = 30.0  # Default FPS

        cap = cv2.VideoCapture(movie_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {movie_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        audio = None

        if start_index >= total_frames:
            start_index = start_index % total_frames

        images = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        
        # Calculate target frame time based on force_rate
        if force_rate == 0:
            target_frame_time = 1 / fps
        else:
            target_frame_time = 1 / force_rate
            
        base_frame_time = 1 / fps
        time_offset = target_frame_time
        
        # Calculate total frames to process
        if total_frames > 0:
            if force_rate != 0:
                yieldable_frames = int(total_frames / fps * force_rate)
            else:
                yieldable_frames = total_frames
            if skip_n > 0:
                yieldable_frames //= (skip_n + 1)
            if image_load_cap > 0:
                yieldable_frames = min(image_load_cap, yieldable_frames)
        else:
            yieldable_frames = 0

        frames_added = 0
        total_frame_count = 0
        total_frames_evaluated = -1
        
        while cap.isOpened() and frames_added < yieldable_frames:
            if time_offset < target_frame_time:
                is_returned = cap.grab()
                if not is_returned:
                    break
                time_offset += base_frame_time
                continue
                
            time_offset -= target_frame_time
            total_frame_count += 1
            
            # Skip frames before start_index
            if total_frame_count <= start_index:
                continue
            else:
                total_frames_evaluated += 1
                
            # Skip frames based on skip_n
            if skip_n > 0 and total_frames_evaluated % (skip_n + 1) != 0:
                continue
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame))
            frames_added += 1

        cap.release()

        # Extract audio if available
        audio = lazy_get_audio(movie_path, start_time=0, duration=total_frames / fps)
        
        # Update fps if force_rate is set
        if force_rate > 0:
            fps = force_rate
            
        return images, fps, audio, len(images)

    def pil2tensor(self, x):
        return torch.from_numpy(np.array(x).astype(np.float32) / 255.0).unsqueeze(0)

    def resize_right(self, image, target_size):
        img_ratio = image.width / image.height
        target_ratio = target_size[0] / target_size[1]
        resize_width, resize_height = (
            (target_size[0], round(target_size[0] / img_ratio)) if target_ratio > img_ratio else
            (round(target_size[1] * img_ratio), target_size[1])
        )
        image = image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        x_crop, y_crop = (resize_width - target_size[0]) // 2, (resize_height - target_size[1]) // 2
        return image.crop((x_crop, y_crop, x_crop + target_size[0], y_crop + target_size[1]))

    def build_metadata(self, image_path, img):
        metadata = {}
        if isinstance(img, PngImageFile):
            metadata = {k: v for k, v in img.info.items() if k != "exif"}
        elif isinstance(img, JpegImageFile):
            exif_data = img._getexif()
            if exif_data:
                metadata = {TAGS.get(k, k): v for k, v in exif_data.items()}
        metadata["file_path"] = image_path
        return metadata

    def extract_positive_prompt(self, metadata):
        return metadata.get("prompt", "")




class LazyAudioMap:
    def __init__(self, file, start_time, duration):
        self.file = file
        self.start_time = start_time
        self.duration = duration
        self._dict = None

    def __getitem__(self, key):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return self._dict[key]

    def __iter__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return iter(self._dict)

    def __len__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return len(self._dict)

def lazy_get_audio(file, start_time=0, duration=0):
    return LazyAudioMap(file, start_time, duration)

def get_audio(file, start_time=0, duration=0):
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        res = subprocess.run(args + ["-f", "f32le", "-"], capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(', (\\d+) Hz, (\\w+), ', res.stderr.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to extract audio from {file}:\n" + e.stderr.decode("utf-8"))
    if match:
        ar = int(match.group(1))
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1, ac)).transpose(0, 1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}

