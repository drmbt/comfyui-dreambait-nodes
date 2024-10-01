from PIL import Image, ImageOps, ImageSequence
import os
import sys
import json
import piexif
import hashlib
from datetime import datetime
import subprocess
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

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadMedia:
    """
    Loads media from a specified path, which can be an image path or directory of images, a video file, zip archive or a URL.
    It supports extracting frames from video files and treating them as image sequences.

    TODO:
    - METADATA and PROMPT extraction aren't working properly for images loaded from a directory or zip archive.
    - get FPS output
    - get
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"image_upload": True}),
                "resize_images_to_first": ("BOOLEAN", {"default": True}),
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index_use_seed": ("BOOLEAN", {"default": False, "tooltip": "Use seed as the start_index value."}),
                "error_after_last_frame": ("BOOLEAN", {"default": False, "tooltip": "Raise an exception if start_index > COUNT, otherwise use start_index % COUNT."}),
                "skip_n": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "Number of images to skip. 0 means no skipping."}),
                "sort": (["None", "alphabetical", "date_created", "date_modified", "random"], {"default": "None", "tooltip": "sort method for multi image inputs"}),
                "reverse_order": ("BOOLEAN", {"default": False}),
                "seed": ("INT",{"default": 0}),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "STRING", "STRING", "STRING", "FLOAT", "AUDIO", "METADATA_RAW")
    RETURN_NAMES = ("image", "mask", "WIDTH", "HEIGHT", "COUNT", "FILE_NAME", "FILE_PATH", "PARENT_DIRECTORY", "FPS", "AUDIO", "METADATA_RAW")
    FUNCTION = "load_media"

    @classmethod
    def IS_CHANGED(cls, path, resize_images_to_first, image_load_cap, start_index, start_index_use_seed, sort, seed, reverse_order):
        path = str(path).replace('"', "")
        if not path.startswith(("http://", "https://")):
            path = os.path.normpath(path).replace("\\", "/")
        m = hashlib.sha256()
        if not path.startswith(("http://", "https://")):
            with open(path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        else:
            m.update(path.encode("utf-8"))
            return m.digest().hex()
        
    def load_media(self, path, seed, image_load_cap, start_index, start_index_use_seed, error_after_last_frame, skip_n, resize_images_to_first, sort, reverse_order):
        if start_index_use_seed:
            start_index = seed

        if path.startswith(("http://", "https://")):
            local_filename = self.get_local_file_path(path)
            print(f"!!!!! URL provided: {path}")
            if not os.path.exists(local_filename):
                self.download_file(path, local_filename)
            print(f"!!!!! Downloaded file: {local_filename}")
            path = local_filename
        else:
            path = os.path.normpath(path)

        # Update parent_directory logic
        if os.path.isdir(path):
            parent_directory = path
        else:
            parent_directory = os.path.dirname(path)
        print(f"!!!!! Path for processing: {path}")

        if os.path.isdir(path):
            dir_files = os.listdir(path)
            frame_count = len([f for f in dir_files if os.path.isfile(os.path.join(path, f))])
        elif path.lower().endswith(('.mp4', '.mov')):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        elif path.lower().endswith(('.zip', '.tar', '.tar.gz', '.tar.bz2', '.7z')):  # Add .7z support
            tmpdirname = self.create_persistent_temp_dir(os.path.splitext(os.path.basename(path))[0])
            if path.lower().endswith('.zip'):
                with zipfile.ZipFile(path, 'r') as z:
                    z.extractall(tmpdirname)
                    frame_count = len([f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.tga', '.tiff', '.webp'))])
            elif path.lower().endswith('.7z'):  # Add .7z extraction logic
                with py7zr.SevenZipFile(path, mode='r') as z:
                    z.extractall(path=tmpdirname)
                    frame_count = len([f for f in z.getnames() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.tga', '.tiff', '.webp', '.mp4', '.mov'))])
            else:
                with tarfile.open(path, 'r') as t:
                    t.extractall(tmpdirname)
                    frame_count = len([f for f in t.getnames() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.tga', '.tiff', '.webp'))])
            print(f"!!!!! Extracted to persistent directory: {tmpdirname}")
            path = tmpdirname  # Update path to the persistent directory

            # Check if the extracted content is a single directory
            extracted_items = os.listdir(tmpdirname)
            print(f"!!!!! Extracted items: {extracted_items}")
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(tmpdirname, extracted_items[0])):
                tmpdirname = os.path.join(tmpdirname, extracted_items[0])
                print(f"!!!!! Updated path to single extracted directory: {tmpdirname}")
            path = tmpdirname
            parent_directory = path  # Update parent_directory to the extracted directory

        else:
            frame_count = 1

        if frame_count == 0:
            raise ValueError(f"No valid frames found in the provided path: {path}")

        if start_index >= frame_count:
            if error_after_last_frame:
                raise ValueError(f"start_index {start_index} is greater than the number of frames {frame_count}.")
            else:
                start_index = start_index % frame_count

        if os.path.isdir(path):
            extracted_items = os.listdir(path)
            if extracted_items:
                first_item = extracted_items[0]
                if first_item.lower().endswith(('.mp4', '.mov')):
                    return self.load_images_from_movie(path=os.path.join(path, first_item), image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n, reverse_order=reverse_order, resize_images_to_first=resize_images_to_first, parent_directory=parent_directory)
            return self.load_images_from_folder(path=path, image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n, resize_images_to_first=resize_images_to_first, seed=seed, sort=sort, reverse_order=reverse_order, parent_directory=parent_directory)
        elif path.lower().endswith(('.mp4', '.mov')):
            return self.load_images_from_movie(path=path, image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n, reverse_order=reverse_order, resize_images_to_first=resize_images_to_first, parent_directory=parent_directory)
        elif path.lower().endswith(('.zip', '.tar', '.tar.gz', '.tar.bz2', '.7z')):  # Add .7z support
            return self.load_images_from_archive(path=path, image_load_cap=image_load_cap, start_index=start_index, resize_images_to_first=resize_images_to_first, seed=seed, sort=sort, reverse_order=reverse_order, skip_n=skip_n, parent_directory=parent_directory)
        else:
            return self.load_image(image=path, image_load_cap=image_load_cap, start_index=start_index, resize_images_to_first=resize_images_to_first, parent_directory=parent_directory)
           
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

        if i.format == 'GIF' and getattr(i, "is_animated", False):
            frames = []
            for frame in ImageSequence.Iterator(i):
                frame = frame.convert("RGB")
                frames.append(frame)
            if start_index >= len(frames):
                start_index = start_index % len(frames)
            i = frames[start_index]
        else:
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
        return (image, mask, width, height, 1, file_name, image_path, parent_directory, 1.0, None, metadata)

    def load_images_from_folder(self, path, image_load_cap, start_index, skip_n, resize_images_to_first, seed, sort, reverse_order, parent_directory):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Folder '{path}' cannot be found.")
        dir_files = os.listdir(path)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{path}'.")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No valid image files in directory '{path}'.")

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

        if reverse_order:
            dir_files.reverse()

        dir_files = [os.path.normpath(os.path.join(path, x)).replace("\\", "/") for x in dir_files][start_index:]

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

        if not images:
            errmsg = f"No valid images found in directory '{path}'!"
            logger.error(errmsg)
            raise ValueError(errmsg)

        images = [self.pil2tensor(img) for img in images]
        width, height = first_image_size  # Ensure width and height are assigned
        if len(images) == 1:
            image = images[0]
            return (image, masks[0], width, height, 1, file_name_list[0], file_path_list[0], parent_directory, 1.0, None, metadata_list[0])
        images = torch.cat(images, dim=0)
        return (images, masks, width, height, len(images), file_name_list, file_path_list, parent_directory, 1.0, None, metadata_list)
    
    def load_images_from_movie(self, path, image_load_cap, start_index, skip_n, reverse_order, resize_images_to_first, parent_directory):
        images_from_movie, fps, audio = self.extract_images_from_movie(movie_path=path, image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n)
        if not images_from_movie:
            raise ValueError(f"No images extracted from movie file: {path}")

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

        if len(images) == 1:
            image = images[0]
            return (image, masks[0], width, height, 1, file_name_list[0], file_path_list[0], parent_directory, fps, audio, metadata_list[0])
        images = torch.cat(images, dim=0)
        return (images, masks, width, height, len(images), file_name_list, file_path_list, parent_directory, fps, audio, metadata_list)

    def load_images_from_archive(self, path, image_load_cap, start_index, resize_images_to_first, seed, sort, reverse_order, skip_n, parent_directory):
        supported_image_formats = ('.png', '.jpg', '.jpeg', '.gif', '.tga', '.tiff', '.webp')
        supported_video_formats = ('.mp4', '.mov')
        valid_extensions = supported_image_formats + supported_video_formats
        images, masks, file_path_list, file_name_list, metadata_list, prompts = [], [], [], [], [], []
        first_image_size = None

        with tempfile.TemporaryDirectory() as tmpdirname:
            if path.lower().endswith('.zip'):
                with zipfile.ZipFile(path, 'r') as archive:
                    archive.extractall(tmpdirname)
            elif path.lower().endswith('.7z'):  # Add .7z extraction logic
                with py7zr.SevenZipFile(path, mode='r') as archive:
                    archive.extractall(path=tmpdirname)
            elif path.lower().endswith(('.tar', '.tar.gz', '.tar.bz2')):
                with tarfile.open(path, 'r') as archive:
                    archive.extractall(tmpdirname)
            else:
                raise ValueError(f"Unsupported archive format: {path}")

            # Check if the extracted content is a single directory
            extracted_items = os.listdir(tmpdirname)
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(tmpdirname, extracted_items[0])):
                tmpdirname = os.path.join(tmpdirname, extracted_items[0])

            # Gather all files with valid extensions in the current directory
            file_names = [os.path.join(tmpdirname, f) for f in os.listdir(tmpdirname) if any(f.lower().endswith(ext) for ext in valid_extensions)]

            if len(file_names) == 0:
                raise FileNotFoundError(f"No valid image or video files in archive '{path}'.")

            # Sorting logic
            if sort == "alphabetical":
                file_names.sort(key=lambda x: x.lower())
            elif sort == "date_created":
                file_names.sort(key=lambda x: os.path.getctime(x))
            elif sort == "date_modified":
                file_names.sort(key=lambda x: os.path.getmtime(x))
            elif sort == "random":
                random.seed(seed)
                random.shuffle(file_names)

            if reverse_order:
                file_names.reverse()

            if skip_n > 0:
                file_names = file_names[start_index::skip_n + 1] if skip_n > 0 else file_names[start_index:]

                if image_load_cap > 0:
                    file_names = file_names[:image_load_cap]
                for file_name in file_names:
                    if file_name.lower().endswith(supported_video_formats):
                        return self.load_images_from_movie(path=file_name, image_load_cap=image_load_cap, start_index=start_index, skip_n=skip_n, reverse_order=reverse_order, resize_images_to_first=resize_images_to_first, parent_directory=parent_directory)
                    elif file_name.lower().endswith(supported_image_formats):
                        i = Image.open(file_name)
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

            if not images:
                errmsg = f"No valid images found in archive '{path}'!"
                logger.error(errmsg)
                raise ValueError(errmsg)

            images = [self.pil2tensor(img) for img in images]
            width, height = first_image_size  # Ensure width and height are assigned
            if len(images) == 1:
                image = images[0]
                return (image, masks[0], width, height, 1, file_name_list[0], file_path_list[0], parent_directory, 1.0, None, metadata_list[0])
            images = torch.cat(images, dim=0)
            return (images, masks, width, height, len(images), file_name_list, file_path_list, parent_directory, 1.0, None, metadata_list)

    def extract_images_from_movie(self, movie_path, image_load_cap, start_index, skip_n):
        cap = cv2.VideoCapture(movie_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {movie_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        audio = None

        if start_index >= total_frames:
            start_index = start_index % total_frames

        images = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        for _ in range(image_load_cap if image_load_cap > 0 else total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if skip_n > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip_n)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame))

        cap.release()

        # Extract audio if available
        audio = lazy_get_audio(movie_path, start_time=0, duration=total_frames / fps)

        return images, fps, audio

    def pil2tensor(self, img):
        img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    def resize_right(self, img, size):
        return img.resize(size, Image.LANCZOS)

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

    def create_persistent_temp_dir(self, name):
        temp_dir = os.path.join(tempfile.gettempdir(), name)
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def get_local_file_path(self, url):
        return os.path.join(tempfile.gettempdir(), hashlib.sha256(url.encode()).hexdigest())

    def download_file(self, url, local_filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


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