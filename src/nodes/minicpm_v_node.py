import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    AutoProcessor,
    AutoModel
)
import folder_paths
from huggingface_hub import HfFolder, login, snapshot_download
from PIL import Image
import numpy as np
try:
    from moviepy import VideoFileClip  # MoviePy 2.x
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip  # MoviePy 1.x

import comfy.model_management as mm

class DownloadAndLoadMiniCPMV:
    def __init__(self):
        self.device = mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()
        self.model = None
        self.tokenizer = None
        self.processor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": ([
                    "MiniCPM-V (Full)",
                    "MiniCPM-V-2_6-int4 (7GB VRAM)"
                ], {"default": "MiniCPM-V (Full)"}),
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "For int4 model, this only affects intermediate computations"
                }),
                "attention": (["sdpa", "eager"], {
                    "default": "sdpa",
                    "tooltip": "SDPA: Faster & memory efficient (modern GPUs), Eager: More compatible but slower"
                }),
            }
        }

    RETURN_TYPES = ("MINICPMV_MODEL",)
    FUNCTION = "download_and_load"
    CATEGORY = "MiniCPM-V"

    def download_and_load(self, model_version, precision, attention):
        if self.model is not None:
            print("Model already loaded, reusing...")
            return ({"model": self.model, "tokenizer": self.tokenizer, "processor": self.processor},)

        # Map friendly names to actual repo IDs
        model_paths = {
            "MiniCPM-V (Full)": "openbmb/MiniCPM-V",
            "MiniCPM-V-2_6-int4 (7GB VRAM)": "openbmb/MiniCPM-V-2_6-int4"
        }
        model_path = model_paths[model_version]
        model_name = model_path.split('/')[-1]
        cache_dir = os.path.join(folder_paths.models_dir, "LLM", model_name)
        
        print(f"Selected model version: {model_version}")
        print(f"Model path: {model_path}")
        print(f"Cache directory: {cache_dir}")
        
        # For int4, we default to bf16 for compute dtype regardless of precision setting
        is_int4 = "int4" in model_version
        compute_dtype = torch.bfloat16 if is_int4 else {
            "bf16": torch.bfloat16, 
            "fp16": torch.float16, 
            "fp32": torch.float32
        }[precision]
        
        print(f"Is int4: {is_int4}")
        print(f"Compute dtype: {compute_dtype}")

        if not os.path.exists(cache_dir):
            print(f"Downloading {model_version} to: {cache_dir}")
            token = os.getenv('HF_TOKEN') or HfFolder.get_token()
            if not token:
                raise ValueError(
                    "This model requires authentication. Please:\n"
                    f"1. Accept the license at https://huggingface.co/{model_path}\n"
                    "2. Get your token from https://huggingface.co/settings/tokens\n"
                    "3. Set HF_TOKEN environment variable or run 'huggingface-cli login'"
                )

            login(token=token)
            snapshot_download(
                repo_id=model_path,
                local_dir=cache_dir,
                token=token,
                local_dir_use_symlinks=False
            )
            print("Model downloaded successfully!")

        print(f"Loading model from {cache_dir}")
        
        try:
            model_class = AutoModel if is_int4 else AutoModelForCausalLM
            print(f"Using model class: {model_class.__name__}")
            
            # Prepare model loading kwargs based on version
            model_kwargs = {
                "trust_remote_code": True,
                "attn_implementation": attention,
                "torch_dtype": compute_dtype,
                "device_map": self.device  # Use ComfyUI's device management
            }
            
            # Add int4-specific parameters only for int4 version
            if is_int4:
                model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": compute_dtype
                })

            self.model = model_class.from_pretrained(
                cache_dir,
                **model_kwargs
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                cache_dir,
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                cache_dir,
                trust_remote_code=True
            )
            
            result = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "processor": self.processor,
                "dtype": compute_dtype
            }
            
            print("Components loaded:", list(result.keys()))
            return (result,)
            
        except Exception as e:
            print(f"Error loading model components: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise e

class MiniCPMVNode:
    def __init__(self):
        self.device = mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()
        self.model = None
        self.tokenizer = None
        self.processor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "minicpmv_model": ("MINICPMV_MODEL",),
                "question": ("STRING", {
                    "default": "Describe this scene in great detail in English", 
                    "multiline": True,
                    "tooltip": "Question or prompt for the model to analyze the image(s)"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Input image or batch of frames"
                }),
                "path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to: video file, image file, or folder of images"
                }),
                "n_frames": ("INT", {
                    "default": 4, 
                    "min": 0, 
                    "max": 64,
                    "tooltip": "Frames to sample. 0: all frames (may OOM), 1: middle, 2: first/last, 3: first/mid/last, 4+: evenly spaced"
                }),
                "max_tokens": ("INT", {
                    "default": 512, 
                    "min": 1, 
                    "max": 2048,
                    "tooltip": "Maximum length of generated response"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for random generation (0 for random)"
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Higher values make output more random, lower values more deterministic"
                }),
                "force_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the model will be offloaded to save memory after generation"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "MiniCPM-V"

    def process(self, minicpmv_model, question, image=None, path="", n_frames=4, max_tokens=512, seed=0, temperature=1.0, force_offload=True):
        if self.model is None:
            print("Loading model components...")
            self.model = minicpmv_model["model"].to(self.device)
            self.tokenizer = minicpmv_model["tokenizer"]
            self.processor = minicpmv_model["processor"]

        # Debug output and error checking
        print("Model type:", type(self.model))
        if self.model is None:
            print("Error: Model is None")
            print("minicpmv_model contents:", minicpmv_model)
            raise ValueError("Model is None. Please check that the model was loaded correctly.")

        frames = []
        if path:
            frames = self.load_path_frames(path, n_frames)
        elif image is not None:
            frames = self.tensor_to_pil(image)
            if len(frames) > 1 and n_frames > 0:
                frames = self.sample_frames(frames, n_frames)

        if not frames:
            raise ValueError("No input provided. Please provide either an image/batch or a path to media")

        # Ensure all frames are RGB
        frames = [frame.convert('RGB') for frame in frames]
        
        # Enforce frame count limits
        MAX_FRAMES = 32  # Reduced from 64 to be safer
        if len(frames) > MAX_FRAMES:
            print(f"Warning: Too many frames ({len(frames)}), sampling down to {MAX_FRAMES}")
            indices = np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) > 1:
            # For multiple frames, ensure we have a power of 2 number of frames
            target_frames = 2 ** int(np.log2(len(frames)))
            if target_frames != len(frames):
                print(f"Adjusting frame count from {len(frames)} to {target_frames} for compatibility")
                indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
                frames = [frames[i] for i in indices]

        # Print frame info for debugging
        print(f"Number of frames: {len(frames)}")
        print(f"Frame sizes: {[f.size for f in frames]}")

        # Format messages exactly like their example
        msgs = [{'role': 'user', 'content': frames + [question]}]
        print(f"Message format: {[type(item) for item in msgs[0]['content']]}")

        # Set random seed if provided
        if seed > 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # Set decode params for video
        params = {
            "use_image_id": False,
            "max_slice_nums": 1 if any(f.size[0] > 448 or f.size[1] > 448 for f in frames) else 2,
            "temperature": temperature,
            "do_sample": True,
            "top_p": 0.9,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0
        }

        print(f"Using params: {params}")

        try:
            # Generate response using their chat method
            generated_text = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                max_new_tokens=max_tokens,
                **params
            )

            # If force_offload is enabled, move model to offload device and clear memory
            if force_offload:
                self.model = self.model.to(self.offload_device)
                self.model = None
                mm.soft_empty_cache()

            return (generated_text,)
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def sample_frames(self, frames, n_frames):
        """Sample n_frames evenly from the input frames"""
        if len(frames) <= n_frames:
            return frames
        
        total_frames = len(frames)
        
        if n_frames == 1:
            # Middle frame for single sample
            return [frames[total_frames // 2]]
        elif n_frames == 2:
            # First and last frames
            return [frames[0], frames[-1]]
        elif n_frames == 3:
            # First, middle, and last frames
            return [frames[0], 
                    frames[total_frames // 2],
                    frames[-1]]
        else:
            # For 4+ frames, sample evenly across the sequence
            indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
            return [frames[i] for i in indices]

    def load_path_frames(self, path, n_frames):
        """Load frames from path (video/image/folder)"""
        if not os.path.exists(path):
            raise ValueError(f"Path not found: {path}")
        
        frames = []
        
        # Handle directory of images
        if os.path.isdir(path):
            image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
            files = [f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in image_extensions]
            files.sort()  # Sort for consistent ordering
            
            for file in files:
                img_path = os.path.join(path, file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    frames.append(img)
                except Exception as e:
                    print(f"Warning: Could not load {file}: {str(e)}")
        
        # Handle video file
        elif path.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
            clip = VideoFileClip(path)
            total_frames = int(clip.fps * clip.duration)
            
            if n_frames == 0:
                # Extract frames at a reduced rate to avoid OOM
                target_fps = min(clip.fps, 30)  # Cap at 30fps
                frame_interval = 1.0 / target_fps
                
                # Ensure we get a number of frames divisible by the model's requirements
                num_frames = int(clip.duration * target_fps)
                # Round down to nearest multiple of 8 (common requirement for transformers)
                num_frames = (num_frames // 8) * 8
                
                print(f"Extracting {num_frames} frames at {target_fps} fps")
                frame_times = np.linspace(0, clip.duration, num_frames)
                
                for time in frame_times:
                    frame = clip.get_frame(time)
                    frames.append(Image.fromarray(frame))
                
                print(f"Extracted {len(frames)} frames")
            else:
                # Sample n_frames
                frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
                for idx in frame_indices:
                    time = idx / clip.fps
                    frame = clip.get_frame(time)
                    frames.append(Image.fromarray(frame))
            
            clip.close()
        
        # Handle single image file
        elif path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            frames = [Image.open(path).convert('RGB')]
        
        else:
            raise ValueError(f"Unsupported file type: {path}")
        
        # Apply frame sampling if needed and not a video
        if n_frames > 0 and len(frames) > n_frames and not path.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
            frames = self.sample_frames(frames, n_frames)
        
        # Ensure we don't exceed model's max frame limit and maintain valid frame count
        MAX_FRAMES = 64
        if len(frames) > MAX_FRAMES:
            print(f"Warning: Too many frames ({len(frames)}), sampling down to {MAX_FRAMES}")
            # Round down to nearest multiple of 8
            target_frames = (MAX_FRAMES // 8) * 8
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) > 8:  # Only adjust if we have more than 8 frames
            # Round to nearest multiple of 8
            target_frames = ((len(frames) + 4) // 8) * 8
            if target_frames != len(frames):
                print(f"Adjusting frame count from {len(frames)} to {target_frames} for compatibility")
                indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
                frames = [frames[i] for i in indices]
        
        return frames

    def tensor_to_pil(self, image_tensor):
        """Convert tensor image to PIL Image"""
        if len(image_tensor.shape) == 4:
            # If batch of images, convert each one
            return [Image.fromarray((img.cpu().numpy() * 255).astype('uint8')) 
                   for img in image_tensor]
        else:
            # Single image
            return [Image.fromarray((image_tensor.cpu().numpy() * 255).astype('uint8'))] 