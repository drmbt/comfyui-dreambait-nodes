import torch
import os
import hashlib
import folder_paths
import torchaudio
import torchaudio.backend.sox_io_backend
import torchaudio.backend.soundfile_backend
try:
    from moviepy import VideoFileClip  # MoviePy 2.x
except ImportError:
    from moviepy.editor import VideoFileClip  # MoviePy 1.x

import numpy as np
import random
import io
import json
from comfy.cli_args import args

class SaveAudioPlus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def save_audio(self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.flac"

            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")

            with open(os.path.join(full_output_folder, file), 'wb') as f:
                f.write(buff.getbuffer())

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "audio": results } }

class PreviewAudioPlus(SaveAudioPlus):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

class LoadAudioPlus:
    """
    Enhanced version of ComfyUI's LoadAudio node with better video support
    and more detailed error handling. Uses moviepy 2.0+ syntax.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return {"required": 
                {"audio": (sorted(files), {"widget": "audio"}),
                 "duration_cap": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1})},
                "hidden": {"audio_upload": "AUDIO_UPLOAD"}}

    CATEGORY = "audio"
    RETURN_TYPES = ("AUDIO", "AUDIOINFO")
    RETURN_NAMES = ("audio", "audio_info")
    FUNCTION = "load"
    OUTPUT_NODE = True

    def load(self, audio, duration_cap=0.0, audio_upload=None):
        # Add output_info parameter to track if audio_info is needed
        output_info = not (hasattr(self, 'return_names_to_compute') and 
                         self.return_names_to_compute is not None and 
                         "audio_info" not in self.return_names_to_compute)
        
        if audio_upload is not None:
            audio = audio_upload
            
        audio_path = folder_paths.get_annotated_filepath(audio)
        
        # Check if it's a video file
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        if os.path.splitext(audio_path)[1].lower() in video_extensions:
            try:
                with VideoFileClip(audio_path) as video:
                    if not video.audio:
                        raise ValueError(f"No audio track found in video file: {audio_path}")
                    
                    audio_frames = video.audio.iter_frames()
                    audio_array = np.array(list(audio_frames))
                    sample_rate = int(video.audio.fps)
                    
                    if len(audio_array.shape) == 1:
                        waveform = torch.from_numpy(audio_array[None, :])
                        num_channels = 1 if output_info else None
                    else:
                        waveform = torch.from_numpy(audio_array.T)
                        num_channels = waveform.shape[0] if output_info else None
                    
                    waveform = waveform.float()
                    
                    if waveform.abs().max() > 1.0:
                        waveform = waveform / waveform.abs().max()

                    duration = len(audio_array) / sample_rate if output_info else None
                
            except Exception as e:
                raise RuntimeError(f"Error loading video file {audio_path}: {str(e)}")
        else:
            backends = [
                (torchaudio.backend.soundfile_backend, "soundfile"),
                (torchaudio.backend.sox_io_backend, "sox_io")
            ]
            
            last_error = None
            for backend, name in backends:
                try:
                    torchaudio.set_audio_backend(name)
                    waveform, sample_rate = torchaudio.load(audio_path)
                    if output_info:
                        num_channels = waveform.shape[0]
                        duration = waveform.shape[1] / sample_rate
                    else:
                        num_channels = None
                        duration = None
                    break
                except Exception as e:
                    last_error = e
            else:
                raise RuntimeError(f"Failed to load audio file {audio_path} with any backend. Last error: {str(last_error)}")

        # Apply duration cap if specified
        if duration_cap > 0:
            max_samples = int(duration_cap * sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
                if output_info:
                    duration = duration_cap

        # Create audio data dictionary
        audio_data = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}

        # Always create audio_info since AudioInfoPlus needs it
        audio_info = {
            "sample_rate": sample_rate,
            "num_channels": num_channels if output_info else waveform.shape[0],
            "duration": duration if output_info else waveform.shape[1] / sample_rate,
            "num_samples": waveform.shape[1],
            "file_path": audio_path,
            "file_name": os.path.basename(audio_path),
            "file_extension": os.path.splitext(audio_path)[1].lower(),
            "waveform": waveform,
        }

        return (audio_data, audio_info)

    @classmethod
    def IS_CHANGED(s, audio, duration_cap=0.0):
        image_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio, duration_cap=0.0):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True 

class AudioInfoPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "audio_info": ("AUDIOINFO",),
                    }
                }

    CATEGORY = "audio"

    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT", "STRING", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "sample_rate",
        "num_channels", 
        "duration",
        "num_samples",
        "file_path",
        "file_name",
        "file_extension",
        "max_amplitude",
        "mean_amplitude",
        "rms_amplitude"
    )
    FUNCTION = "get_audio_info"

    def get_audio_info(self, audio_info):
        if audio_info is None:
            raise ValueError("Audio info is None. Make sure the audio file was loaded correctly.")

        # Calculate amplitude statistics from the waveform
        waveform = audio_info["waveform"]
        max_amplitude = float(torch.max(torch.abs(waveform)))
        mean_amplitude = float(torch.mean(torch.abs(waveform)))
        rms_amplitude = float(torch.sqrt(torch.mean(waveform ** 2)))

        return (
            audio_info["sample_rate"],
            audio_info["num_channels"],
            audio_info["duration"],
            audio_info["num_samples"],
            audio_info["file_path"],
            audio_info["file_name"],
            audio_info["file_extension"],
            max_amplitude,
            mean_amplitude,
            rms_amplitude
        )

WEB_DIRECTORY = "js" 

