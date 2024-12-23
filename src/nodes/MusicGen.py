import os
import torch
import random
import numpy as np
import folder_paths
import warnings
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Suppress specific PyTorch warning about attention masks
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")

# Register our model type with ComfyUI
folder_paths.folder_names_and_paths["musicgen"] = ([os.path.join(folder_paths.models_dir, "musicgen")], folder_paths.supported_pt_extensions)

MUSICGEN_MODELS = {
    # Original models (trained on public datasets)
    "small": "facebook/musicgen-small",        # 300M parameters, most reliable
    "medium": "facebook/musicgen-medium",      # 1.5B parameters, balanced
    "large": "facebook/musicgen-large",        # 3.3B parameters, best quality
    
    # Stereo models (for spatial audio)
    "stereo-small": "facebook/musicgen-stereo-small",  # Basic stereo
    "stereo-medium": "facebook/musicgen-stereo-medium", # Better stereo
    "stereo-large": "facebook/musicgen-stereo-large",   # Best stereo
}

def init_audio_model(model_name):
    checkpoint = MUSICGEN_MODELS[model_name]
    base_path = folder_paths.folder_names_and_paths["musicgen"][0][0]
    model_path = os.path.join(base_path, model_name)
    
    # First try to load from HF cache or local path
    try:
        print(f"Attempting to load MusicGen model {model_name}...")
        audio_processor = MusicgenProcessor.from_pretrained(checkpoint)
        audio_model = MusicgenForConditionalGeneration.from_pretrained(checkpoint)
        
        # If loaded from cache, copy to ComfyUI models directory
        if not os.path.exists(os.path.join(model_path, "config.json")):
            print(f"Model found in cache, copying to ComfyUI models directory: {model_path}")
            os.makedirs(model_path, exist_ok=True)
            audio_processor.save_pretrained(model_path)
            audio_model.save_pretrained(model_path)
    except Exception as e:
        print(f"Could not load from cache: {str(e)}")
        
        # If not in cache, download directly to model_path
        if not os.path.exists(os.path.join(model_path, "config.json")):
            try:
                print(f"Downloading MusicGen model {model_name} from {checkpoint}...")
                os.makedirs(model_path, exist_ok=True)
                
                # Try direct download first
                try:
                    snapshot_download(
                        repo_id=checkpoint,
                        local_dir=model_path,
                        endpoint='https://huggingface.co'
                    )
                except Exception as e:
                    print(f"Main endpoint failed, trying mirror: {str(e)}")
                    snapshot_download(
                        repo_id=checkpoint,
                        local_dir=model_path,
                        endpoint='https://hf-mirror.com'
                    )
                
                if not os.path.exists(os.path.join(model_path, "config.json")):
                    raise Exception("Download appeared to succeed but model files are missing")
                    
                print(f"Download complete. Model saved to {model_path}")
                
            except Exception as e:
                raise Exception(f"""Failed to download MusicGen model {model_name}.
Error: {str(e)}

Possible solutions:
1. Check your internet connection
2. Try a different model (e.g., 'small' instead of '{model_name}')
3. Download the model manually from {checkpoint} and place it in:
   {model_path}
""")

    try:
        print(f"Loading MusicGen model {model_name} from {model_path}...")
        audio_processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        audio_model = MusicgenForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        audio_model = audio_model.to(torch.device('cpu'))
        audio_model.generation_config.guidance_scale = 4.0
        audio_model.generation_config.max_new_tokens = 1500
        audio_model.generation_config.temperature = 1.5
        print(f"Successfully loaded MusicGen model {model_name}")
        return (audio_processor, audio_model)
    except Exception as e:
        raise Exception(f"""Failed to load MusicGen model {model_name}.
Error: {str(e)}

The model files may be corrupted. Try:
1. Deleting the folder: {model_path}
2. Letting the node download the model again
3. Or downloading manually from {checkpoint}""")

class MusicGen:
    def __init__(self):
        self.audio_model = None
        self.current_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (list(MUSICGEN_MODELS.keys()), {
                "default": "small",
                "tooltip": """Available models:
                    Original models (most stable):
                    - small: 300M parameters, fastest, reliable
                    - medium: 1.5B parameters, balanced
                    - large: 3.3B parameters, highest quality
                    
                    Stereo models:
                    - stereo-small/medium/large: For spatial audio generation"""
            }),
            "prompt": ("STRING", {
                "multiline": True, 
                "default": '',
                "dynamicPrompts": True
            }),
            "seconds": ("FLOAT", {
                "default": 5, 
                "min": 1,
                "max": 1000,
                "step": 0.1,
                "display": "number"
            }),
            "guidance_scale": ("FLOAT", {
                "default": 4.0, 
                "min": 0,
                "max": 20,
            }),
            "seed": ("INT", {"default": 0, "min": 0, "max": np.iinfo(np.int32).max}),
            "device": (["auto","cpu"],),
        }}
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "audio"
    DESCRIPTION = """Generate music using Meta's MusicGen model based on text prompts. 
    Forked from comfyui-sound-lab's musicNode and modified to conform to ComfyUI's standard AUDIO format,
    making it compatible with core audio nodes like VAEDecodeAudio, SaveAudio, and PreviewAudio."""
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
  
    def run(self, model, prompt, seconds, guidance_scale, seed, device):
        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
      
        # Reinitialize model if it's not loaded or if a different model is selected
        if self.audio_model is None or self.current_model != model:
            self.audio_processor, self.audio_model = init_audio_model(model)
            self.current_model = model

        inputs = self.audio_processor(
            text=prompt,
            padding=True,
            return_tensors="pt",
        )

        if device == 'auto':
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.audio_model.to(torch.device(device))

        tokens_per_second = 1500 / 30
        max_tokens = int(tokens_per_second * seconds)

        # Generate audio
        sampling_rate = self.audio_model.config.audio_encoder.sampling_rate
        audio_values = self.audio_model.generate(**inputs.to(device), 
                    do_sample=True, 
                    guidance_scale=guidance_scale, 
                    max_new_tokens=max_tokens,
                    )
        
        self.audio_model.to(torch.device('cpu'))

        audio_numpy = audio_values[0, 0].cpu().numpy()
        audio_tensor = torch.from_numpy(audio_numpy).float()
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

        return ({"waveform": audio_tensor, "sample_rate": sampling_rate}, ) 