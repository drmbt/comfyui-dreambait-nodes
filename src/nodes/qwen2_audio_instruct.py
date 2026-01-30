import torch
import os
import folder_paths
import librosa
from transformers import (
    Qwen2AudioForConditionalGeneration, 
    AutoProcessor,
)
from huggingface_hub import login, get_token
import comfy.model_management as mm

class Qwen2AudioInstruct:
    """
    A node that provides access to Qwen2-Audio-7B-Instruct for audio analysis and chat.
    Designed to work with ComfyUI's native audio handling.
    """
    def __init__(self):
        self.device = mm.get_torch_device()
        self.model = None
        self.processor = None
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (["Qwen2-Audio-7B-Instruct-Int4"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {
                    "default": 256, 
                    "min": 1, 
                    "max": 2048,
                    "step": 1,
                    "tooltip": "Maximum number of tokens to generate"
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
                "force_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If true, the model will be offloaded to save memory after generation"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "audio/text"

    def inference(self, text, model, force_offload, seed=0, max_tokens=256, audio=None):
        if seed > 0:
            torch.manual_seed(seed)
            
        model_id = f"Sergei6000/{model}"
        model_name = os.path.basename(model_id)
        cache_dir = os.path.join(folder_paths.models_dir, "LLM", model_name)
        
        # Download model if needed
        if not os.path.exists(cache_dir):
            print(f"Downloading {model} to: {cache_dir}")
            token = os.getenv('HF_TOKEN') or get_token()
            if not token:
                raise ValueError(
                    "This model requires authentication. Please:\n"
                    f"1. Accept the license at https://huggingface.co/{model_id}\n"
                    "2. Get your token from https://huggingface.co/settings/tokens\n"
                    "3. Set HF_TOKEN environment variable or run 'huggingface-cli login'"
                )
            login(token=token)
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=cache_dir,
                token=token,
                local_dir_use_symlinks=False
            )

        # Load processor and model if needed
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(cache_dir)
            
        if self.model is None:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                cache_dir,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
            )

        with torch.no_grad():
            if audio is not None:
                # Convert ComfyUI audio format to raw waveform
                waveform = audio["waveform"][0]  # Take first item from batch
                sample_rate = audio["sample_rate"]
                
                # Resample if needed
                if sample_rate != self.processor.feature_extractor.sampling_rate:
                    import torchaudio.functional as F
                    waveform = F.resample(
                        waveform, 
                        sample_rate, 
                        self.processor.feature_extractor.sampling_rate
                    )
                
                # Convert to numpy for librosa compatibility
                audio_np = waveform.mean(dim=0).cpu().numpy()  # Convert stereo to mono if needed
                
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio",
                                "audio": audio_np,  # Pass numpy array directly
                                "sampling_rate": self.processor.feature_extractor.sampling_rate  # Add explicit sampling rate
                            },
                            {"type": "text", "text": text},
                        ],
                    },
                ]
                
                text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                
                inputs = self.processor(
                    text=text,
                    audios=[audio_np],
                    return_tensors="pt",
                    padding=True
                )
                
            else:
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    },
                ]
                text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    padding=True
                )

            # Move inputs to correct device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            generate_ids = self.model.generate(
                **inputs,
                max_length=max_tokens,
                do_sample=True if seed > 0 else False
            )
            generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            if force_offload:
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return (response,) 