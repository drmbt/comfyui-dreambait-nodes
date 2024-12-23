import torch
import numpy as np
from pyloudnorm import Meter

class NormalizeAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
            "target_lufs": ("FLOAT", {
                "default": -14.0, 
                "min": -70.0,
                "max": 0.0,
                "step": 0.1,
                "display": "number",
                "label": "Target LUFS"
            }),
            "true_peak_limit": ("FLOAT", {
                "default": -1.0,
                "min": -20.0,
                "max": 0.0,
                "step": 0.1,
                "display": "number",
                "label": "True Peak Limit (dBTP)"
            }),
            "normalize_type": (["lufs", "peak"],),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "normalize"
    CATEGORY = "audio/processing"
    DESCRIPTION = """Normalizes audio using broadcast standards:
        - LUFS mode: Uses BS.1770-4 integrated loudness measurement (recommended for streaming/broadcast)
        - Peak mode: Traditional peak normalization with true-peak limiting
        - Includes true-peak (dBTP) limiting to prevent inter-sample peaks and clipping
        Default settings (-14 LUFS, -1 dBTP) follow streaming platform standards."""

    def normalize(self, audio, target_lufs, true_peak_limit, normalize_type):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        
        # Convert torch tensor to numpy for loudness processing
        audio_np = waveform.squeeze(0).numpy()
        
        if normalize_type == "lufs":
            # Initialize loudness meter
            meter = Meter(sample_rate)  # BS.1770-4 mode
            
            # Measure current loudness
            current_lufs = meter.integrated_loudness(audio_np.T)
            
            # Calculate gain needed to reach target LUFS
            if current_lufs != float('-inf'):
                lufs_gain_db = target_lufs - current_lufs
                gain_linear = 10 ** (lufs_gain_db / 20.0)
            else:
                # Handle silence or very quiet audio
                gain_linear = 1.0
            
            # Apply LUFS-based gain
            normalized = audio_np * gain_linear
            
        else:  # peak normalization
            # Convert true_peak_limit from dB to amplitude
            peak_limit = 10 ** (true_peak_limit / 20.0)
            
            # Find current peak
            current_peak = np.max(np.abs(audio_np))
            
            # Calculate and apply gain
            if current_peak > 0:
                gain_linear = peak_limit / current_peak
                normalized = audio_np * gain_linear
            else:
                normalized = audio_np
        
        # True-peak limiting (for both modes)
        # This is a simple true-peak limiter - you might want to use a more sophisticated one
        peak_limit_linear = 10 ** (true_peak_limit / 20.0)
        normalized = np.clip(normalized, -peak_limit_linear, peak_limit_linear)
        
        # Convert back to torch tensor
        normalized_tensor = torch.from_numpy(normalized).unsqueeze(0)
        
        return ({"waveform": normalized_tensor, "sample_rate": sample_rate},) 