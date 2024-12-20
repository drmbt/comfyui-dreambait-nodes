import torch
import math
import torch.nn.functional as F

# Try to import RIFE, but don't fail if not available
try:
    from vfi_models.rife import RIFE_VFI
    RIFE_AVAILABLE = True
except ImportError:
    print("RIFE frame interpolation not available - install ComfyUI-Frame-Interpolation for enhanced interpolation")
    RIFE_AVAILABLE = False

class ImageFrameBlend:
    @classmethod
    def INPUT_TYPES(s):
        base_inputs = {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image sequence/batch to be time-remapped"
                }),
                "target_frames": ("INT", { 
                    "default": 16, 
                    "min": 1, 
                    "step": 1,
                    "tooltip": "Desired number of output frames. Can be more or less than input frames."
                }),
                "blend_strength": ("FLOAT", { 
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "How much to blend between frames. 0 = nearest frame, 1 = full interpolation"
                }),
                "method": (["linear", "cosine", "nearest"], {
                    "tooltip": "Interpolation method: linear (simple), cosine (smooth), or nearest (no blending)"
                }),
                "loop_seamless": ("BOOLEAN", { 
                    "default": False,
                    "tooltip": "Preserve seamless looping sequences by interpolating between last and first frame"
                }),
            }
        }
        if RIFE_AVAILABLE:
            base_inputs["required"]["use_rife"] = ("BOOLEAN", {
                "default": True,
                "tooltip": "Use RIFE AI frame interpolation when expanding frame count (higher quality but slower)"
            })
        return base_inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/animation"
    DESCRIPTION = """
    Adjusts the number of frames in an image sequence using various time remapping interpolation methods.

    Methods:
    - linear: Simple linear interpolation between frames
    - cosine: Smoother transitions using cosine interpolation
    - nearest: Smart frame selection without blending
    - RIFE: AI-powered frame interpolation (if ComfyUI-Frame-Interpolation package is installed)

    Features:
    - Seamless loop option for preserving looping sequences
    - Blend strength control for mixing interpolated and nearest frames
    - Can expand or reduce frame count
    - smart frame selection for smooth distribution of frames

    Note: RIFE option only appears if ComfyUI-Frame-Interpolation is installed."""
    
    def cosine_interpolation(self, x):
        """Cosine interpolation for smoother blending"""
        return (1 - torch.cos(x * math.pi)) / 2

    def smart_frame_selection(self, orig_size, target_size):
        """Intelligently select frames to keep when reducing, or duplicate when expanding"""
        if target_size >= orig_size:
            # For expansion, calculate how many times to repeat each frame
            repeats = torch.zeros(orig_size, dtype=torch.int)
            base_repeat = target_size // orig_size
            remainder = target_size % orig_size
            
            # Start with base repeats for all frames
            repeats += base_repeat
            
            if remainder > 0:
                # Distribute remaining repeats evenly, prioritizing middle frames
                if orig_size > 2:
                    middle_indices = torch.linspace(0, orig_size-1, remainder).round().long()
                    repeats[middle_indices] += 1
                else:
                    repeats[0] += remainder

            # Generate frame indices
            indices = []
            for i, r in enumerate(repeats):
                indices.extend([i] * r.item())
            
            return torch.tensor(indices)
        else:
            # For reduction, keep frames at regular intervals
            # Always include first and last frame
            if target_size == 1:
                return torch.tensor([0])
            elif target_size == 2:
                return torch.tensor([0, orig_size-1])
            else:
                # Calculate middle frame indices
                middle_indices = torch.linspace(1, orig_size-2, target_size-2).round().long()
                return torch.cat([torch.tensor([0]), middle_indices, torch.tensor([orig_size-1])])

    def execute(self, image, target_frames, blend_strength, method="linear", loop_seamless=False, use_rife=True):
        orig_size = image.shape[0]

        # Special handling for single image input - immediately return repeated image
        if orig_size == 1:
            return (image.repeat(target_frames, 1, 1, 1),)

        if orig_size == target_frames:
            return (image,)

        if target_frames <= 1:
            return (image[:1],)

        # For looping, we'll treat the sequence as continuous rather than adding a frame
        if loop_seamless and orig_size > 1:
            # Ensure we have at least 2 frames for looping
            target_frames = max(2, target_frames)
            # No need to make divisible when reducing frames
            if target_frames > orig_size:
                target_frames = (target_frames // orig_size) * orig_size

        # Try RIFE first if enabled and available
        if RIFE_AVAILABLE and use_rife and target_frames > orig_size:
            try:
                rife = RIFE_VFI()
                
                if loop_seamless:
                    # For looping, include first frame at end for interpolation
                    frames_for_rife = torch.cat([image, image[:1]], dim=0)
                else:
                    frames_for_rife = image

                # Calculate proper multiplier to reach target frames
                required_multiplier = max(2, math.ceil((target_frames - 1) / (orig_size - 1)))
                
                interpolated = rife.vfi(
                    ckpt_name="rife47.pth",
                    frames=frames_for_rife,
                    multiplier=required_multiplier,
                    clear_cache_after_n_frames=10,
                    fast_mode=True,
                    ensemble=True,
                    scale_factor=1.0
                )[0]

                # Now we need to resample to exactly match target_frames
                if loop_seamless:
                    # For looping, we want to interpolate in a circular fashion
                    # First, remove the duplicate frame at the end
                    interpolated = interpolated[:-1]
                    # Then sample circularly
                    indices = torch.linspace(0, interpolated.shape[0], target_frames + 1)[:-1]
                    indices = indices % interpolated.shape[0]
                else:
                    indices = torch.linspace(0, interpolated.shape[0] - 1, target_frames)
                
                indices_low = indices.floor().long()
                indices_high = (indices.ceil().long() % interpolated.shape[0])
                weights = (indices - indices_low).view(-1, 1, 1, 1)
                interpolated = (
                    interpolated[indices_low] * (1 - weights) + 
                    interpolated[indices_high] * weights
                )

                # Apply blend strength
                if blend_strength < 1.0:
                    nearest = torch.empty_like(interpolated)
                    for i in range(target_frames):
                        if loop_seamless:
                            nearest_idx = (i * orig_size // target_frames) % orig_size
                        else:
                            nearest_idx = min(i * orig_size // target_frames, orig_size - 1)
                        nearest[i] = image[nearest_idx]
                    interpolated = blend_strength * interpolated + (1 - blend_strength) * nearest

                return (interpolated,)
            except Exception as e:
                print(f"RIFE interpolation failed, falling back to {method}: {str(e)}")

        # Handle nearest-neighbor frame selection
        if method == "nearest":
            if loop_seamless:
                # Use modulo for circular frame selection
                indices = (torch.linspace(0, orig_size, target_frames) % orig_size).long()
                return (image[indices],)
            else:
                indices = self.smart_frame_selection(orig_size, target_frames)
                return (image[indices],)

        # Fallback to linear/cosine interpolation
        out = torch.empty([target_frames] + list(image.shape)[1:], dtype=image.dtype, device=image.device)
        
        if loop_seamless:
            # For looping, interpolate in a circular fashion
            positions = torch.linspace(0, orig_size, target_frames + 1)[:-1]
            for i in range(target_frames):
                pos = positions[i]
                idx_low = int(pos % orig_size)
                idx_high = int((pos + 1) % orig_size)
                w = pos - int(pos)
                
                if method == "cosine":
                    w = self.cosine_interpolation(torch.tensor(w))
                
                w = w * blend_strength
                out[i] = (1 - w) * image[idx_low] + w * image[idx_high]
        else:
            # Original non-looping interpolation code...
            if target_frames > orig_size:
                frame_indices = torch.linspace(0, target_frames-1, orig_size).round().long()
                out[frame_indices] = image

                for i in range(orig_size - 1):
                    start_idx = frame_indices[i]
                    end_idx = frame_indices[i + 1]
                    if end_idx - start_idx > 1:
                        steps = end_idx - start_idx
                        weights = torch.linspace(0, 1, steps + 1)[1:-1]
                        
                        if method == "cosine":
                            weights = self.cosine_interpolation(weights)
                        
                        weights = weights * blend_strength
                        
                        for j, w in enumerate(weights, 1):
                            out[start_idx + j] = (1 - w) * image[i] + w * image[i + 1]
            else:
                scale = (orig_size - 1) / (target_frames - 1)
                
                for i in range(target_frames):
                    pos = i * scale
                    idx_low = int(pos)
                    idx_high = min(idx_low + 1, orig_size - 1)
                    w = pos - idx_low
                    
                    if method == "cosine":
                        w = self.cosine_interpolation(torch.tensor(w))
                    
                    w = w * blend_strength
                    out[i] = (1 - w) * image[idx_low] + w * image[idx_high]

        return (out,)

