## comfyui-dreambait-nodes

A collection of forks, QoL nodes, and utilities for ComfyUI.

### Nodes Overview

#### Aspect Pad Image For Outpainting

A node to calculate arguments for the default ComfyUI node 'Pad Image For Outpainting' based on justifying and expanding to common SDXL and SD1.5 aspect ratios.

![](/examples/aspect_pad_for_outpainting.png)

#### Load Media

Loads media from various sources with advanced options for sequence handling.

- Supports images, directories, videos, archives (zip/tar/7z), and URLs.
- Smart sorting options including visual similarity-based ordering.
- Seamless loop creation with `loop_first_frame`.


#### Multi Min/Max

A node that finds the minimum or maximum value among multiple inputs. Forked from Impact Pack's MinMax node, extended to support up to 8 inputs. Accepts any comparable type and returns the min/max value of the same type.

#### String Item Menu

A context selector that extracts an item's index from a list of delimiter-separated strings.

#### TextPlusPlus

Combines text with optional prepend, append, and override operations.

#### NumberPlusPlus


A node to perform arithmetic operations on a number, including pre-addition, multiplication, and post-addition. It also provides formatted string outputs for the number, its integer representation, and a boolean indicating if the number is greater than 0.

#### NumberRemap

A node to remap a number from one range to another.

#### BoolPlusPlus

A node to handle boolean operations with additional features.

#### SwitchDuo

A node that switches between two outputs based on a boolean value. Supports overriding the boolean value with an optional input.

#### ListItemSelector

A node to select an item from a list based on a given index.

#### ImageFrameBlend

A node to blend frames of an image sequence.

#### Image Resize Face Aware

A fork of ComfyUI_essentials' Image Resize node with added face detection capabilities:
- All original resize methods (stretch, keep proportion, fill/crop, pad)
- New 'crop to face' method that detects faces, crops to them, and resizes
- New 'crop to face (keep_size)' method that produces a square crop centered on the largest detected face without resizing

Uses OpenCV's DNN face detector for reliable face detection, falling back to center crop if no face is found.

![](/examples/ImageResizeFaceAware.jpg)

#### Text Line Select

A node to select specific lines from a text input based on line numbers, with an optional random toggle.

#### Text Lines To List

Converts multi-line text input into a list, with each line as a separate item.

#### List Item Extract

Extracts a specific item from a list at the given index, with options for handling out-of-range indices.

#### MusicGen

A node that generates music using Meta's MusicGen model based on text prompts. This is a fork of 
comfyui-sound-lab's musicNode, modified to conform to ComfyUI's standard AUDIO format for better 
integration with the core audio workflow.

Features:
- Text-to-music generation using any MusicGen model:
  - Basic models:
    - small (300M parameters): Fastest, good for testing
    - medium (1.5B parameters): Balanced performance
    - large (3.3B parameters): Highest quality
  - Melody models:
    - melody: Base melody-conditioned generation
    - melody-large: High quality melody-conditioned generation
  - Stereo models:
    - stereo-small: Basic stereo generation
    - stereo-medium: Better stereo quality
    - stereo-large: Best stereo quality
  - Meta's dataset models:
    - small-fb, medium-fb, large-fb: New versions trained on Meta's dataset
- Adjustable duration (1-1000 seconds)
- Guidance scale control (0-20)
- Automatic model download from Hugging Face (downloads only the selected model)
- Fully compatible with ComfyUI's core audio nodes (VAEDecodeAudio, SaveAudio, PreviewAudio)
- Supports both CPU and CUDA acceleration

Credits: Original implementation from comfyui-sound-lab by anonymous author

The node outputs audio in the standard ComfyUI AUDIO format, compatible with audio preview and save nodes.

Models are stored in ComfyUI's models directory under 'musicgen/[model_name]' and are downloaded
automatically when first selected. You can also manually place model files in these directories
if you've downloaded them separately.

## Available Nodes

### Audio Processing

#### Normalize Audio
comfy Professional-grade audio normalization using broadcast standards (BS.1770-4) with true-peak limiting.

Inputs:
- `audio`: AUDIO format input
- `target_lufs`: Target integrated loudness (default: -14 LUFS for streaming)
- `true_peak_limit`: Maximum true-peak level (default: -1 dBTP)
- `normalize_type`: Choose between LUFS or peak normalization

Output:
- Normalized AUDIO

Example usage:
1. Connect your audio source to the NormalizeAudio node
2. Choose normalization type:
   - LUFS: For streaming/broadcast (measures perceived loudness)
   - Peak: For sample libraries or technical applications
3. Set target levels:
   - Streaming: -14 LUFS, -1 dBTP
   - Broadcast: -23 LUFS, -2 dBTP
   - Sample Library: Peak mode, -0.3 dBTP
4. Connect to output nodes

Technical Details:
- Uses BS.1770-4 standard for loudness measurement
- True-peak limiting prevents inter-sample peaks
- LUFS measurement considers perceived loudness
- Handles silence and very quiet audio gracefully
