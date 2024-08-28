---
updated: 2024-08-22T00:08:10+02:00
---

## comfyui-dreambait-nodes

A collection of forks, QoL nodes and utilities for ComfyUI

### Aspect Pad Image For Outpainting

A node to calculate args for default comfy node 'Pad Image For Outpainting' based on justifying and expanding to common SDXL and SD1.5 aspect ratios

![](/examples/aspect_pad_for_outpainting.png)

### Load Media

LoadMedia class for loading images, and videos as image sequences.

    This class provides functionality to load images from a file path, which can be a single image,
    a directory of images, or a video file (.mp4, .mov). It supports extracting frames from video files
    and treating them as image sequences.

    Disclaimer:
    There is a known issue with resizing mismatched resolution images. If using a path to a directory,
    make sure all images are the same size to avoid potential issues.

    TO-DO:
    - properly handle mismatched tensor sizes in dir with different resolution assets
    - fork in [Crystools](https://github.com/crystian/ComfyUI-Crystools) style metadata extractor
    
### Multi Min/Max

a fork of [Impact-Pack MinMax](https://github.com/ltdrdata/ComfyUI-Impact-Pack) that can accept up to 8 values

### String Item Menu

a context selector that extracts an item's index from a list of delimiter separated strings
