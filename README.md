
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
