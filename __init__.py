import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from img_utils.img_nodes import *

NODE_CLASS_MAPPINGS = {
   
    "Aspect Pad Image For Outpainting": AspectPadImageForOutpainting
}

