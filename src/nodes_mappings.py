from .nodes.aspect_pad_for_outpainting import AspectPadImageForOutpainting
from .nodes.string_item_menu import DRMBT_String_Item_Menu
from .nodes.DRMBT_MultiMinMax import DRMBT_MultiMinMax
from .nodes.drmbt_load_media import LoadMedia
from .nodes.TextPlusPlus import TextPlusPlus
from .nodes.NumberPlusPlus import NumberPlusPlus
from .nodes.NumberRemap import NumberRemap
from .nodes.BoolPlusPlus import BoolPlusPlus
from .nodes.SwitchDuo import SwitchDuo
from .nodes.ListItemSelector import ListItemSelector
from .nodes.ImageFrameBlend import ImageFrameBlend
from .nodes.TextLineSelect import TextLineSelect
from .nodes.TextLinesToList import TextLinesToList
from .nodes.ListItemExtract import ListItemExtract
from .nodes.drmbt_image_resize import ImageResizeFaceAware
from .nodes.MusicGen import MusicGen
from .nodes.NormalizeAudio import NormalizeAudio
from .nodes.minicpm_v_node import MiniCPMVNode, DownloadAndLoadMiniCPMV
from .nodes.qwen2_audio_instruct import Qwen2AudioInstruct
from .nodes.LoadAudioPlus import LoadAudioPlus, PreviewAudioPlus, AudioInfoPlus
from .nodes.utils import DynamicDictionary

NODE_CLASS_MAPPINGS = {
    "DRMBT_AspectPadImageForOutpainting": AspectPadImageForOutpainting,
    "DRMBT_String_Item_Menu": DRMBT_String_Item_Menu,
    "DRMBT_MultiMinMax": DRMBT_MultiMinMax,
    "DRMBT_LoadMedia": LoadMedia,
    "TextPlusPlus": TextPlusPlus,
    "NumberPlusPlus": NumberPlusPlus,
    "NumberRemap": NumberRemap,
    "BoolPlusPlus": BoolPlusPlus,
    "SwitchDuo": SwitchDuo,
    "ListItemSelector": ListItemSelector,
    "ImageFrameBlend": ImageFrameBlend,
    "TextLineSelect": TextLineSelect,
    "TextLinesToList": TextLinesToList,
    "ListItemExtract": ListItemExtract,
    "ImageResizeFaceAware": ImageResizeFaceAware,
    "MusicGen": MusicGen,
    "NormalizeAudio": NormalizeAudio,
    "MiniCPMVNode": MiniCPMVNode,
    "DownloadAndLoadMiniCPMV": DownloadAndLoadMiniCPMV,
    "Qwen2AudioInstruct": Qwen2AudioInstruct,
    "LoadAudioPlus": LoadAudioPlus,
    "AudioInfoPlus": AudioInfoPlus,
    "DynamicDictionary": DynamicDictionary
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DRMBT_AspectPadImageForOutpainting": "Aspect Pad For Outpainting",
    "DRMBT_MultiMinMax": "Multi Min/Max",
    "DRMBT_String_Item_Menu": "String Item Menu",
    "DRMBT_LoadMedia":"Load Media (img, vid or dir)",
    "TextPlusPlus": "Text ++",
    "NumberPlusPlus": "Number ++",
    "NumberRemap": "Number Remap Range",
    "BoolPlusPlus": "BOOL ++",
    "SwitchDuo": "Switch Duo",
    "ListItemSelector": "List Item Selector",
    "ImageFrameBlend": "Frame Blend",
    "TextLineSelect": "Text Line Select",
    "TextLinesToList": "Text Lines To List",
    "ImageResizeFaceAware": "Image Resize Face Aware",
    "MusicGen": "MusicGen",
    "NormalizeAudio": "Normalize Audio",
    "MiniCPMVNode": "MiniCPM-V Q&A",
    "DownloadAndLoadMiniCPMV": "Load MiniCPM-V Model",
    "Qwen2AudioInstruct": "Qwen2 Audio Q&A",
    "LoadAudioPlus": "Load Audio Plus +",
    "AudioInfoPlus": "Audio Info Plus + 🎵",
    "DynamicDictionary": "Dynamic Dictionary"
}