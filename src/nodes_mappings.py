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
from .nodes.utils import DynamicDictionary, StringToDict, DictToOutputs, DynamicStringConcatenate
from .nodes.folder_opener import DreambaitFolderOpener
from .nodes.draw_text_drmbt import DrawText, TextMargins, TextBoxStyle, TextShadow
from .nodes.draw_icons import DrawMana
from .nodes.compare_image_similarity import CompareImageSimilarity
from .nodes.shot_history import ShotHistory

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
    "DynamicDictionary": DynamicDictionary,
    "StringToDict": StringToDict,
    "DictToOutputs": DictToOutputs,
    "DynamicStringConcatenate": DynamicStringConcatenate,
    "DreambaitFolderOpener": DreambaitFolderOpener,
    "DrawText": DrawText,
    "TextMargins": TextMargins,
    "TextBoxStyle": TextBoxStyle,
    "TextShadow": TextShadow,
    "DrawMana": DrawMana,
    "CompareImageSimilarity": CompareImageSimilarity,
    "ShotHistory": ShotHistory
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
    "DynamicDictionary": "Dynamic Dictionary",
    "StringToDict": "Smart Dictionary from String 📝",
    "DictToOutputs": "Dictionary to Outputs 🔀",
    "DynamicStringConcatenate": "Dynamic String Concatenate 🔗",
    "DreambaitFolderOpener": "Dreambait Folder Opener 📁",
    "DrawText": "Draw Text 📝",
    "TextMargins": "Text Margins",
    "TextBoxStyle": "Text Box Style",
    "TextShadow": "Text Shadow",
    "DrawMana": "Draw Mana Symbols 🎴",
    "CompareImageSimilarity": "Compare Image Similarity 🔍",
    "ShotHistory": "Shot History"
} 
