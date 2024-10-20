# src/nodes_mappings.py
from .nodes.aspect_pad_for_outpainting import AspectPadImageForOutpainting
from .nodes.string_item_menu import DRMBT_String_Item_Menu
from .nodes.DRMBT_MultiMinMax import DRMBT_MultiMinMax
from .nodes.drmbt_load_media import LoadMedia
from .nodes.plot_parameters import PlotParametersDRMBT
from .nodes.TextPlusPlus import TextPlusPlus
from .nodes.NumberPlusPlus import NumberPlusPlus
from .nodes.NumberRemap import NumberRemap
from .nodes.BoolPlusPlus import BoolPlusPlus
from .nodes.SwitchDuo import SwitchDuo
from .nodes.ListItemSelector import ListItemSelector

NODE_CLASS_MAPPINGS = {
    "DRMBT_AspectPadImageForOutpainting": AspectPadImageForOutpainting,
    "DRMBT_String_Item_Menu": DRMBT_String_Item_Menu,
    "DRMBT_MultiMinMax": DRMBT_MultiMinMax,
    "DRMBT_LoadMedia": LoadMedia,
    "DRMBT_PlotParameters": PlotParametersDRMBT,
    "TextPlusPlus": TextPlusPlus,
    "NumberPlusPlus": NumberPlusPlus,
    "NumberRemap": NumberRemap,
    "BoolPlusPlus": BoolPlusPlus,
    "SwitchDuo": SwitchDuo,
    "ListItemSelector": ListItemSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DRMBT_AspectPadImageForOutpainting": "Aspect Pad For Outpainting",
    "DRMBT_MultiMinMax": "Multi Min/Max",
    "DRMBT_String_Item_Menu": "String Item Menu",
    "DRMBT_LoadMedia":"Load Media (img, vid or dir)",
    "DRMBT_PlotParameters": "Plot Parameters",
    "TextPlusPlus": "Text ++",
    "NumberPlusPlus": "Number ++",
    "NumberRemap": "Number Remap Range",
    "BoolPlusPlus": "BOOL ++",
    "SwitchDuo": "Switch Duo",
    "ListItemSelector": "List Item Selector"
}