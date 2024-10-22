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
    "ListItemSelector": ListItemSelector
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
    "ListItemSelector": "List Item Selector"
}