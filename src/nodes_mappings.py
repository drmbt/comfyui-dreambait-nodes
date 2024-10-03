# src/nodes_mappings.py
from .nodes.aspect_pad_for_outpainting import AspectPadImageForOutpainting
from .nodes.model_context_selector import DRMBT_Model_Context_Selector
from .nodes.string_item_menu import DRMBT_String_Item_Menu
from .nodes.DRMBT_MultiMinMax import DRMBT_MultiMinMax
from .nodes.drmbt_load_media import LoadMedia
from .nodes.plot_parameters import PlotParametersDRMBT
from .nodes.DRMBT_execution_time import ExecutionTime
from .nodes.TextPlusPlus import TextPlusPlus
from .nodes.NumberPlusPlus import NumberPlusPlus
from .nodes.NumberRemap import NumberRemap
from .nodes.BoolPlusPlus import BoolPlusPlus
from .nodes.SwitchDuo import SwitchDuo

NODE_CLASS_MAPPINGS = {
    "DRMBT_AspectPadImageForOutpainting": AspectPadImageForOutpainting,
    "DRMBT_Model_Context_Selector": DRMBT_Model_Context_Selector,  # Keep class reference
    "DRMBT_String_Item_Menu": DRMBT_String_Item_Menu,
    "DRMBT_MultiMinMax": DRMBT_MultiMinMax,
    "DRMBT_LoadMedia": LoadMedia,
    "DRMBT_PlotParameters": PlotParametersDRMBT,
    "DRMBT_ExecutionTime" : ExecutionTime,
    "TextPlusPlus": TextPlusPlus,
    "NumberPlusPlus": NumberPlusPlus,
    "NumberRemap": NumberRemap,
    "BoolPlusPlus": BoolPlusPlus,
    "SwitchDuo": SwitchDuo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DRMBT_AspectPadImageForOutpainting": "Aspect Pad For Outpainting",
    "DRMBT_Model_Context_Selector": "Model Context Selector",
    "DRMBT_MultiMinMax": "Multi Min/Max",
    "DRMBT_String_Item_Menu": "String Item Menu",
    "DRMBT_LoadMedia":"Load Media (img, vid or dir)",
    "DRMBT_PlotParameters": "Plot Parameters",
    "DRMBT_ExecutionTime" : "DRMBT_ExecutionTime",
    "TextPlusPlus": "Text ++",
    "NumberPlusPlus": "Number ++",
    "NumberRemap": "Number Remap Range",
    "BoolPlusPlus": "BOOL ++",
    "SwitchDuo": "Switch Duo"
}