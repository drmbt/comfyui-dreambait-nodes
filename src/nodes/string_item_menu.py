from .utils import any_typ
class DRMBT_String_Item_Menu:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "menu": (
                    "STRING", 
                    {
                        "default": "",
                        "multiline": True, 
                        "tooltip": "The text to be split into a list by the delimiter."
                    }
                ),
                "delimiter": (
                    "STRING", 
                    {
                        "default": ",", 
                        "tooltip": "Delimiter to split the input string."
                    }
                ),
                "select_item": (
                    "STRING", 
                    {
                        "default": '', 
                        "tooltip": "Select item and index by name from the menu."
                    }
                )
            },
            "optional": {
                "int_select": (
                    "INT", 
                    {
                        "default": -1, 
                        "min": 0, 
                        "step": 1, 
                        "forceInput": True,
                        "tooltip": "Input string to be split into menu items."
                    }
                )
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_string", "index")
    FUNCTION = "parse_list"
    CATEGORY = "string selection"
    DESCRIPTION = "returns a string or int by selection"

    def parse_list(self, menu, delimiter, select_item, int_select=-1):
        menu_list = menu.split(delimiter)
        if int_select <0:
            if select_item not in menu_list:
                select_item = menu_list[0]  # Default to the first item if the selected item is not in the list
            select_index = menu_list.index(select_item)
        else:
            select_item = menu_list[int_select]
            select_index = int_select
        return (select_item, select_index)