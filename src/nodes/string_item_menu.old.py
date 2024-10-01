class DRMBT_String_Item_Menu:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "menu": ("STRING", {"tooltip": "Input string to be split into menu items."}),
                "delimiter": ("STRING", {"default": ",", "tooltip": "Delimiter to split the input string."}),
                "select_item": ("STRING", {"default": '', "tooltip": "Selected item from the menu."})
            },
            "optional": {
                "int_select": ("INT", {"default": None, "min": 0, "step": 1, "defaultInput": True})
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("output_string", "index")
    FUNCTION = "parse_list"
    CATEGORY = "string selection"
    DESCRIPTION = "returns a string or int by selection"
    TOOLTIPS = {
        "input": { 
            "menu": "The text to be split into a list by the delimiter.",
            "delimiter": "The delimiter used to split the text into a list.",
            "select_item": "The string to be matched against the list items.",
            "int_select": "optional int to select string item"
        },
        "output": ("The original input string and the index of the matching item in the list.",)
    }

    def parse_list(self, menu, delimiter, select_item, int_select=None):
        menu_list = menu.split(delimiter)
        if int_select is None:
            if select_item not in menu_list:
                select_item = menu_list[0]  # Default to the first item if the selected item is not in the list
            select_index = menu_list.index(select_item)
        else:
            select_item = menu_list[int_select]
            select_index = int_select
        return (select_item, select_index)