class DRMBT_String_Item_Menu:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "menu_list": ("STRING", {"tooltip": "Input string to be split into menu items."}),
                "delimiter": ("STRING", {"default": ",", "tooltip": "Delimiter to split the input string."}),
                "selected_item": ("STRING", {"tooltip": "Selected item from the menu."})
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    FUNCTION = "select_model_context"

    CATEGORY = "model_context"
    DESCRIPTION = "Selects a model context from a dynamically generated list and outputs the selected value and its index."

    def select_model_context(self, menu_list, delimiter=",", selected_item=""):
        model_context_list = menu_list.split(delimiter)
        if selected_item not in model_context_list:
            selected_item = model_context_list[0]  # Default to the first item if the selected item is not in the list
        selected_index = model_context_list.index(selected_item)
        return (selected_item, selected_index)