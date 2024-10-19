from .utils import any_typ
import re

class ListItemSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": (
                    any_typ,
                    {
                        "default": None,
                        "defaultInput": True,
                        "tooltip": "Input list or string representation of a list"
                    },
                ),
                "item_indices": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Indices or patterns to select items from the list"
                    },
                )
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": True, "tooltip": "Enable debug print statements"})
            }
        }

    RETURN_TYPES = ("any_typ", "STRING", "INT")
    RETURN_NAMES = ("any", "string", "length")
    OUTPUT_NODE = True
    FUNCTION = "select_items"
    CATEGORY = "DRMBT nodes"

    def select_items(self, list_input=None, item_indices="", debug=True):
        def debug_print(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        debug_print(f"DEBUG: Incoming list_input: {list_input}")
        debug_print(f"DEBUG: Type of list_input: {type(list_input)}")
        debug_print(f"DEBUG: Incoming item_indices: {item_indices}")

        # Convert input to list
        if isinstance(list_input, list):
            prepared_list = list_input
            debug_print("DEBUG: Input is already a list")
        elif isinstance(list_input, str):
            debug_print("DEBUG: Input is a string")
            if list_input.startswith('[') and list_input.endswith(']'):
                prepared_list = eval(list_input)  # Caution: eval can be dangerous
                debug_print("DEBUG: String input converted to list using eval")
            else:
                prepared_list = re.split(r'[,\s\t\n]+', list_input.strip())
                debug_print("DEBUG: String input split into list")
        else:
            debug_print(f"DEBUG: Unexpected input type: {type(list_input)}")
            return {"any": "", "string": "", "length": 0}

        debug_print(f"DEBUG: Prepared list: {prepared_list}")
        debug_print(f"DEBUG: Type of prepared list: {type(prepared_list)}")

        # Parse item_indices
        selected_items = []
        indices = re.split(r'[,\s]+', item_indices.strip())
        debug_print(f"DEBUG: Parsed indices: {indices}")

        for index in indices:
            index = index.strip()
            debug_print(f"DEBUG: Processing index: {index}")
            if ':' in index:  # Slice notation
                parts = index.split(':')
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if len(parts) > 1 and parts[1] else None
                step = int(parts[2]) if len(parts) > 2 and parts[2] else None
                selected_items.extend(prepared_list[start:end:step])
                debug_print(f"DEBUG: Slice {start}:{end}:{step} added: {prepared_list[start:end:step]}")
            elif '*' in index:  # Wildcard
                pattern = re.compile(index.replace('*', '.*').lower())
                matched_items = [item for item in prepared_list if pattern.search(str(item).lower())]
                selected_items.extend(matched_items)
                debug_print(f"DEBUG: Wildcard pattern '{index}' matched: {matched_items}")
            else:  # Single index (including negative)
                try:
                    idx = int(index)
                    item = prepared_list[idx]
                    selected_items.append(item)
                    debug_print(f"DEBUG: Single index {idx} added: {item}")
                except (ValueError, IndexError):
                    debug_print(f"DEBUG: Invalid index or out of range: {index}")

        debug_print(f"DEBUG: Final selected items: {selected_items}")

        # Prepare outputs
        if not selected_items:
            debug_print("DEBUG: No items selected")
            return {"any": "", "string": "", "length": 0}
        
        if len(selected_items) == 1:
            single_item = selected_items[0]
            try:
                any_output = float(single_item) if '.' in str(single_item) else single_item
            except ValueError:
                any_output = single_item
            string_output = str(single_item).strip('\'"')
            debug_print(f"DEBUG: Single item output - any: {any_output}, string: {string_output}")
        else:
            any_output = selected_items
            string_output = ' '.join(map(lambda x: str(x).strip('\'"'), selected_items))
            debug_print(f"DEBUG: Multiple items output - any: {any_output}, string: {string_output}")

        length_output = len(selected_items)
        debug_print(f"DEBUG: Length of selected items: {length_output}")
        return {
            "result": (any_output, string_output, length_output)
        }