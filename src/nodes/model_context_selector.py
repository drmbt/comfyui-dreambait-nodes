class DRMBT_Model_Context_Selector:
    model_context_list = ["SDXL", "SD15", "SD3", "FLUX"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_context": (cls.model_context_list, {"tooltip": "Select the model context."})
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    FUNCTION = "select_model_context"

    CATEGORY = "model_context"
    DESCRIPTION = "Selects a model context from a predefined list and outputs the selected value and its index."

    def select_model_context(self, model_context):
        selected_index = self.model_context_list.index(model_context)
        return (model_context, selected_index)