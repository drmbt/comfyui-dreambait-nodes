import markdown2

class MarkdownRender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "render"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def render(self, text, unique_id=None, extra_pnginfo=None):
        # Convert markdown to HTML
        html_content = []
        for t in text:
            if isinstance(t, list):
                for item in t:
                    html_content.append(markdown2.markdown(str(item), extras=["fenced-code-blocks", "tables"]))
            else:
                html_content.append(markdown2.markdown(str(t), extras=["fenced-code-blocks", "tables"]))

        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = html_content

        # Send both the original text and HTML content
        return {
            "ui": {
                "text": text,  # Keep original text for compatibility
                "html": html_content  # Add HTML content for rendering
            },
            "result": (text,)
        }


NODE_CLASS_MAPPINGS = {
    "MarkdownRender": MarkdownRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MarkdownRender": "Markdown Render 📝",
} 