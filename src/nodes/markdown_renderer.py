import markdown2


class MarkdownRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "render_markdown"
    CATEGORY = "storyboard"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

    def render_markdown(self, unique_id=None, extra_pnginfo=None, text=None):
        print(f"[MarkdownRenderer] Processing markdown: {text}")

        # Handle input - if no input connected, we'll get default content from the frontend
        html_content = []
        text_content = []

        if text is not None:
            # Process input from connected nodes
            for t in text:
                if isinstance(t, list):
                    for item in t:
                        text_str = str(item)
                        text_content.append(text_str)
                        # Convert markdown to HTML using markdown2 with extras
                        html_content.append(
                            markdown2.markdown(
                                text_str,
                                extras=[
                                    "fenced-code-blocks",
                                    "tables",
                                    "code-friendly",
                                ],
                            )
                        )
                else:
                    text_str = str(t)
                    text_content.append(text_str)
                    html_content.append(
                        markdown2.markdown(
                            text_str,
                            extras=["fenced-code-blocks", "tables", "code-friendly"],
                        )
                    )
        else:
            # No input connected - will be handled by frontend editing
            text_content = [""]
            html_content = [""]

        print(f"[MarkdownRenderer] Converted {len(html_content)} items to HTML")

        # Store in workflow metadata if available
        if unique_id is not None and extra_pnginfo is not None:
            try:
                if isinstance(extra_pnginfo, list) and len(extra_pnginfo) > 0:
                    if (
                        isinstance(extra_pnginfo[0], dict)
                        and "workflow" in extra_pnginfo[0]
                    ):
                        workflow = extra_pnginfo[0]["workflow"]
                        node = next(
                            (
                                x
                                for x in workflow["nodes"]
                                if str(x["id"]) == str(unique_id[0])
                            ),
                            None,
                        )
                        if node:
                            node["widgets_values"] = html_content
            except Exception as e:
                print(f"[MarkdownRenderer] Error updating workflow metadata: {e}")

        # Return both UI data for frontend and result for output
        return {
            "ui": {"text": text_content, "html": html_content},
            "result": (text_content,),
        }
