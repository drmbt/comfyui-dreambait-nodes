import os
import sys
import re
import markdown2

class MarkdownInteractive:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("markdown_text", "plain_text")
    FUNCTION = "process"
    CATEGORY = "dreambait/text"

    def process(self, text):
        # Return the original markdown text as the first output
        markdown_text = text
        
        # Convert markdown to plain text by removing markdown syntax
        # First convert to HTML
        html = markdown2.markdown(text)
        # Then strip HTML tags
        plain_text = re.sub(r'<[^>]+>', '', html)
        # Remove extra whitespace
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()
        
        return (markdown_text, plain_text)


NODE_CLASS_MAPPINGS = {
    "MarkdownInteractive": MarkdownInteractive,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MarkdownInteractive": "Markdown Interactive 📝",
} 