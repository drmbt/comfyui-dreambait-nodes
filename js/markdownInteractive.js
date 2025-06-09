import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

class MarkdownInteractiveWidget extends ComfyWidgets.ComfyWidget {
    constructor(node, inputName, inputData, app) {
        super(node, inputName, inputData, app);
        this.isEditing = false;
        this.originalValue = "";
    }

    onNodeCreated() {
        // Create the container for both textarea and preview
        this.container = document.createElement("div");
        this.container.style.display = "flex";
        this.container.style.flexDirection = "column";
        this.container.style.width = "100%";
        this.container.style.height = "100%";
        this.container.style.gap = "8px";

        // Create the textarea
        this.textarea = document.createElement("textarea");
        this.textarea.style.width = "100%";
        this.textarea.style.height = "100px";
        this.textarea.style.resize = "vertical";
        this.textarea.style.padding = "8px";
        this.textarea.style.border = "1px solid var(--border-color)";
        this.textarea.style.borderRadius = "4px";
        this.textarea.style.backgroundColor = "var(--comfy-input-bg)";
        this.textarea.style.color = "var(--comfy-menu-text)";
        this.textarea.style.fontFamily = "monospace";
        this.style.fontSize = "14px";
        this.style.lineHeight = "1.4";
        this.textarea.style.display = "none"; // Initially hidden

        // Create the preview div
        this.preview = document.createElement("div");
        this.preview.style.width = "100%";
        this.preview.style.minHeight = "100px";
        this.preview.style.padding = "8px";
        this.preview.style.border = "1px solid var(--border-color)";
        this.preview.style.borderRadius = "4px";
        this.preview.style.backgroundColor = "var(--comfy-input-bg)";
        this.preview.style.color = "var(--comfy-menu-text)";
        this.preview.style.fontSize = "14px";
        this.preview.style.lineHeight = "1.4";
        this.preview.style.overflowY = "auto";
        this.preview.style.maxHeight = "300px";
        this.preview.style.cursor = "pointer";

        // Add both elements to the container
        this.container.appendChild(this.textarea);
        this.container.appendChild(this.preview);

        // Add the container to the widget
        this.widget.appendChild(this.container);

        // Set up event listeners
        this.textarea.addEventListener("focus", () => {
            this.isEditing = true;
            this.originalValue = this.textarea.value;
        });

        this.textarea.addEventListener("blur", () => {
            this.isEditing = false;
            if (this.textarea.value !== this.originalValue) {
                this.node.setDirtyCanvas(true);
                this.node.setDirtyFlow(true);
            }
            this.updatePreview();
        });

        this.textarea.addEventListener("input", () => {
            this.updatePreview();
        });

        // Add click handler to preview to start editing
        this.preview.addEventListener("click", () => {
            this.startEditing();
        });

        // Initial setup
        this.updatePreview();
    }

    startEditing() {
        this.textarea.style.display = "block";
        this.preview.style.display = "none";
        this.textarea.focus();
    }

    updatePreview() {
        // Show preview and hide textarea when not editing
        if (!this.isEditing) {
            this.textarea.style.display = "none";
            this.preview.style.display = "block";
        }

        // Convert markdown to HTML using the backend's markdown processor
        const html = this.textarea.value;
        this.preview.innerHTML = html;
    }

    getValue() {
        return this.textarea.value;
    }

    setValue(value) {
        this.textarea.value = value;
        this.updatePreview();
    }
}

// Register the widget
app.registerWidgetType("MarkdownInteractive", MarkdownInteractiveWidget); 