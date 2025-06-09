import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays input text as rendered markdown on a node

app.registerExtension({
    name: "dreambait.MarkdownRender",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MarkdownRender") {
            // Add custom CSS for markdown styling
            const style = document.createElement('style');
            style.textContent = `
                .markdown-content {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #e0e0e0;
                    padding: 10px;
                    width: 100%;
                    height: 100%;
                    background: #1a1a1a;
                    border-radius: 4px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                    overflow-y: auto;
                    overflow-x: hidden;
                    box-sizing: border-box;
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                }
                .markdown-content::-webkit-scrollbar {
                    width: 8px;
                }
                .markdown-content::-webkit-scrollbar-track {
                    background: #2a2a2a;
                    border-radius: 4px;
                }
                .markdown-content::-webkit-scrollbar-thumb {
                    background: #444;
                    border-radius: 4px;
                }
                .markdown-content::-webkit-scrollbar-thumb:hover {
                    background: #555;
                }
                .markdown-content h1, .markdown-content h2, .markdown-content h3 {
                    margin-top: 1em;
                    margin-bottom: 0.5em;
                    color: #fff;
                }
                .markdown-content code {
                    background: #2a2a2a;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: monospace;
                    color: #e0e0e0;
                }
                .markdown-content pre {
                    background: #2a2a2a;
                    padding: 1em;
                    border-radius: 5px;
                    overflow-x: auto;
                    color: #e0e0e0;
                }
                .markdown-content blockquote {
                    border-left: 4px solid #444;
                    margin: 0;
                    padding-left: 1em;
                    color: #aaa;
                }
                .markdown-content table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1em 0;
                }
                .markdown-content th, .markdown-content td {
                    border: 1px solid #444;
                    padding: 8px;
                }
                .markdown-content th {
                    background: #2a2a2a;
                }
                .markdown-content a {
                    color: #4a9eff;
                }
                .markdown-content a:hover {
                    color: #6fb1ff;
                }
            `;
            document.head.appendChild(style);

            function populate(html) {
                if (this.widgets) {
                    // Remove existing widgets
                    for (let i = 0; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = 0;
                }

                const v = [...html];
                if (!v[0]) {
                    v.shift();
                }

                for (let list of v) {
                    // Force list to be an array
                    if (!(list instanceof Array)) list = [list];
                    for (const l of list) {
                        // Create a container for the markdown content
                        const container = document.createElement('div');
                        container.className = 'markdown-content';
                        container.innerHTML = l;

                        // Add the widget using addDOMWidget
                        this.addDOMWidget(
                            "text_" + this.widgets?.length ?? 0,
                            "STRING",
                            container,
                            {
                                getValue: () => l,
                                setValue: (value) => {
                                    container.innerHTML = value;
                                }
                            }
                        );
                    }
                }
            }

            // When the node is executed we will be sent the input text, display this in the widget
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message.html) {
                    populate.call(this, message.html);
                }
            };

            const VALUES = Symbol();
            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function () {
                // Store unmodified widget values as they get removed on configure by new frontend
                this[VALUES] = arguments[0]?.widgets_values;
                return configure?.apply(this, arguments);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                const widgets_values = this[VALUES];
                if (widgets_values?.length) {
                    // In newer frontend there seems to be a delay in creating the initial widget
                    requestAnimationFrame(() => {
                        populate.call(this, widgets_values);
                    });
                }
            };
        }
    },
}); 