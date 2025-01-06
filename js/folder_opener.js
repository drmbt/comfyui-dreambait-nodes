import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function getNodeFileInfo(node) {
    let filePath = null;
    let isPreview = false;

    // Check for preview images (both PreviewImage node and any node with preview)
    if (node.imgs && node.imgs[0] && node.imgs[0].src) {
        const src = node.imgs[0].src;
        if (src.startsWith('/view?filename=')) {
            filePath = decodeURIComponent(src.split('filename=')[1].split('&')[0]);
            isPreview = true;
        }
    }
    
    // Check node properties for file information
    if (!filePath && node.properties) {
        if (node.properties.filename) filePath = node.properties.filename;
        else if (node.properties.path) filePath = node.properties.path;
        else if (node.properties.file_path) filePath = node.properties.file_path;
    }

    // Check widgets for file information
    if (!filePath && node.widgets) {
        const fileWidgets = node.widgets.filter(w => 
            (w.name === "filename_prefix" && node.type === "SaveImage") ||
            (w.name === "image" && node.type === "LoadImage") ||
            (w.name === "video" && node.type.includes("LoadVideo")) ||
            (w.name === "directory" && node.type.includes("LoadImagesFromDirectory")) ||
            w.name === "filename" || 
            w.name === "path" ||
            w.name === "file_path"
        );
        
        if (fileWidgets.length > 0) {
            const widget = fileWidgets[0];
            filePath = node.type === "SaveImage" ? "output" : widget.value;
        }
    }

    // Check node outputs for file information
    if (!filePath && node.outputs) {
        for (const output of node.outputs) {
            if (output.links && output.links.length > 0) {
                const link = app.graph.links[output.links[0]];
                if (link && link.data && link.data.path) {
                    filePath = link.data.path;
                    break;
                }
            }
        }
    }

    return { filePath, isPreview };
}

function addOpenFolderOption() {
    const originalGetNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
    
    LGraphCanvas.prototype.getNodeMenuOptions = function(node) {
        const options = originalGetNodeMenuOptions.call(this, node);
        const newOptions = [];
        
        // Add option for image/video related nodes
        const isMediaNode = node.type === "LoadImage" || 
                          node.type === "SaveImage" || 
                          node.type === "PreviewImage" ||
                          node.type.includes("VHS_") ||  // All VHS nodes
                          node.type.includes("Load") ||   // Any load node
                          node.type.includes("Save") ||   // Any save node
                          node.imgs ||                    // Any node with image preview
                          (node.widgets && node.widgets.some(w => 
                              w.type === "image" || 
                              w.type === "video" || 
                              w.name === "filename" || 
                              w.name === "path"
                          ));

        if (isMediaNode) {
            const { filePath, isPreview } = getNodeFileInfo(node);

            if (filePath) {
                newOptions.push({
                    content: isPreview ? "🖼️ Open Preview Location" : "🖼️ Open File Location",
                    callback: async () => {
                        try {
                            const response = await api.fetchApi("/dreambait/open_image_folder", {
                                method: "POST",
                                body: JSON.stringify({
                                    filename: filePath,
                                    is_preview: isPreview
                                })
                            });
                            
                            const data = await response.json();
                            if (!data.success) {
                                console.error("Failed to open folder:", data.error);
                                app.ui.dialog.show("Error", `Failed to open folder: ${data.error}`);
                            }
                        } catch (error) {
                            console.error("Error opening folder:", error);
                            app.ui.dialog.show("Error", `Error opening folder: ${error.message || error}`);
                        }
                    }
                });
            }
        }

        // Add option for custom nodes
        if (node.type && !node.type.startsWith("Reroute") && !node.type.startsWith("Primitive")) {
            newOptions.push({
                content: "📁 Open Node Package Folder",
                callback: async () => {
                    try {
                        const response = await api.fetchApi("/dreambait/open_node_folder", {
                            method: "POST",
                            body: JSON.stringify({
                                class_type: node.type
                            })
                        });
                        
                        const data = await response.json();
                        if (!data.success) {
                            console.error("Failed to open folder:", data.error);
                            app.ui.dialog.show("Error", `Failed to open folder: ${data.error}`);
                        }
                    } catch (error) {
                        console.error("Error opening folder:", error);
                        app.ui.dialog.show("Error", `Error opening folder: ${error.message || error}`);
                    }
                }
            });
            newOptions.push(null);
        }
        
        return [...newOptions, ...options];
    };
}

app.registerExtension({
    name: "Dreambait.FolderOpener",
    async setup() {
        addOpenFolderOption();
    }
}); 