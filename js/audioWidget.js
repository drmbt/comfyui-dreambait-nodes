import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Dreambait.AudioWidget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.input?.required?.audio?.audio_upload) {
            // Store the original onAdded
            const onAddedOriginal = nodeType.prototype.onAdded;

            nodeType.prototype.onAdded = function() {
                if (onAddedOriginal) {
                    onAddedOriginal.apply(this);
                }

                const widget = {
                    type: "audio",
                    name: "audio_upload",
                    value: "",
                    options: {},
                    draw: function(ctx, node, width, posY, height) {
                        // Your custom drawing code here if needed
                    },
                    computeSize: function() {
                        return [200, 40];
                    },
                    async serializeValue() {
                        return this.value;
                    }
                };

                widget.element = document.createElement("div");
                widget.element.style.width = "100%";
                widget.element.style.height = "40px";
                widget.element.style.overflow = "hidden";

                const audioElement = document.createElement("audio");
                audioElement.style.width = "100%";
                audioElement.controls = true;
                widget.element.appendChild(audioElement);

                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = "audio/*,video/*";
                fileInput.style.display = "none";

                fileInput.addEventListener("change", () => {
                    const file = fileInput.files[0];
                    if (file) {
                        const formData = new FormData();
                        formData.append("image", file);
                        formData.append("type", "input");
                        api.fetchApi("/upload/audio", {
                            method: "POST",
                            body: formData
                        }).then(response => {
                            if (response.status === 200) {
                                response.json().then(data => {
                                    this.value = data.name;
                                    audioElement.src = api.apiURL(`/view?filename=${encodeURIComponent(data.name)}&type=input`);
                                });
                            }
                        });
                    }
                });

                widget.element.addEventListener("click", () => {
                    fileInput.click();
                });

                widget.element.appendChild(fileInput);
                this.addCustomWidget(widget);
            };
        }
    }
}); 