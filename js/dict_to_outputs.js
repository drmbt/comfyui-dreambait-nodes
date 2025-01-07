import { app } from "../../../scripts/app.js"

app.registerExtension({
    name: 'dreambait.DictToOutputs',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === 'DictToOutputs') {
            // Store the original onNodeCreated
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this);
                
                // Add reload button widget
                this.addWidget("button", "reload", "Reload Outputs", () => {
                    const outputCount = this.widgets.find(w => w.name === "output_count")?.value || 4;
                    
                    // Update inputs (keep existing values)
                    const existingValues = {};
                    this.widgets.forEach(w => {
                        if (w.name.startsWith('key_')) {
                            existingValues[w.name] = w.value;
                        }
                    });
                    
                    // Remove old key inputs
                    this.widgets = this.widgets.filter(w => !w.name.startsWith('key_'));
                    
                    // Add new key inputs
                    for (let i = 1; i <= outputCount; i++) {
                        const name = `key_${i}`;
                        this.addWidget("text", name, existingValues[name] || "", () => {}, {
                            multiline: false,
                            placeholder: `Key ${i}`,
                        });
                    }
                    
                    // Move output_count and reload to end
                    const outputCountWidget = this.widgets.find(w => w.name === "output_count");
                    const reloadWidget = this.widgets.find(w => w.name === "reload");
                    this.widgets = this.widgets.filter(w => w !== outputCountWidget && w !== reloadWidget);
                    this.widgets.push(outputCountWidget, reloadWidget);
                    
                    // Create base outputs
                    const outputs = [
                        { name: "dictionary", type: "DICT", links: null, slot_index: 0 }
                    ];
                    
                    // Add numbered outputs
                    for (let i = 0; i < outputCount; i++) {
                        outputs.push({
                            name: `value_${i + 1}`,
                            type: "*",
                            links: null,
                            slot_index: i + 1
                        });
                    }
                    
                    this.outputs = outputs;
                    
                    // Update size with both inputs and outputs
                    const minHeight = 100;
                    const heightPerItem = 22;  // Slightly increased for better spacing
                    const baseWidgetHeight = 63;  // Height for dictionary input + output_count + reload (one height unit smaller)
                    const inputsHeight = outputCount * heightPerItem;  // Height for key inputs
                    const outputsHeight = outputs.length * heightPerItem;  // Height for outputs
                    const width = 200;
                    const height = Math.max(minHeight, baseWidgetHeight + inputsHeight + outputsHeight);
                    
                    this.size = [width, height];
                    this.setDirtyCanvas(true);
                });
                
                return result;
            };

            // Add compute size method
            nodeType.prototype.computeSize = function() {
                const minHeight = 100;
                const heightPerItem = 22;  // Slightly increased for better spacing
                const baseWidgetHeight = 63;  // Height for dictionary input + output_count + reload (one height unit smaller)
                const outputCount = this.widgets?.find(w => w.name === "output_count")?.value || 4;
                const inputsHeight = outputCount * heightPerItem;  // Height for key inputs
                const outputsHeight = (this.outputs?.length || 0) * heightPerItem;  // Height for outputs
                const width = 200;
                const height = Math.max(minHeight, baseWidgetHeight + inputsHeight + outputsHeight);
                return [width, height];
            };
        }
        else if (nodeData.name === 'StringToDict') {
            // Store the original onNodeCreated
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this);
                
                // Force the input string widget to be small
                const inputWidget = this.widgets.find(w => w.name === "input_string");
                if (inputWidget) {
                    inputWidget.options.multiline = false;
                }
                
                return result;
            };

            // Add compute size method for StringToDict
            nodeType.prototype.computeSize = function() {
                return [180, 50];  // Minimum size for input + output
            };
        }
    }
}); 