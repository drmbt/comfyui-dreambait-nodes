import { app } from "../../../scripts/app.js"

const TypeSlot = {
    Input: 1,
    Output: 2,
};

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

const _ID = "DynamicStringConcatenate";
const _PREFIX = "STRING";
const _EMPTY_PREFIX = "string";
const _TYPE = "STRING";

app.registerExtension({
    name: 'dreambait.DynamicStringConcatenate',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _ID) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const result = onNodeCreated?.apply(this);
            // Add initial empty string input (lowercase)
            this.addInput(_EMPTY_PREFIX, _TYPE);
            return result;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const result = onConnectionsChange?.apply(this, arguments);

            // Only handle input connections, and skip the control inputs (delimiter, skip_empty, trim_whitespace)
            if (slotType === TypeSlot.Input && 
                node_slot.name !== 'delimiter' && 
                node_slot.name !== 'skip_empty' && 
                node_slot.name !== 'trim_whitespace') {
                
                if (link_info && event === TypeSlotEvent.Connect) {
                    // When connecting, ensure all string inputs have proper sequential names
                    this.renameStringInputsSequentially();
                    
                    // Add a new empty slot for the next connection
                    this.addInput(_EMPTY_PREFIX, _TYPE);
                    
                } else if (event === TypeSlotEvent.Disconnect) {
                    // When disconnecting, remove the slot and rename remaining ones
                    this.removeInput(slot_idx);
                    this.renameStringInputsSequentially();
                    
                    // Ensure we always have at least one empty slot
                    let hasEmptyStringSlot = this.inputs.some(slot => 
                        slot.name !== 'delimiter' && 
                        slot.name !== 'skip_empty' && 
                        slot.name !== 'trim_whitespace' && 
                        !slot.link
                    );
                    
                    if (!hasEmptyStringSlot) {
                        this.addInput(_EMPTY_PREFIX, _TYPE);
                    }
                }

                this.graph?.setDirtyCanvas(true);
            }
            return result;
        }

        // Add helper method to rename string inputs sequentially
        nodeType.prototype.renameStringInputsSequentially = function() {
            let stringSlots = this.inputs.filter(slot => 
                slot.name !== 'delimiter' && 
                slot.name !== 'skip_empty' && 
                slot.name !== 'trim_whitespace'
            );
            
            let connectedCount = 0;
            
            // First pass: rename connected slots
            for (let slot of stringSlots) {
                if (slot.link) {
                    connectedCount++;
                    slot.name = _PREFIX + connectedCount; // "STRING1", "STRING2", "STRING3", etc.
                }
            }
            
            // Second pass: ensure unconnected slots are named properly
            for (let slot of stringSlots) {
                if (!slot.link) {
                    slot.name = _EMPTY_PREFIX; // Empty slots are always just "string"
                }
            }
        }

        // Add input change handler to detect when users type into empty slots
        const onPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function(name, value, prev_value) {
            const result = onPropertyChanged?.apply(this, arguments);
            
            // If someone types into an empty string slot, create a new empty slot
            if (name && name.startsWith(_EMPTY_PREFIX) && value && value.trim() && value !== prev_value) {
                // Count how many empty slots we have
                let emptySlots = this.inputs.filter(slot => 
                    slot.name === _EMPTY_PREFIX && 
                    (!slot.widget || !slot.widget.value || !slot.widget.value.trim())
                );
                
                // If we don't have any empty slots, add one
                if (emptySlots.length === 0) {
                    this.addInput(_EMPTY_PREFIX, _TYPE);
                    this.graph?.setDirtyCanvas(true);
                }
            }
            
            return result;
        }
    },
}); 