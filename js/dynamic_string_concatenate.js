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
            // Add initial string input
            this.addInput(_PREFIX, _TYPE);
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
                    this.addInput(_PREFIX, _TYPE);
                    
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
                        this.addInput(_PREFIX, _TYPE);
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
                    if (connectedCount === 1) {
                        slot.name = _PREFIX; // First one is just "STRING"
                    } else {
                        slot.name = _PREFIX + connectedCount; // "STRING2", "STRING3", etc.
                    }
                }
            }
            
            // Second pass: ensure unconnected slots are named properly
            for (let slot of stringSlots) {
                if (!slot.link) {
                    slot.name = _PREFIX; // Empty slots are always just "STRING"
                }
            }
        }
    },
}); 