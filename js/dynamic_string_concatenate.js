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
            // Add initial empty string input (lowercase), only if none exist
            if (!this.inputs.some(slot => slot.name === _EMPTY_PREFIX)) {
                this.addInput(_EMPTY_PREFIX, _TYPE);
            }
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
                
                // Always rename string inputs after any connect/disconnect
                this.renameStringInputsSequentially();

                // Ensure only one empty slot exists
                this.ensureSingleEmptyStringSlot();

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
            // First pass: rename connected or filled slots
            for (let slot of stringSlots) {
                // If slot is connected or has a value, give it a sequential name
                if (slot.link || (slot.widget && slot.widget.value && slot.widget.value.trim())) {
                    connectedCount++;
                    slot.name = _PREFIX + connectedCount; // "STRING1", "STRING2", ...
                    if (slot.widget) slot.widget.name = _PREFIX + connectedCount;
                }
            }
            // Second pass: all others become the empty slot (will be deduped later)
            for (let slot of stringSlots) {
                if (!slot.link && (!slot.widget || !slot.widget.value || !slot.widget.value.trim())) {
                    slot.name = _EMPTY_PREFIX;
                    if (slot.widget) slot.widget.name = _EMPTY_PREFIX;
                }
            }
        }

        // Ensure only one empty 'string' slot exists
        nodeType.prototype.ensureSingleEmptyStringSlot = function() {
            let emptySlots = this.inputs.filter(slot => 
                slot.name === _EMPTY_PREFIX && 
                (!slot.link && (!slot.widget || !slot.widget.value || !slot.widget.value.trim()))
            );
            // If more than one empty slot, remove extras
            for (let i = 1; i < emptySlots.length; i++) {
                let idx = this.inputs.indexOf(emptySlots[i]);
                if (idx !== -1) this.removeInput(idx);
            }
            // If no empty slot, add one
            if (emptySlots.length === 0) {
                this.addInput(_EMPTY_PREFIX, _TYPE);
            }
        }

        // Add input change handler to detect when users type into empty slots
        const onPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function(name, value, prev_value) {
            const result = onPropertyChanged?.apply(this, arguments);
            // Always rename and ensure only one empty slot
            this.renameStringInputsSequentially();
            this.ensureSingleEmptyStringSlot();
            this.graph?.setDirtyCanvas(true);
            return result;
        }
    },
}); 