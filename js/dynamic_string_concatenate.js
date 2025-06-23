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

        // Gets all dynamic string inputs
        const getDynamicStringInputs = (node) => {
            return node.inputs.filter(slot =>
                slot.name !== 'delimiter' &&
                slot.name !== 'skip_empty' &&
                slot.name !== 'trim_whitespace'
            );
        };

        // The core function to manage inputs
        nodeType.prototype.updateDynamicInputs = function() {
            let stringInputs = getDynamicStringInputs(this);
            let connectedCount = 0;

            // First pass: rename everything based on its state
            for (const slot of stringInputs) {
                const isConnected = !!slot.link;
                const hasText = slot.widget && slot.widget.value && String(slot.widget.value).trim();
                if (isConnected || hasText) {
                    connectedCount++;
                    const newName = _PREFIX + connectedCount;
                    slot.name = newName;
                    if (slot.widget) slot.widget.name = newName;
                } else {
                    slot.name = _EMPTY_PREFIX;
                    if (slot.widget) slot.widget.name = _EMPTY_PREFIX;
                }
            }

            // Second pass: remove duplicate empty slots
            const emptySlots = this.inputs.filter(slot => slot.name === _EMPTY_PREFIX);
            for (let i = emptySlots.length - 1; i > 0; i--) {
                const slotToRemove = emptySlots[i];
                const index = this.inputs.indexOf(slotToRemove);
                if (index > -1) {
                    this.removeInput(index);
                }
            }
            
            // Final pass: ensure at least one empty slot exists if all are used
            stringInputs = getDynamicStringInputs(this);
            const hasEmptySlot = stringInputs.some(slot => slot.name === _EMPTY_PREFIX);
            if (!hasEmptySlot) {
                this.addInput(_EMPTY_PREFIX, _TYPE);
            }

            this.graph?.setDirtyCanvas(true);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            setTimeout(() => {
                if (!this.inputs.some(slot => slot.name === _EMPTY_PREFIX)) {
                    this.addInput(_EMPTY_PREFIX, _TYPE);
                }
                this.updateDynamicInputs();
            }, 10);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const result = onConnectionsChange?.apply(this, arguments);
            if (slotType === TypeSlot.Input) {
                this.updateDynamicInputs();
            }
            return result;
        };

        const onPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function(name, value, prev_value) {
            const result = onPropertyChanged?.apply(this, arguments);
            if (name !== 'delimiter' && name !== 'skip_empty' && name !== 'trim_whitespace') {
                this.updateDynamicInputs();
            }
            return result;
        };
    },
}); 