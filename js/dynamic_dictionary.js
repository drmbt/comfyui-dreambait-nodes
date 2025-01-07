import { app } from "../../../scripts/app.js"

const TypeSlot = {
    Input: 1,
    Output: 2,
};

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

const _ID = "DynamicDictionary";
const _PREFIX = "input";
const _TYPE = "*";

app.registerExtension({
    name: 'dreambait.DynamicDictionary',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _ID) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const result = onNodeCreated?.apply(this);
            this.addInput(_PREFIX, _TYPE);
            return result;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const result = onConnectionsChange?.apply(this, arguments);

            if (slotType === TypeSlot.Input && node_slot.name !== 'keys') {
                if (link_info && event === TypeSlotEvent.Connect) {
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    );

                    if (fromNode) {
                        const parent_link = fromNode.outputs[link_info.origin_slot];
                        if (parent_link) {
                            node_slot.type = parent_link.type;
                            const outputName = parent_link.name.toLowerCase().replace(/\s+/g, '_');
                            node_slot.name = outputName;
                        }
                    }
                } else if (event === TypeSlotEvent.Disconnect) {
                    this.removeInput(slot_idx);
                }

                // Track and rename duplicate connections
                let nameCount = {};
                for (const slot of this.inputs) {
                    if (slot.name === 'keys' || !slot.link) continue;
                    
                    const baseName = slot.name;
                    nameCount[baseName] = (nameCount[baseName] || 0) + 1;
                    
                    if (nameCount[baseName] > 1) {
                        slot.name = `${baseName}_${nameCount[baseName]}`;
                    }
                }

                // Ensure there's always an empty slot for new connections
                let last = this.inputs[this.inputs.length - 1];
                if (last === undefined || (last.name === 'keys') || 
                    (last.name != _PREFIX || last.type != _TYPE)) {
                    this.addInput(_PREFIX, _TYPE);
                }

                this.graph?.setDirtyCanvas(true);
            }
            return result;
        }
    },
}); 