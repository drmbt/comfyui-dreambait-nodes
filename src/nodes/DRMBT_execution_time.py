import time

import execution
import server


class ExecutionTime:
    CATEGORY = "TyDev-Utils/Debug"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("DICT",)
    FUNCTION = "process"

    def process(self):
        global CURRENT_START_EXECUTION_DATA
        if CURRENT_START_EXECUTION_DATA:
            return CURRENT_START_EXECUTION_DATA
        return {}


CURRENT_START_EXECUTION_DATA = {}


def handle_execute(class_type, last_node_id, prompt_id, server, unique_id, node_ID, node_title):
    if not CURRENT_START_EXECUTION_DATA:
        return {}
    start_time = CURRENT_START_EXECUTION_DATA['nodes_start_perf_time'].get(unique_id)
    if start_time:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        if server.client_id is not None and last_node_id != server.last_node_id:
            server.send_sync(
                "TyDev-Utils.ExecutionTime.executed",
                {"node": unique_id, "prompt_id": prompt_id, "execution_time": int(execution_time * 1000)},
                server.client_id
            )
        print(f"#{unique_id} [{class_type}]: {execution_time:.2f}s")
        # Create and return the dictionary
        key = f"{node_ID}_{node_title}"
        value = time.time()
        return {key: value}
    return {}


try:
    origin_execute = execution.execute


    def swizzle_execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list,
                        pending_subgraph_results):
        unique_id = current_item
        class_type = dynprompt.get_node(unique_id)['class_type']
        last_node_id = server.last_node_id
        node_ID = unique_id  # Assuming unique_id is the node_ID
        node_data = dynprompt.get_node(unique_id)
        node_title = node_data.get('title', 'unknown')  # Use 'unknown' if title is not available
        result = origin_execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id,
                                execution_list,
                                pending_subgraph_results)
        execution_dict = handle_execute(class_type, last_node_id, prompt_id, server, unique_id, node_ID, node_title)
        if execution_dict:
            CURRENT_START_EXECUTION_DATA.update(execution_dict)
        print(CURRENT_START_EXECUTION_DATA)  # Print or use the dictionary as needed
        return result


    execution.execute = swizzle_execute
except Exception as e:
    pass

# region: Deprecated
try:
    # The execute method in the old version of ComfyUI is now deprecated.
    origin_recursive_execute = execution.recursive_execute


    def swizzle_origin_recursive_execute(server, prompt, outputs, current_item, extra_data, executed, prompt_id,
                                         outputs_ui,
                                         object_storage):
        unique_id = current_item
        class_type = prompt[unique_id]['class_type']
        last_node_id = server.last_node_id
        node_ID = unique_id  # Assuming unique_id is the node_ID
        node_data = prompt[unique_id]
        node_title = node_data.get('title', 'unknown')  # Use 'unknown' if title is not available
        result = origin_recursive_execute(server, prompt, outputs, current_item, extra_data, executed, prompt_id,
                                          outputs_ui,
                                          object_storage)
        execution_dict = handle_execute(class_type, last_node_id, prompt_id, server, unique_id, node_ID, node_title)
        if execution_dict:
            CURRENT_START_EXECUTION_DATA.update(execution_dict)
        print(CURRENT_START_EXECUTION_DATA)  # Print or use the dictionary as needed
        return result


    execution.recursive_execute = swizzle_origin_recursive_execute
except Exception as e:
    pass
# endregion

origin_func = server.PromptServer.send_sync


def swizzle_send_sync(self, event, data, sid=None):
    # print(f"swizzle_send_sync, event: {event}, data: {data}")
    global CURRENT_START_EXECUTION_DATA
    if event == "execution_start":
        CURRENT_START_EXECUTION_DATA = dict(
            start_perf_time=time.perf_counter(),
            nodes_start_perf_time={}
        )

    origin_func(self, event=event, data=data, sid=sid)

    if event == "executing" and data and CURRENT_START_EXECUTION_DATA:
        if data.get("node") is None:
            if sid is not None:
                start_perf_time = CURRENT_START_EXECUTION_DATA.get('start_perf_time')
                new_data = data.copy()
                if start_perf_time is not None:
                    execution_time = time.perf_counter() - start_perf_time
                    new_data['execution_time'] = int(execution_time * 1000)
                origin_func(
                    self,
                    event="TyDev-Utils.ExecutionTime.execution_end",
                    data=new_data,
                    sid=sid
                )
        else:
            node_id = data.get("node")
            CURRENT_START_EXECUTION_DATA['nodes_start_perf_time'][node_id] = time.perf_counter()


server.PromptServer.send_sync = swizzle_send_sync