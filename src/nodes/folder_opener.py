import os
import subprocess
import platform
from server import PromptServer
import importlib
from aiohttp import web
import sys
import nodes  # ComfyUI's nodes module
import time
import folder_paths  # Add this import
import ctypes
from ctypes import wintypes

if platform.system() == "Windows":
    try:
        import win32gui
        import win32con
        import win32process
        import win32com.client
        
        # Windows API constants
        SW_RESTORE = 9
        SW_SHOW = 5
        
        def force_window_focus(window_title=None, process_name=None):
            def _window_enum_callback(hwnd, results):
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                
                window_text = win32gui.GetWindowText(hwnd)
                if window_title and window_title.lower() in window_text.lower():
                    results.append(hwnd)
                elif process_name:
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        process = win32gui.GetWindowText(hwnd)
                        if process_name.lower() in process.lower():
                            results.append(hwnd)
                    except:
                        pass
                return True
            
            results = []
            win32gui.EnumWindows(_window_enum_callback, results)
            
            if results:
                hwnd = results[0]
                # Force window to front
                win32gui.ShowWindow(hwnd, SW_RESTORE)
                win32gui.ShowWindow(hwnd, SW_SHOW)
                win32gui.SetForegroundWindow(hwnd)
                
                # Use SetWindowPos as backup
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                return True
            return False

    except ImportError:
        print("pywin32 not installed, Windows focus features will be limited")
        force_window_focus = None

class DreambaitFolderOpener:
    """A utility node that provides folder opening functionality"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {}}
    
    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "utils"

    def noop(self):
        """Empty function since this node doesn't process anything"""
        return {}

    @classmethod
    def get_node_path(cls, class_type):
        """Get the package path for a given node class type"""
        try:
            # Get the custom_nodes directory path
            custom_nodes_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            comfy_root = os.path.dirname(custom_nodes_path)
            
            # Try to find the node in ComfyUI's node database
            if hasattr(nodes, "NODE_CLASS_MAPPINGS"):
                node_class = nodes.NODE_CLASS_MAPPINGS.get(class_type)
                if node_class:
                    # Get the module name from the class
                    module_name = node_class.__module__.split('.')[0]
                    
                    # If it's a built-in node (not in custom_nodes)
                    if module_name in ['nodes', 'comfy', 'comfy_extras']:
                        return comfy_root
                    
                    # Look for a directory matching the module name for custom nodes
                    for dir_name in os.listdir(custom_nodes_path):
                        if dir_name.lower().replace('-', '_').replace('.', '_') == module_name.lower().replace('-', '_').replace('.', '_'):
                            dir_path = os.path.join(custom_nodes_path, dir_name)
                            if os.path.isdir(dir_path):
                                return dir_path
            
            return None
        except Exception as e:
            print(f"[Folder Opener] Error: {e}")
            return None

    @classmethod
    def open_folder(cls, folder_path, select_file=None):
        """
        Opens a folder and optionally selects a specific file
        folder_path: The directory to open
        select_file: Optional file to highlight within the directory
        """
        try:
            folder_path = os.path.normpath(folder_path)
            # Get relative path from ComfyUI root
            comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(folder_path))))
            rel_path = os.path.relpath(folder_path, comfy_root)

            if platform.system() == "Windows":
                target_path = os.path.join(folder_path, select_file) if select_file else folder_path
                
                # Start Explorer
                subprocess.Popen(['explorer', '/select,', target_path])
                time.sleep(0.5)  # Give Explorer time to open
                
                if force_window_focus:
                    # Try multiple ways to find and focus the window
                    titles_to_try = [
                        os.path.basename(target_path),
                        os.path.basename(folder_path),
                        "File Explorer"
                    ]
                    
                    focused = False
                    for title in titles_to_try:
                        if force_window_focus(window_title=title):
                            focused = True
                            break
                    
                    # If title search failed, try process name
                    if not focused:
                        force_window_focus(process_name="explorer.exe")

            elif platform.system() == "Darwin":
                try:
                    target_path = os.path.join(folder_path, select_file) if select_file else folder_path
                    # More aggressive macOS focus script
                    script = f'''
                    tell application "Finder"
                        activate
                        reveal POSIX file "{target_path}"
                    end tell
                    
                    delay 0.5
                    
                    tell application "System Events"
                        set frontmost of process "Finder" to true
                    end tell
                    
                    tell application "Finder"
                        activate
                    end tell
                    '''
                    subprocess.run(['osascript', '-e', script])
                except:
                    subprocess.Popen(["open", folder_path])
            
            else:  # Linux
                file_managers = [
                    ["nautilus", folder_path],
                    ["dolphin", folder_path],
                    ["nemo", folder_path],
                    ["thunar", folder_path],
                    ["xdg-open", folder_path]
                ]
                
                success = False
                for fm_command in file_managers:
                    try:
                        if select_file:
                            if fm_command[0] == "nautilus":
                                fm_command[1] = os.path.join(folder_path, select_file)
                            elif fm_command[0] == "dolphin":
                                fm_command.extend(["--select", os.path.join(folder_path, select_file)])
                            elif fm_command[0] == "nemo":
                                fm_command[1] = os.path.join(folder_path, select_file)
                        
                        if fm_command[0] != "xdg-open":
                            fm_command.insert(1, "--new-window")
                        
                        process = subprocess.Popen(fm_command)
                        success = True
                        
                        # Try multiple ways to focus the window
                        try:
                            # Try wmctrl with different window titles
                            titles = [
                                os.path.basename(target_path) if select_file else os.path.basename(folder_path),
                                folder_path,
                                fm_command[0]  # The file manager name
                            ]
                            for title in titles:
                                try:
                                    subprocess.run(["wmctrl", "-a", title], check=False)
                                except:
                                    continue
                        except:
                            pass
                        break
                    except FileNotFoundError:
                        continue
                
                if not success:
                    print(f"[Folder Opener] Failed: No suitable file manager found for {rel_path}")
                    return False
            
            print(f"[Folder Opener] Successfully opened: {rel_path}{' -> ' + select_file if select_file else ''}")
            return True
        except Exception as e:
            print(f"[Folder Opener] Error: {e}")
            return False

# API endpoint registration
@PromptServer.instance.routes.post("/dreambait/open_node_folder")
async def open_node_folder(request):
    try:
        data = await request.json()
        class_type = data.get("class_type")
        
        if not class_type:
            return web.json_response({"success": False, "error": "No class type provided"})

        package_path = DreambaitFolderOpener.get_node_path(class_type)
        
        if package_path and os.path.exists(package_path):
            # Pass None as select_file to just open the folder
            success = DreambaitFolderOpener.open_folder(package_path, select_file=None)
            if success:
                return web.json_response({"success": True, "path": package_path})
            else:
                return web.json_response({"success": False, "error": "Failed to open folder"})
        return web.json_response({"success": False, "error": f"Package path not found for {class_type}"})
    except Exception as e:
        print(f"[Folder Opener] Error: {e}")
        return web.json_response({"success": False, "error": str(e)})

@PromptServer.instance.routes.post("/dreambait/open_image_folder")
async def open_image_folder(request):
    try:
        data = await request.json()
        filename = data.get("filename")
        is_preview = data.get("is_preview", False)
        
        if not filename:
            return web.json_response({"success": False, "error": "No filename provided"})

        # Handle different cases
        if filename == "output":
            # For SaveImage nodes, open the output directory
            folder_path = folder_paths.get_output_directory()
            # Try to find the most recent file in output
            files = sorted(os.listdir(folder_path), key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
            select_file = files[0] if files else None
        elif is_preview:
            # For preview images, they're in the temp directory
            folder_path = folder_paths.get_temp_directory()
            select_file = os.path.basename(filename)
        elif os.path.isdir(filename):
            # For directory inputs
            folder_path = filename
            select_file = None
        else:
            # For single file inputs
            if os.path.isabs(filename):
                folder_path = os.path.dirname(filename)
                select_file = os.path.basename(filename)
            else:
                # Check input directory first
                input_path = os.path.join(folder_paths.get_input_directory(), filename)
                if os.path.exists(input_path):
                    folder_path = os.path.dirname(input_path)
                    select_file = os.path.basename(input_path)
                else:
                    # Check temp directory
                    temp_path = os.path.join(folder_paths.get_temp_directory(), filename)
                    if os.path.exists(temp_path):
                        folder_path = os.path.dirname(temp_path)
                        select_file = os.path.basename(temp_path)
                    else:
                        # Check output directory
                        output_path = os.path.join(folder_paths.get_output_directory(), filename)
                        folder_path = os.path.dirname(output_path)
                        select_file = os.path.basename(output_path)
        
        if os.path.exists(folder_path):
            success = DreambaitFolderOpener.open_folder(folder_path, select_file)
            if success:
                return web.json_response({"success": True, "path": folder_path})
            else:
                return web.json_response({"success": False, "error": "Failed to open folder"})
        return web.json_response({"success": False, "error": f"Folder not found: {folder_path}"})
    except Exception as e:
        print(f"[Folder Opener] Error: {e}")
        return web.json_response({"success": False, "error": str(e)}) 