import os
import json
import shutil
from PIL import Image
import torch
import numpy as np
from server import PromptServer
from aiohttp import web
import io


class ShotHistory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "./shots"}),
            },
            "optional": {
                "selected_files": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "paths")
    OUTPUT_IS_LIST = (False, True)  # Images are batched, paths are a list
    FUNCTION = "load_shots"
    OUTPUT_NODE = True
    CATEGORY = "🎬 Storyboard"

    def load_shots(self, path, selected_files=""):
        if not selected_files or selected_files.strip() == "":
            # Return empty batch if no files selected
            empty_image = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
            return (empty_image, "")
        
        try:
            # Parse selected files JSON
            file_list = json.loads(selected_files) if selected_files else []
            
            if not file_list:
                empty_image = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
                return (empty_image, "")
            
            images = []
            paths = []
            target_size = None  # Will be set to first image's size
            
            print(f"Processing {len(file_list)} selected files...")
            
            for i, filename in enumerate(file_list):
                full_path = os.path.join(path, filename)
                if os.path.isfile(full_path):
                    try:
                        image = Image.open(full_path)
                        # Convert to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Set target size from first image
                        if target_size is None:
                            target_size = image.size  # (width, height)
                            print(f"Target size set to: {target_size} from first image: {filename}")
                        else:
                            # Resize subsequent images to match first image using crop/fill
                            if image.size != target_size:
                                print(f"Crop/fill resizing {filename} from {image.size} to {target_size}")
                                image = crop_fill_resize(image, target_size)
                        
                        # Convert to tensor [H, W, C] - ComfyUI standard format
                        image_array = np.array(image).astype(np.float32) / 255.0
                        image_tensor = torch.from_numpy(image_array)
                        
                        # Ensure tensor has proper dimensions [H, W, C]
                        if len(image_tensor.shape) == 2:  # Grayscale
                            image_tensor = image_tensor.unsqueeze(-1).repeat(1, 1, 3)
                        elif len(image_tensor.shape) == 3 and image_tensor.shape[2] == 4:  # RGBA
                            image_tensor = image_tensor[:, :, :3]  # Remove alpha channel
                        elif len(image_tensor.shape) == 3 and image_tensor.shape[2] != 3:
                            # Handle other weird formats by converting to RGB
                            pil_img = Image.fromarray((image_tensor.numpy() * 255).astype(np.uint8))
                            pil_img = pil_img.convert('RGB')
                            image_array = np.array(pil_img).astype(np.float32) / 255.0
                            image_tensor = torch.from_numpy(image_array)
                        
                        images.append(image_tensor)
                        paths.append(full_path)
                        print(f"Loaded image: {filename} with final shape: {image_tensor.shape}")
                        
                    except Exception as e:
                        print(f"Error loading image {full_path}: {e}")
                        continue
            
            if not images:
                print("No valid images found")
                empty_image = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
                return (empty_image, "")
            
            # Stack images into batch [B, H, W, C] - ComfyUI format
            batch_tensor = torch.stack(images, dim=0)
            
            # For paths, return as list for batch processing
            # Each path corresponds to each image in the batch
            paths_output = paths  # Return as list, not JSON string
            
            print(f"Final batch shape: {batch_tensor.shape}")
            print(f"Returning {len(paths)} paths: {paths}")
            
            # Return tuple with proper format - note the comma for single tuple element
            return (batch_tensor, paths_output)
            
        except Exception as e:
            print(f"Error in load_shots: {e}")
            import traceback
            traceback.print_exc()
            empty_image = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
            return (empty_image, "")

    @classmethod
    def IS_CHANGED(cls, path, selected_files=""):
        return selected_files

    @classmethod
    def VALIDATE_INPUTS(cls, path, selected_files=""):
        if not os.path.isdir(path):
            return f"Directory does not exist: {path}"
        return True


def crop_fill_resize(image, target_size):
    """
    Resize image to target size using crop/fill method to preserve aspect ratio.
    This will crop the image if it's larger than target aspect ratio,
    or pad it if it's smaller, then resize to exact target size.
    """
    target_width, target_height = target_size
    target_ratio = target_width / target_height
    
    current_width, current_height = image.size
    current_ratio = current_width / current_height
    
    if abs(current_ratio - target_ratio) < 0.01:  # Very close aspect ratios
        # Just resize directly
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    if current_ratio > target_ratio:
        # Image is wider than target - crop width (center crop)
        new_width = int(current_height * target_ratio)
        left = (current_width - new_width) // 2
        image = image.crop((left, 0, left + new_width, current_height))
    else:
        # Image is taller than target - crop height (center crop)
        new_height = int(current_width / target_ratio)
        top = (current_height - new_height) // 2
        image = image.crop((0, top, current_width, top + new_height))
    
    # Now resize to exact target size
    return image.resize(target_size, Image.Resampling.LANCZOS)


def get_shot_files(path):
    """Get list of image files in the directory"""
    if not os.path.exists(path):
        return []
    
    try:
        files = []
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                files.append(f)
        return sorted(files)
    except Exception as e:
        print(f"Error reading directory {path}: {e}")
        return []


def get_file_info(path, filename):
    """Get detailed information about a file"""
    full_path = os.path.join(path, filename)
    
    if not os.path.exists(full_path):
        return None
    
    try:
        # Get file stats
        stat = os.stat(full_path)
        file_size = stat.st_size
        
        # Get timestamps
        import datetime
        created_time = datetime.datetime.fromtimestamp(stat.st_ctime)
        modified_time = datetime.datetime.fromtimestamp(stat.st_mtime)
        
        # Format file size
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        # Get image dimensions and additional metadata
        with Image.open(full_path) as img:
            width, height = img.size
            mode = img.mode
            format_name = img.format or "Unknown"
            
            # Try to get DPI info
            dpi = getattr(img, 'info', {}).get('dpi', (72, 72))
            if isinstance(dpi, (list, tuple)) and len(dpi) >= 2:
                dpi_str = f"{dpi[0]}x{dpi[1]}"
            else:
                dpi_str = "72x72"
        
        # Get file extension
        _, ext = os.path.splitext(filename)
        
        # Calculate megapixels
        megapixels = round((width * height) / 1000000, 2)
        
        return {
            "filename": filename,
            "full_path": full_path,
            "extension": ext.lower(),
            "format": format_name,
            "file_size": file_size,
            "file_size_str": size_str,
            "width": width,
            "height": height,
            "megapixels": megapixels,
            "mode": mode,
            "dpi": dpi_str,
            "aspect_ratio": round(width / height, 2) if height > 0 else 0,
            "created_date": created_time.strftime("%Y-%m-%d %H:%M:%S"),
            "modified_date": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
            "created_timestamp": stat.st_ctime,
            "modified_timestamp": stat.st_mtime
        }
    except Exception as e:
        print(f"Error getting file info for {full_path}: {e}")
        return None


@PromptServer.instance.routes.post("/shot_history/get_files")
async def api_get_shot_files(request):
    """API endpoint to get list of image files in directory"""
    data = await request.json()
    path = data.get("path", "./shots")
    
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    files = get_shot_files(path)
    return web.json_response({"files": files, "path": path})


@PromptServer.instance.routes.post("/shot_history/get_file_info")
async def api_get_file_info(request):
    """API endpoint to get detailed file information"""
    data = await request.json()
    path = data.get("path", "./shots")
    filename = data.get("filename", "")
    
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    info = get_file_info(path, filename)
    if info:
        return web.json_response(info)
    else:
        return web.json_response({"error": "File not found or error reading file"}, status=404)


@PromptServer.instance.routes.post("/shot_history/get_thumbnail")
async def api_get_shot_thumbnail(request):
    """API endpoint to get thumbnail for a specific image"""
    data = await request.json()
    path = data.get("path", "./shots")
    filename = data.get("filename", "")
    
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        return web.json_response({"error": "File does not exist"}, status=400)
    
    try:
        with Image.open(full_path) as img:
            # Create thumbnail
            img.thumbnail((100, 100), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as PNG bytes
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            
            return web.Response(body=buf.read(), content_type='image/png')
    except Exception as e:
        print(f"Error creating thumbnail for {full_path}: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/shot_history/add_file")
async def api_add_shot_file(request):
    """API endpoint to copy a file to the shots directory"""
    data = await request.json()
    source_path = data.get("source_path", "")
    target_dir = data.get("target_dir", "./shots")
    
    if not source_path or not os.path.exists(source_path):
        return web.json_response({"error": "Source file does not exist"}, status=400)
    
    if not os.path.isabs(target_dir):
        target_dir = os.path.abspath(target_dir)
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        filename = os.path.basename(source_path)
        target_path = os.path.join(target_dir, filename)
        
        # Handle filename conflicts
        counter = 1
        base, ext = os.path.splitext(filename)
        while os.path.exists(target_path):
            new_filename = f"{base}_{counter}{ext}"
            target_path = os.path.join(target_dir, new_filename)
            counter += 1
        
        # Copy the file
        shutil.copy2(source_path, target_path)
        
        return web.json_response({
            "success": True, 
            "filename": os.path.basename(target_path),
            "path": target_path
        })
        
    except Exception as e:
        print(f"Error copying file {source_path} to {target_dir}: {e}")
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/shot_history/upload_file")
async def api_upload_shot_file(request):
    """API endpoint to upload files to the shots directory"""
    try:
        reader = await request.multipart()
        target_dir = "./shots"
        uploaded_files = []
        
        async for field in reader:
            if field.name == 'target_dir':
                target_dir = await field.text()
            elif field.name == 'file':
                filename = field.filename
                if not filename:
                    continue
                
                if not os.path.isabs(target_dir):
                    target_dir = os.path.abspath(target_dir)
                
                # Create target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Handle filename conflicts
                target_path = os.path.join(target_dir, filename)
                counter = 1
                base, ext = os.path.splitext(filename)
                while os.path.exists(target_path):
                    new_filename = f"{base}_{counter}{ext}"
                    target_path = os.path.join(target_dir, new_filename)
                    counter += 1
                
                # Save the file
                with open(target_path, 'wb') as f:
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
                
                uploaded_files.append({
                    "filename": os.path.basename(target_path),
                    "path": target_path
                })
        
        return web.json_response({
            "success": True,
            "files": uploaded_files
        })
        
    except Exception as e:
        print(f"Error uploading files: {e}")
        return web.json_response({"error": str(e)}, status=500)


NODE_CLASS_MAPPINGS = {
    "ShotHistory": ShotHistory
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShotHistory": "Shot History"
} 