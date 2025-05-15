import torch
import torch.nn.functional as F
import numpy as np
import comfy.utils
import cv2
import os
from nodes import MAX_RESOLUTION

class ImageResizeFaceAware:
    def __init__(self):
        # Initialize the DNN face detector
        self.face_detector = None
        self.face_net = None
        
    def load_face_detector(self):
        if self.face_detector is None:
            # Get the directory where the model files should be
            model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "face_detector")
            os.makedirs(model_dir, exist_ok=True)
            
            # Paths for the model files
            prototxt_path = os.path.join(model_dir, "deploy.prototxt")
            caffemodel_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            
            # Download model files if they don't exist
            if not os.path.exists(prototxt_path):
                prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                self.download_file(prototxt_url, prototxt_path)
            
            if not os.path.exists(caffemodel_path):
                caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                self.download_file(caffemodel_url, caffemodel_path)
            
            # Load the model
            self.face_net = cv2.dnn.readNet(prototxt_path, caffemodel_path)

    def download_file(self, url, save_path):
        import urllib.request
        urllib.request.urlretrieve(url, save_path)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to be resized"}),
                "width": ("INT", { 
                    "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                    "tooltip": "Target width. Set to 0 to automatically calculate based on height while preserving aspect ratio"
                }),
                "height": ("INT", { 
                    "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                    "tooltip": "Target height. Set to 0 to automatically calculate based on width while preserving aspect ratio"
                }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"], {
                    "default": "lanczos",
                    "tooltip": "Method used for interpolation. Lanczos generally gives best quality, nearest is fastest"
                }),
                "method": (["stretch", "keep proportion", "fill / crop", "pad", "crop to face", "crop to face (keep_size)", "crop to subject", "crop to subject (largest square)", "crop to subject (square dimensions)"], {
                    "default": "keep proportion",
                    "tooltip": "Resize method: stretch (distort), keep proportion (scale), fill/crop (fill area and crop excess), pad (add borders), crop to face/subject (detect and crop+resize), crop to face/subject (keep_size) (square crop without resizing)"
                }),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"], {
                    "default": "always",
                    "tooltip": "When to apply the resize operation: always, only when downscaling, only when upscaling, or based on total pixel area"
                }),
                "multiple_of": ("INT", { 
                    "default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Ensure output dimensions are multiples of this value. Set to 0 to disable"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "dreambooth/image"
    DESCRIPTION = """Face-aware image resizing node, forked from ComfyUI_essentials Image Resize.

This node extends the original Image Resize functionality with face detection capabilities:
- All original resize methods (stretch, keep proportion, fill/crop, pad)
- New 'crop to face' method that detects faces, crops to them, and resizes
- New 'face-centered square crop' method that produces a square crop centered on the largest detected face without resizing

Face detection uses OpenCV's DNN face detector. If no face is detected, falls back to center crop.
The face detector model is downloaded automatically on first use.

Original Image Resize node by WASasquatch."""

    def find_face_crop(self, image, target_size):
        try:
            # Ensure face detector is loaded
            self.load_face_detector()
            
            # Convert the PyTorch tensor to a numpy array
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            
            # Get image dimensions
            h_img, w_img = image_np.shape[:2]
            
            # Prepare the image blob for the DNN
            blob = cv2.dnn.blobFromImage(image_np, 1.0, (300, 300), [104, 117, 123], False, False)
            
            # Set the input and get detection results
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            # Find the detection with highest confidence
            max_confidence = 0
            best_detection = None
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Minimum confidence threshold
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_detection = detections[0, 0, i, 3:7]
            
            if best_detection is None:
                return None  # No faces found
                
            # Convert relative coordinates to absolute
            box = best_detection * np.array([w_img, h_img, w_img, h_img])
            x1, y1, x2, y2 = box.astype(int)
            
            # Calculate face center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Calculate the square crop size (minimum of image dimensions)
            crop_size = min(w_img, h_img)
            
            # Calculate crop coordinates ensuring the crop stays within image bounds
            x1 = max(0, min(w_img - crop_size, cx - crop_size//2))
            y1 = max(0, min(h_img - crop_size, cy - crop_size//2))
            
            return [x1, y1, x1 + crop_size, y1 + crop_size]
            
        except Exception as e:
            print(f"Face detection failed: {str(e)}")
            return None  # Return None to trigger center crop fallback

    def find_subject_crop(self, image, target_width=None, target_height=None, crop_mode="default"):
        try:
            # Convert the PyTorch tensor to a numpy array
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            
            # Get image dimensions
            h_img, w_img = image_np.shape[:2]
            
            # Create a saliency detector
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            
            # Convert to grayscale if needed
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
                
            # Compute saliency map
            (success, saliency_map) = saliency.computeSaliency(gray)
            if not success:
                return None
                
            # Convert to 8-bit
            saliency_map = (saliency_map * 255).astype(np.uint8)
            
            # Find the centroid of the saliency map
            M = cv2.moments(saliency_map)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # If moments fail, use center of image
                cx = w_img // 2
                cy = h_img // 2
            
            if crop_mode == "largest_square":
                # Use shortest side of input image
                crop_size = min(w_img, h_img)
                x1 = max(0, min(w_img - crop_size, cx - crop_size//2))
                y1 = max(0, min(h_img - crop_size, cy - crop_size//2))
                return [x1, y1, x1 + crop_size, y1 + crop_size]
                
            elif crop_mode == "square_dimensions":
                # Use max of target dimensions or shortest side
                if target_width is not None and target_height is not None:
                    crop_size = max(min(w_img, h_img), max(target_width, target_height))
                else:
                    crop_size = min(w_img, h_img)
                x1 = max(0, min(w_img - crop_size, cx - crop_size//2))
                y1 = max(0, min(h_img - crop_size, cy - crop_size//2))
                return [x1, y1, x1 + crop_size, y1 + crop_size]
                
            else:  # default mode - rectangular crop
                if target_width is None or target_height is None:
                    # If no target dimensions provided, use shortest side
                    crop_size = min(w_img, h_img)
                    x1 = max(0, min(w_img - crop_size, cx - crop_size//2))
                    y1 = max(0, min(h_img - crop_size, cy - crop_size//2))
                    return [x1, y1, x1 + crop_size, y1 + crop_size]
                else:
                    # Calculate crop coordinates for rectangular crop
                    x1 = max(0, min(w_img - target_width, cx - target_width//2))
                    y1 = max(0, min(h_img - target_height, cy - target_height//2))
                    return [x1, y1, x1 + target_width, y1 + target_height]
            
        except Exception as e:
            print(f"Subject detection failed: {str(e)}")
            return None  # Return None to trigger center crop fallback

    def execute(self, image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if method.startswith("crop to subject"):
            # Determine crop mode based on method
            crop_mode = "default"
            if method == "crop to subject (largest square)":
                crop_mode = "largest_square"
            elif method == "crop to subject (square dimensions)":
                crop_mode = "square_dimensions"
            
            # Find subject and get crop coordinates
            crop_coords = self.find_subject_crop(image, width, height, crop_mode)
            
            if crop_coords is None:
                # If no subject found, center crop
                if crop_mode == "largest_square" or crop_mode == "square_dimensions":
                    crop_size = min(ow, oh)
                    x = (ow - crop_size) // 2
                    y = (oh - crop_size) // 2
                    x2 = x + crop_size
                    y2 = y + crop_size
                else:
                    # For rectangular crop, use input dimensions
                    x = (ow - width) // 2
                    y = (oh - height) // 2
                    x2 = x + width
                    y2 = y + height
            else:
                x, y, x2, y2 = crop_coords
            
            # Crop the image
            image = image[:, y:y2, x:x2, :]
            
            # For largest square and square dimensions modes, we're done
            if crop_mode in ["largest_square", "square_dimensions"]:
                width = int(x2 - x)
                height = int(y2 - y)
            else:
                # For default mode, resize to target dimensions
                if width == 0:
                    width = height
                if height == 0:
                    height = width
                
            #    width = height = min(width, height)  # Ensure square output

        elif method == "crop to face" or method == "crop to face (keep_size)":
            # Find face and get crop coordinates
            crop_coords = self.find_face_crop(image, min(width, height))
            
            if crop_coords is None:
                # If no face found, center crop
                crop_size = min(ow, oh)
                x = (ow - crop_size) // 2
                y = (oh - crop_size) // 2
                x2 = x + crop_size
                y2 = y + crop_size
            else:
                x, y, x2, y2 = crop_coords
            
            # Crop the image
            image = image[:, y:y2, x:x2, :]
            
            if method == "crop to face (keep_size)":
                # For square crop to face, we're done - use the original dimensions
                width = int(x2 - x)  # Convert numpy.int32 to Python int
                height = int(y2 - y)  # Convert numpy.int32 to Python int
            else:
                # Now resize the cropped square to the target size
                if width == 0:
                    width = height
                if height == 0:
                    height = width
                    
                width = height = min(width, height)  # Ensure square output
            
        elif method == 'keep proportion' or method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = oh

            ratio = min(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)

            if method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith('fill'):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) \
            or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) \
            or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]
        
        outputs = torch.clamp(outputs, 0, 1)

        return(outputs, outputs.shape[2], outputs.shape[1],) 