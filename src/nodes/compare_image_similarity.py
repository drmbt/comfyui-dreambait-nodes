import os
import numpy as np
import torch
from PIL import Image, ImageOps
import hashlib
import pickle
from pathlib import Path
import folder_paths
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompareImageSimilarity:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "folder_path": ("STRING", {
                    "tooltip": "Path to folder containing images to search through."
                }),
                "similarity_method": (["cosine", "euclidean"], {
                    "default": "cosine",
                    "tooltip": "Method to calculate similarity: cosine for angle-based similarity, euclidean for distance-based."
                }),
                "resize_images": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, resize output images to match input image dimensions. If False, keep original dimensions (recommended)."
                }),
                "return_best_n": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 100, 
                    "step": 1,
                    "tooltip": "Return the N most similar images."
                }),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use cached embeddings for faster processing. Cache is only recalculated if folder contents change."
                }),
                "force_refresh_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force recalculation of embeddings regardless of cache."
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable detailed debug logging for troubleshooting."
                })
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "mask", "FILE_PATH", "FILE_NAME", "SIMILARITY_SCORE")
    FUNCTION = "find_similar_image"
    DESCRIPTION = "Finds the most similar image to the input image from a folder of images. Returns images at their original resolution unless resize_images is enabled."
    
    def log_debug(self, message, debug_mode=False):
        """Helper function for conditional debug logging"""
        if debug_mode:
            logger.info(f"DEBUG: {message}")

    def calculate_cosine_similarity(self, feature1, feature2):
        """Calculate cosine similarity between two feature vectors."""
        # Normalize vectors
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        # Calculate cosine similarity
        dot_product = np.dot(feature1, feature2)
        cosine_similarity = dot_product / (norm1 * norm2)
        return cosine_similarity
    
    def calculate_euclidean_distance(self, feature1, feature2):
        """Calculate euclidean distance between two feature vectors."""
        distance = np.linalg.norm(feature1 - feature2)
        # Convert to similarity score (higher is more similar)
        similarity = 1.0 / (1.0 + distance)
        return similarity
        
    def extract_features(self, image, debug_mode=False):
        """
        Extract feature vector from image.
        This is used for similarity comparison only, not for the final output.
        """
        try:
            # Log input type and shape
            self.log_debug(f"Input image type: {type(image)}", debug_mode)
            if isinstance(image, torch.Tensor):
                self.log_debug(f"Input tensor shape: {image.shape}", debug_mode)
                
                # Special case: grayscale input with shape [1, H, W, 1]
                if len(image.shape) == 4 and image.shape[0] == 1 and image.shape[3] == 1:
                    self.log_debug(f"Processing grayscale ComfyUI format [1,H,W,1]", debug_mode)
                    # Extract first image from batch and the single channel
                    try:
                        # Convert from [1,H,W,1] to [H,W]
                        image_hw = image[0, :, :, 0].cpu().numpy()
                        # Convert to PIL - ComfyUI uses float values 0-1
                        image_gray = (image_hw * 255).astype(np.uint8)
                        image_pil = Image.fromarray(image_gray, mode='L').convert('RGB')
                    except Exception as e:
                        self.log_debug(f"Error processing grayscale tensor: {e}", debug_mode)
                        image_pil = Image.new('RGB', (224, 224), color=(128, 128, 128))
                
                # Regular ComfyUI format: [B,H,W,C] where C=3 (RGB)
                elif len(image.shape) == 4:
                    self.log_debug(f"Processing ComfyUI format [B,H,W,C]", debug_mode)
                    # Extract first image from batch
                    try:
                        image_hwc = image[0].cpu().numpy()  # Now we have [H,W,C]
                        
                        # Convert to PIL directly - ComfyUI uses float values 0-1
                        if image.shape[3] >= 3:  # At least 3 channels
                            self.log_debug(f"Converting RGB tensor", debug_mode)
                            image_rgb = (image_hwc[:,:,:3] * 255).astype(np.uint8)
                            image_pil = Image.fromarray(image_rgb, mode='RGB')
                        elif image.shape[3] == 1:  # Grayscale
                            self.log_debug(f"Converting grayscale tensor", debug_mode)
                            image_rgb = (image_hwc[:,:,0] * 255).astype(np.uint8)
                            image_pil = Image.fromarray(image_rgb, mode='L').convert('RGB')
                        else:
                            # Unusual number of channels
                            self.log_debug(f"Unusual channel count: {image.shape[3]}", debug_mode)
                            raise ValueError(f"Unusual channel count: {image.shape[3]}")
                    except Exception as e:
                        self.log_debug(f"Error processing 4D tensor: {e}", debug_mode)
                        # Create grayscale placeholder
                        image_pil = Image.new('RGB', (224, 224), color=(128, 128, 128))
                
                elif len(image.shape) == 3:
                    # Single image, could be either [H,W,C] or [C,H,W]
                    try:
                        if image.shape[2] == 3:  # Last dim is channels, so [H,W,C]
                            self.log_debug(f"Processing [H,W,C] format", debug_mode)
                            image_hwc = image.cpu().numpy()
                            image_rgb = (image_hwc[:,:,:3] * 255).astype(np.uint8)
                            image_pil = Image.fromarray(image_rgb, mode='RGB')
                        else:
                            # Assume [C,H,W] format
                            self.log_debug(f"Processing [C,H,W] format", debug_mode)
                            # Convert to [H,W,C]
                            image_chw = image.cpu().numpy()
                            image_hwc = np.transpose(image_chw, (1, 2, 0))
                            # Use only first 3 channels if there are more
                            channels = min(image_hwc.shape[2], 3)
                            if channels == 1:
                                # Grayscale - duplicate to 3 channels
                                image_rgb = np.repeat(image_hwc[:,:,0:1], 3, axis=2)
                            else:
                                # RGB - use as is or take first 3 channels
                                image_rgb = image_hwc[:,:,:3]
                            image_rgb = (image_rgb * 255).astype(np.uint8)
                            image_pil = Image.fromarray(image_rgb, mode='RGB')
                    except Exception as e:
                        self.log_debug(f"Error processing 3D tensor: {e}", debug_mode)
                        image_pil = Image.new('RGB', (224, 224), color=(128, 128, 128))
                else:
                    # Unusual format - try a fallback approach
                    self.log_debug(f"Unusual tensor format with {len(image.shape)} dimensions", debug_mode)
                    # Create a placeholder image
                    image_pil = Image.new('RGB', (224, 224), color=(128, 128, 128))
                    
            elif isinstance(image, Image.Image):
                self.log_debug(f"Processing PIL image directly", debug_mode)
                image_pil = image
            else:
                self.log_debug(f"Unsupported image type, creating placeholder", debug_mode)
                # Create a placeholder for unsupported types
                image_pil = Image.new('RGB', (224, 224), color=(128, 128, 128))
                
            # Create a copy of the image for feature extraction to avoid modifying the original
            feature_img = image_pil.copy()
            
            # Resize to a standard size ONLY for feature extraction
            feature_img = feature_img.resize((224, 224), Image.Resampling.LANCZOS)
            # Ensure image is in RGB mode
            feature_img = feature_img.convert("RGB")
            # Extract features (simple pixel-based features)
            features = np.array(feature_img).flatten()
            self.log_debug(f"Extracted feature vector with shape: {features.shape}", debug_mode)
            return features
        except Exception as e:
            self.log_debug(f"Critical error in feature extraction: {e}", debug_mode)
            # Return a default feature vector to prevent complete failure
            return np.ones(224*224*3, dtype=np.float32)
            
    def load_image_for_feature_extraction(self, img_path, debug_mode=False):
        """Load image and extract features for similarity comparison.
           This maintains separation between feature extraction and final image loading."""
        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            # Extract features (standardized to 224x224 inside)
            features = self.extract_features(img, debug_mode)
            return features
        except Exception as e:
            self.log_debug(f"Error processing {img_path} for features: {e}", debug_mode)
            return None
            
    def load_image_for_output(self, img_path, resize_to_size=None, debug_mode=False):
        """Load image at original resolution for output, optionally resize if requested."""
        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            
            orig_w, orig_h = img.size
            self.log_debug(f"Loading output image at original size: {orig_w}x{orig_h}", debug_mode)
            
            # Only resize if specifically requested
            if resize_to_size is not None:
                self.log_debug(f"Resizing output image to: {resize_to_size}", debug_mode)
                img = img.resize(resize_to_size, Image.Resampling.LANCZOS)
            
            # Convert to tensor in ComfyUI format [B,H,W,C]
            img = img.convert("RGB")
            # Convert to numpy array first (HWC format)
            img_np = np.array(img).astype(np.float32) / 255.0
            # Add batch dimension to get BHWC format
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, 3]
            
            # Create properly sized mask
            mask = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2], 1), dtype=torch.float32)
            
            return img_tensor, mask
        except Exception as e:
            self.log_debug(f"Error loading image for output: {e}", debug_mode)
            # Return a small placeholder image on failure
            placeholder = Image.new('RGB', (64, 64), color=(128, 128, 128))
            placeholder_np = np.array(placeholder).astype(np.float32) / 255.0
            placeholder_tensor = torch.from_numpy(placeholder_np).unsqueeze(0)
            placeholder_mask = torch.zeros((1, 64, 64, 1), dtype=torch.float32)
            return placeholder_tensor, placeholder_mask
    
    def get_cache_path(self, folder_path):
        """Generate a path for the cache file based on the folder path."""
        # Create a hash of the folder path to use as the cache filename
        folder_hash = hashlib.md5(folder_path.encode()).hexdigest()
        # Store cache in ComfyUI's temp directory
        cache_dir = os.path.join(folder_paths.get_temp_directory(), "drmbt_embedding_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{folder_hash}.pkl")
    
    def get_folder_fingerprint(self, folder_path, image_files):
        """Generate a fingerprint for the folder based on file list and modification times."""
        # Sort files to ensure consistent order
        image_files = sorted(image_files)
        
        # Create fingerprint from filenames and their modification times
        fingerprint_data = []
        for file_path in image_files:
            try:
                mtime = os.path.getmtime(file_path)
                fingerprint_data.append((file_path, mtime))
            except:
                # If file doesn't exist or can't be accessed, just use the path
                fingerprint_data.append((file_path, 0))
                
        # Create a hash of the fingerprint data
        fingerprint_str = str(fingerprint_data)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def load_cache(self, folder_path, folder_fingerprint, debug_mode=False):
        """Load embeddings cache from disk if it exists and matches the folder fingerprint."""
        cache_path = self.get_cache_path(folder_path)
        self.log_debug(f"Looking for cache at: {cache_path}", debug_mode)
        
        if os.path.exists(cache_path):
            try:
                self.log_debug(f"Cache file found, loading...", debug_mode)
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Cache format: {
                    #   'fingerprint': folder_fingerprint,
                    #   'embeddings': {image_path: embedding}
                    # }
                    
                    # Check if fingerprint matches
                    cached_fingerprint = cache_data.get('fingerprint', '')
                    if cached_fingerprint == folder_fingerprint:
                        self.log_debug(f"Fingerprint match: {cached_fingerprint[:8]}...", debug_mode)
                        return cache_data.get('embeddings', {})
                    else:
                        self.log_debug(f"Fingerprint mismatch: cached {cached_fingerprint[:8]}... vs current {folder_fingerprint[:8]}...", debug_mode)
                        print("Folder contents have changed, recalculating embeddings...")
            except Exception as e:
                self.log_debug(f"Error loading cache: {e}", debug_mode)
                print(f"Error loading cache: {e}")
        else:
            self.log_debug(f"No cache file found", debug_mode)
        return None
    
    def save_cache(self, folder_path, folder_fingerprint, embeddings, debug_mode=False):
        """Save embeddings cache to disk."""
        cache_path = self.get_cache_path(folder_path)
        self.log_debug(f"Saving cache to: {cache_path}", debug_mode)
        cache_data = {
            'fingerprint': folder_fingerprint,
            'embeddings': embeddings
        }
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            self.log_debug(f"Successfully saved cache with {len(embeddings)} embeddings", debug_mode)
        except Exception as e:
            self.log_debug(f"Error saving cache: {e}", debug_mode)
            print(f"Error saving cache: {e}")
    
    def find_similar_image(self, input_image, folder_path, similarity_method="cosine", 
                          resize_images=True, return_best_n=1, use_cache=True, 
                          force_refresh_cache=False, debug_mode=False):
        # Process input image to extract features
        self.log_debug(f"Input image tensor shape: {input_image.shape}", debug_mode)
        
        # Get original image dimensions
        # ComfyUI uses [B,H,W,C] format where B=batch, H=height, W=width, C=channels
        if len(input_image.shape) == 4:
            h, w = input_image.shape[1], input_image.shape[2]
            self.log_debug(f"Image dimensions from ComfyUI tensor: {w}x{h}", debug_mode)
        else:
            # Default fallback size if dimensions cannot be determined
            w, h = 512, 512
            self.log_debug(f"Using default dimensions: {w}x{h}", debug_mode)
        
        input_size = (w, h)
        
        # Extract features from the input image
        input_features = self.extract_features(input_image, debug_mode)
        
        # Resolve folder path
        folder_path = folder_path.strip('"').strip("'")
        if "[" in folder_path:
            name, base_dir = folder_paths.annotated_filepath(folder_path)
            if base_dir is not None:
                folder_path = os.path.join(base_dir, name)
                
        if not os.path.isdir(folder_path):
            # Try input directory
            input_folder_path = os.path.join(folder_paths.get_input_directory(), folder_path)
            if os.path.isdir(input_folder_path):
                folder_path = input_folder_path
            else:
                raise ValueError(f"Folder not found: {folder_path}")
        
        # Get all image files in the folder
        legal_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        image_files = []
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in legal_extensions):
                image_files.append(os.path.join(folder_path, filename))
        
        if not image_files:
            raise ValueError(f"No valid image files found in {folder_path}")
        
        # Generate folder fingerprint based on files and their modification times
        folder_fingerprint = self.get_folder_fingerprint(folder_path, image_files)
        self.log_debug(f"Folder fingerprint: {folder_fingerprint}", debug_mode)
        
        # Handle caching
        embeddings = {}  # Will store the embeddings for current run
        
        if use_cache and not force_refresh_cache:
            # Try to load cache based on folder fingerprint
            cached_embeddings = self.load_cache(folder_path, folder_fingerprint, debug_mode)
            
            if cached_embeddings:
                self.log_debug(f"Using cached embeddings for {len(cached_embeddings)} images", debug_mode)
                embeddings = cached_embeddings
        
        # If we don't have embeddings from cache, calculate them
        if not embeddings:
            self.log_debug(f"Calculating embeddings for {len(image_files)} images...", debug_mode)
            for img_path in image_files:
                features = self.load_image_for_feature_extraction(img_path, debug_mode)
                if features is not None:
                    embeddings[img_path] = features
            
            # Save the new embeddings to cache if requested
            if use_cache:
                self.log_debug(f"Saving embeddings cache for {len(embeddings)} images", debug_mode)
                self.save_cache(folder_path, folder_fingerprint, embeddings, debug_mode)
        
        # Process each image and calculate similarity
        similarities = []
        
        for img_path, img_features in embeddings.items():
            # Calculate similarity based on chosen method
            if similarity_method == "cosine":
                similarity = self.calculate_cosine_similarity(input_features, img_features)
            else:  # euclidean
                similarity = self.calculate_euclidean_distance(input_features, img_features)
            
            similarities.append((img_path, similarity))
        
        if not similarities:
            raise ValueError("Could not process any images in the folder")
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        self.log_debug(f"Found {len(similarities)} images, top similarity score: {similarities[0][1]:.4f}", debug_mode)
        
        # Get the top N results
        best_matches = similarities[:return_best_n]
        self.log_debug(f"Returning top {len(best_matches)} matches", debug_mode)
        
        # Load the best matching images
        images = []
        masks = []
        file_paths = []
        file_names = []
        scores = []
        
        for img_path, score in best_matches:
            try:
                self.log_debug(f"Loading matched image: {img_path} (score: {score:.4f})", debug_mode)
                
                # Determine if we should resize or keep original size
                if resize_images:
                    resize_to = input_size
                else:
                    resize_to = None  # No resizing, keep original
                    
                # Load image for output with proper sizing
                img_tensor, mask = self.load_image_for_output(img_path, resize_to, debug_mode)
                
                images.append(img_tensor)
                masks.append(mask)
                file_paths.append(img_path)
                file_names.append(os.path.basename(img_path).rsplit('.', 1)[0])
                scores.append(score)
            except Exception as e:
                self.log_debug(f"Error loading matched image {img_path}: {e}", debug_mode)
                continue
        
        if not images:
            raise ValueError("Could not load any of the best matching images")
            
        # Combine results
        if len(images) == 1:
            self.log_debug(f"Returning single image result with shape: {images[0].shape}", debug_mode)
            return (images[0], masks[0], file_paths[0], file_names[0], scores[0])
        
        # Multiple results - concatenate on batch dimension
        images = torch.cat(images, dim=0)
        self.log_debug(f"Returning multiple image results, tensor shape: {images.shape}", debug_mode)
        return (images, masks[0], file_paths, file_names, scores) 