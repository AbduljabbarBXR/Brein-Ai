"""
Vision Processor for Brein AI - Multi-modal Image Analysis and Concept Extraction
Implements advanced computer vision capabilities for image understanding and semantic processing.
"""

import os
import base64
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import io
import logging
from datetime import datetime
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from concept_extractor import SemanticConceptExtractor

logger = logging.getLogger(__name__)

class VisionProcessor:
    """
    Advanced vision processing for image analysis, concept extraction, and visual understanding.
    Uses CLIP for vision-language alignment and semantic concept mapping.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 cache_dir: str = "memory/vision_cache"):
        """
        Initialize the vision processor with CLIP model.

        Args:
            model_name: CLIP model to use for vision-language processing
            cache_dir: Directory to cache vision processing results
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize CLIP model and processor
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            logger.info(f"CLIP model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

        # Image processing cache
        self.image_cache: Dict[str, Dict] = {}
        self.cache_max_size = 500

        # Predefined concept categories for image analysis
        self.concept_categories = {
            'objects': ['person', 'car', 'animal', 'food', 'furniture', 'electronics'],
            'scenes': ['indoor', 'outdoor', 'urban', 'nature', 'beach', 'mountain'],
            'activities': ['sports', 'eating', 'working', 'traveling', 'socializing'],
            'emotions': ['happy', 'sad', 'angry', 'surprised', 'calm', 'excited'],
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'bright', 'dark'],
            'qualities': ['beautiful', 'ugly', 'modern', 'ancient', 'clean', 'dirty']
        }

    def process_image(self, image_input: Any, context: Optional[Dict] = None,
                     memory_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an image and extract visual concepts and semantic information.

        Args:
            image_input: PIL Image, file path, or base64 encoded string
            context: Optional context information
            memory_id: Optional memory ID to link concepts to

        Returns:
            Dictionary containing image analysis results
        """
        # Load and preprocess image
        image = self._load_image(image_input)
        if image is None:
            return {"error": "Failed to load image"}

        # Generate image signature for caching
        image_signature = self._generate_image_signature(image)

        # Check cache first
        if image_signature in self.image_cache:
            cached_result = self.image_cache[image_signature].copy()
            cached_result['cached'] = True
            return cached_result

        # Process image
        try:
            # Extract visual features
            visual_features = self._extract_visual_features(image)

            # Generate image description
            description = self._generate_image_description(image)

            # Extract visual concepts
            concepts = self._extract_visual_concepts(image, context)

            # Detect objects and scenes
            detected_objects = self._detect_objects_and_scenes(image)

            # Analyze composition and style
            composition_analysis = self._analyze_composition(image)

            # Create result
            result = {
                'image_signature': image_signature,
                'description': description,
                'visual_features': visual_features,
                'concepts': concepts,
                'detected_objects': detected_objects,
                'composition_analysis': composition_analysis,
                'processing_timestamp': datetime.now().isoformat(),
                'context': context,
                'memory_id': memory_id,
                'cached': False
            }

            # Cache result
            self._cache_result(image_signature, result)

            return result

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": f"Image processing failed: {str(e)}"}

    def _load_image(self, image_input: Any) -> Optional[Image.Image]:
        """Load image from various input formats."""
        try:
            if isinstance(image_input, Image.Image):
                return image_input
            elif isinstance(image_input, str):
                if image_input.startswith('data:image'):
                    # Base64 encoded image
                    header, encoded = image_input.split(',', 1)
                    image_data = base64.b64decode(encoded)
                    return Image.open(io.BytesIO(image_data))
                elif os.path.exists(image_input):
                    # File path
                    return Image.open(image_input)
                else:
                    # Assume base64 without header
                    try:
                        image_data = base64.b64decode(image_input)
                        return Image.open(io.BytesIO(image_data))
                    except:
                        return None
            elif isinstance(image_input, bytes):
                return Image.open(io.BytesIO(image_input))
            else:
                return None
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def _generate_image_signature(self, image: Image.Image) -> str:
        """Generate a unique signature for image caching."""
        # Resize for consistent hashing
        resized = image.resize((64, 64))
        img_array = np.array(resized)

        # Simple hash based on pixel values
        import hashlib
        img_bytes = img_array.tobytes()
        return hashlib.md5(img_bytes).hexdigest()[:16]

    def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract low-level visual features from image."""
        img_array = np.array(image)

        # Basic color statistics
        if len(img_array.shape) == 3:
            # RGB image
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

            color_stats = {
                'red_mean': float(np.mean(r)),
                'green_mean': float(np.mean(g)),
                'blue_mean': float(np.mean(b)),
                'brightness': float(np.mean(img_array)),
                'contrast': float(np.std(img_array))
            }
        else:
            # Grayscale
            color_stats = {
                'gray_mean': float(np.mean(img_array)),
                'brightness': float(np.mean(img_array)),
                'contrast': float(np.std(img_array))
            }

        # Image dimensions and aspect ratio
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1.0

        return {
            'dimensions': {'width': width, 'height': height},
            'aspect_ratio': aspect_ratio,
            'color_statistics': color_stats,
            'mode': image.mode
        }

    def _generate_image_description(self, image: Image.Image) -> str:
        """Generate a natural language description of the image."""
        # Use CLIP for zero-shot image captioning
        # This is a simplified approach - in production, you'd use a dedicated captioning model

        # Prepare candidate descriptions
        candidate_descriptions = [
            "a photograph of a person",
            "a picture of animals",
            "an image of food",
            "a scene from nature",
            "a photo of a building",
            "an image of technology",
            "a picture of art",
            "a photo of transportation",
            "an image of sports",
            "a scene from daily life"
        ]

        try:
            inputs = self.processor(text=candidate_descriptions, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Get top predictions
            top_indices = torch.topk(probs[0], k=3).indices
            top_descriptions = [candidate_descriptions[i] for i in top_indices]

            return ". ".join(top_descriptions)

        except Exception as e:
            logger.warning(f"CLIP description generation failed: {e}")
            return "An image that could not be automatically described"

    def _extract_visual_concepts(self, image: Image.Image, context: Optional[Dict] = None) -> List[Dict]:
        """Extract semantic concepts from the image."""
        concepts = []

        # Use CLIP to score image against concept categories
        for category, category_concepts in self.concept_categories.items():
            try:
                inputs = self.processor(text=category_concepts, images=image, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # Extract high-confidence concepts
                confidence_threshold = 0.1
                for i, prob in enumerate(probs[0]):
                    if prob > confidence_threshold:
                        concepts.append({
                            'concept': category_concepts[i],
                            'category': category,
                            'confidence': float(prob),
                            'source': 'clip_classification'
                        })

            except Exception as e:
                logger.warning(f"Concept extraction failed for category {category}: {e}")

        # Sort by confidence
        concepts.sort(key=lambda x: x['confidence'], reverse=True)

        return concepts[:10]  # Return top 10 concepts

    def _detect_objects_and_scenes(self, image: Image.Image) -> Dict[str, Any]:
        """Detect objects and scene types in the image."""
        # This is a simplified implementation
        # In production, you'd use object detection models like YOLO or DETR

        # For now, return basic scene analysis based on CLIP
        scene_candidates = [
            "indoor scene", "outdoor scene", "urban environment", "natural landscape",
            "home interior", "office space", "restaurant", "street scene"
        ]

        try:
            inputs = self.processor(text=scene_candidates, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Get top scene predictions
            top_indices = torch.topk(probs[0], k=3).indices
            detected_scenes = [
                {'scene': scene_candidates[i], 'confidence': float(probs[0][i])}
                for i in top_indices
            ]

            return {
                'scenes': detected_scenes,
                'objects': []  # Placeholder for object detection
            }

        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")
            return {'scenes': [], 'objects': []}

    def _analyze_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image composition and visual elements."""
        img_array = np.array(image)

        # Basic composition analysis
        height, width = img_array.shape[:2]

        # Rule of thirds analysis (simplified)
        third_h, third_w = height // 3, width // 3

        # Color distribution
        if len(img_array.shape) == 3:
            # Convert to HSV for better color analysis
            hsv = np.array(image.convert('HSV'))
            hue_hist = np.histogram(hsv[:, :, 0], bins=12, range=(0, 180))[0]
            saturation_hist = np.histogram(hsv[:, :, 1], bins=4, range=(0, 255))[0]

            dominant_hue = np.argmax(hue_hist)
            colorfulness = np.mean(hsv[:, :, 1]) / 255.0
        else:
            dominant_hue = 0
            colorfulness = 0.5

        return {
            'dimensions': {'height': height, 'width': width},
            'aspect_ratio': width / height if height > 0 else 1.0,
            'dominant_hue': int(dominant_hue),
            'colorfulness': float(colorfulness),
            'estimated_complexity': self._estimate_visual_complexity(img_array)
        }

    def _estimate_visual_complexity(self, img_array: np.ndarray) -> float:
        """Estimate visual complexity of the image."""
        # Simple edge detection based approach
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Simple edge detection
        dx = np.abs(np.gradient(gray, axis=1))
        dy = np.abs(np.gradient(gray, axis=0))
        edges = np.sqrt(dx**2 + dy**2)

        # Complexity based on edge density
        edge_density = np.mean(edges > 20)  # Threshold for significant edges

        return float(edge_density)

    def _cache_result(self, signature: str, result: Dict):
        """Cache processing result."""
        if len(self.image_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.image_cache))
            del self.image_cache[oldest_key]

        self.image_cache[signature] = result.copy()

    def compare_images(self, image1: Any, image2: Any) -> Dict[str, Any]:
        """Compare two images for similarity."""
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)

        if img1 is None or img2 is None:
            return {"error": "Failed to load one or both images"}

        try:
            inputs = self.processor(images=[img1, img2], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = torch.cosine_similarity(image_features[0], image_features[1], dim=0)

            return {
                'similarity_score': float(similarity),
                'similarity_percentage': float(similarity * 100),
                'are_similar': similarity > 0.8  # Threshold for "similar"
            }

        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            return {"error": f"Comparison failed: {str(e)}"}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the vision processing cache."""
        return {
            'cached_images': len(self.image_cache),
            'max_cache_size': self.cache_max_size,
            'cache_utilization': len(self.image_cache) / self.cache_max_size
        }
