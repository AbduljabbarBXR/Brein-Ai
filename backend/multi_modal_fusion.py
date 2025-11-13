"""
Multi-modal Fusion Engine for Brein AI
Implements cross-modal information integration, unified embeddings, and multi-modal reasoning.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
from collections import defaultdict
import torch
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from concept_extractor import SemanticConceptExtractor
from vision_processor import VisionProcessor
from audio_processor import AudioProcessor
from memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class MultiModalFusionEngine:
    """
    Advanced multi-modal fusion engine that integrates information from text, vision, and audio modalities.
    Creates unified representations and enables cross-modal reasoning and memory.
    """

    def __init__(self, memory_manager: MemoryManager, concept_extractor: SemanticConceptExtractor,
                 vision_processor: Optional[VisionProcessor] = None,
                 audio_processor: Optional[AudioProcessor] = None,
                 cache_dir: str = "memory/fusion_cache"):
        """
        Initialize the multi-modal fusion engine.

        Args:
            memory_manager: Memory manager instance
            concept_extractor: Concept extractor instance
            vision_processor: Optional vision processor
            audio_processor: Optional audio processor
            cache_dir: Directory to cache fusion results
        """
        self.memory_manager = memory_manager
        self.concept_extractor = concept_extractor
        self.vision_processor = vision_processor
        self.audio_processor = audio_processor
        self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Fusion cache
        self.fusion_cache: Dict[str, Dict] = {}
        self.cache_max_size = 300

        # Modality weights for fusion
        self.modality_weights = {
            'text': 0.4,
            'vision': 0.35,
            'audio': 0.25
        }

        # Cross-modal relationship tracking
        self.cross_modal_links: Dict[str, Set[str]] = defaultdict(set)

        # Unified embedding space (simplified - would use projection matrices in production)
        self.embedding_dimensions = {
            'text': 384,    # Sentence transformers
            'vision': 512,  # CLIP vision
            'audio': 128    # Custom audio features
        }

    def process_multi_modal_input(self, inputs: Dict[str, Any],
                                 context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process multiple modalities of input and create unified representation.

        Args:
            inputs: Dictionary with different modality inputs
                   {'text': str, 'image': PIL.Image or str, 'audio': AudioSegment or str}
            context: Optional context information

        Returns:
            Dictionary containing fused multi-modal analysis
        """
        # Generate fusion signature for caching
        fusion_signature = self._generate_fusion_signature(inputs)

        # Check cache
        if fusion_signature in self.fusion_cache:
            cached_result = self.fusion_cache[fusion_signature].copy()
            cached_result['cached'] = True
            return cached_result

        # Process each modality
        modality_results = {}

        # Text processing
        if 'text' in inputs and inputs['text']:
            modality_results['text'] = self._process_text_modality(inputs['text'], context)

        # Vision processing
        if 'image' in inputs and inputs['image'] and self.vision_processor:
            modality_results['vision'] = self._process_vision_modality(inputs['image'], context)

        # Audio processing
        if 'audio' in inputs and inputs['audio'] and self.audio_processor:
            modality_results['audio'] = self._process_audio_modality(inputs['audio'], context)

        # Fuse modalities
        fused_result = self._fuse_modalities(modality_results, context)

        # Create unified concepts
        unified_concepts = self._create_unified_concepts(modality_results)

        # Generate cross-modal associations
        cross_modal_links = self._generate_cross_modal_links(modality_results)

        # Store in memory
        memory_result = self._store_multi_modal_memory(fused_result, unified_concepts, context)

        # Create final result
        result = {
            'fusion_signature': fusion_signature,
            'modality_results': modality_results,
            'fused_representation': fused_result,
            'unified_concepts': unified_concepts,
            'cross_modal_links': cross_modal_links,
            'memory_storage': memory_result,
            'processing_timestamp': datetime.now().isoformat(),
            'context': context,
            'cached': False
        }

        # Cache result
        self._cache_fusion_result(fusion_signature, result)

        return result

    def _process_text_modality(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process text input and extract concepts."""
        # Extract concepts from text
        concepts = self.concept_extractor.extract_concepts_from_text(text, context)

        # Generate text embedding (simplified - would use actual embedding in production)
        text_embedding = self.concept_extractor.embedding_model.encode(text)

        return {
            'input': text,
            'concepts': concepts,
            'embedding': text_embedding,
            'confidence': 0.9,
            'processing_time': datetime.now().isoformat()
        }

    def _process_vision_modality(self, image_input: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process image input and extract visual concepts."""
        vision_result = self.vision_processor.process_image(image_input, context)

        if 'error' in vision_result:
            return vision_result

        # Extract concepts from image description
        if vision_result.get('description'):
            text_concepts = self.concept_extractor.extract_concepts_from_text(
                vision_result['description'], context
            )
        else:
            text_concepts = []

        # Combine visual and text concepts
        all_concepts = vision_result.get('concepts', []) + text_concepts

        return {
            'input_type': 'image',
            'vision_analysis': vision_result,
            'concepts': all_concepts,
            'embedding': self._create_vision_embedding(vision_result),
            'confidence': vision_result.get('confidence', 0.8),
            'processing_time': datetime.now().isoformat()
        }

    def _process_audio_modality(self, audio_input: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process audio input and extract audio concepts."""
        audio_result = self.audio_processor.process_audio(audio_input, context)

        if 'error' in audio_result:
            return audio_result

        # Extract concepts from transcription
        transcription = audio_result.get('transcription', {})
        if transcription.get('success') and transcription.get('text'):
            text_concepts = self.concept_extractor.extract_concepts_from_text(
                transcription['text'], context
            )
        else:
            text_concepts = []

        # Combine audio and text concepts
        all_concepts = audio_result.get('concepts', []) + text_concepts

        return {
            'input_type': 'audio',
            'audio_analysis': audio_result,
            'concepts': all_concepts,
            'embedding': self._create_audio_embedding(audio_result),
            'confidence': audio_result.get('transcription', {}).get('confidence', 0.5),
            'processing_time': datetime.now().isoformat()
        }

    def _create_vision_embedding(self, vision_result: Dict) -> np.ndarray:
        """Create a unified embedding from vision analysis."""
        # Simplified embedding creation - would use projection layers in production
        concepts = vision_result.get('concepts', [])
        if concepts:
            # Average concept embeddings (simplified)
            concept_embeddings = []
            for concept in concepts[:5]:  # Use top 5 concepts
                concept_name = concept.get('concept', '')
                if concept_name:
                    emb = self.concept_extractor.embedding_model.encode(concept_name)
                    concept_embeddings.append(emb)

            if concept_embeddings:
                return np.mean(concept_embeddings, axis=0)

        # Fallback: random embedding
        return np.random.randn(self.embedding_dimensions['vision'])

    def _create_audio_embedding(self, audio_result: Dict) -> np.ndarray:
        """Create a unified embedding from audio analysis."""
        # Simplified embedding based on audio features
        features = audio_result.get('audio_features', {})

        # Create feature vector
        feature_vector = np.array([
            features.get('rms_amplitude', 0) / 10000,  # Normalize
            features.get('zero_crossing_rate', 0),
            features.get('duration_seconds', 0) / 60,  # Normalize to minutes
            features.get('speech_percentage', 0) / 100
        ])

        # Pad or truncate to standard dimension
        if len(feature_vector) < self.embedding_dimensions['audio']:
            padding = np.zeros(self.embedding_dimensions['audio'] - len(feature_vector))
            feature_vector = np.concatenate([feature_vector, padding])
        else:
            feature_vector = feature_vector[:self.embedding_dimensions['audio']]

        return feature_vector

    def _fuse_modalities(self, modality_results: Dict[str, Dict], context: Optional[Dict] = None) -> Dict[str, Any]:
        """Fuse information from multiple modalities."""
        if not modality_results:
            return {'error': 'No modality results to fuse'}

        # Collect all concepts
        all_concepts = []
        modality_contributions = {}

        for modality_name, result in modality_results.items():
            if 'error' not in result:
                concepts = result.get('concepts', [])
                all_concepts.extend(concepts)

                # Track modality contributions
                modality_contributions[modality_name] = {
                    'concept_count': len(concepts),
                    'confidence': result.get('confidence', 0.5),
                    'weight': self.modality_weights.get(modality_name, 0.3)
                }

        # Fuse concepts by finding common and complementary information
        fused_concepts = self._fuse_concepts(all_concepts)

        # Create unified embedding (simplified weighted average)
        embeddings = []
        weights = []

        for modality_name, result in modality_results.items():
            if 'embedding' in result and 'error' not in result:
                embeddings.append(result['embedding'])
                weights.append(self.modality_weights.get(modality_name, 0.3))

        if embeddings:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # Weighted average of embeddings (handle different dimensions)
            unified_embedding = self._weighted_embedding_average(embeddings, normalized_weights)
        else:
            unified_embedding = None

        return {
            'fused_concepts': fused_concepts,
            'unified_embedding': unified_embedding,
            'modality_contributions': modality_contributions,
            'fusion_method': 'weighted_average',
            'confidence': self._calculate_fusion_confidence(modality_contributions)
        }

    def _fuse_concepts(self, all_concepts: List[Dict]) -> List[Dict]:
        """Fuse concepts from different modalities."""
        concept_groups = defaultdict(list)

        # Group concepts by name
        for concept in all_concepts:
            name = concept.get('concept', '').lower()
            if name:
                concept_groups[name].append(concept)

        # Merge concepts within each group
        fused_concepts = []
        for concept_name, concept_list in concept_groups.items():
            if len(concept_list) == 1:
                fused_concepts.append(concept_list[0])
            else:
                # Merge multiple instances
                merged_concept = {
                    'concept': concept_name,
                    'modalities': list(set(c.get('source', 'unknown') for c in concept_list)),
                    'avg_confidence': np.mean([c.get('confidence', 0.5) for c in concept_list]),
                    'max_confidence': max(c.get('confidence', 0.5) for c in concept_list),
                    'category': concept_list[0].get('category', 'unknown'),
                    'fused_from_modalities': True,
                    'modality_count': len(concept_list)
                }
                fused_concepts.append(merged_concept)

        # Sort by confidence
        fused_concepts.sort(key=lambda x: x.get('avg_confidence', x.get('confidence', 0)), reverse=True)

        return fused_concepts[:15]  # Return top 15 fused concepts

    def _weighted_embedding_average(self, embeddings: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Compute weighted average of embeddings with different dimensions."""
        # Normalize all embeddings to same dimension (simplified approach)
        target_dim = max(len(emb) for emb in embeddings)

        normalized_embeddings = []
        for emb in embeddings:
            if len(emb) < target_dim:
                # Pad with zeros
                padded = np.pad(emb, (0, target_dim - len(emb)))
                normalized_embeddings.append(padded)
            elif len(emb) > target_dim:
                # Truncate
                normalized_embeddings.append(emb[:target_dim])
            else:
                normalized_embeddings.append(emb)

        # Compute weighted average
        weighted_sum = np.zeros(target_dim)
        for emb, weight in zip(normalized_embeddings, weights):
            weighted_sum += emb * weight

        return weighted_sum

    def _calculate_fusion_confidence(self, modality_contributions: Dict) -> float:
        """Calculate overall confidence of the fusion."""
        if not modality_contributions:
            return 0.0

        confidences = [contrib.get('confidence', 0.5) for contrib in modality_contributions.values()]
        weights = [contrib.get('weight', 0.3) for contrib in modality_contributions.values()]

        # Weighted average confidence
        total_weight = sum(weights)
        if total_weight == 0:
            return np.mean(confidences)

        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight

        # Bonus for multiple modalities
        modality_bonus = min(0.1, len(modality_contributions) * 0.02)

        return min(1.0, weighted_confidence + modality_bonus)

    def _create_unified_concepts(self, modality_results: Dict[str, Dict]) -> List[Dict]:
        """Create unified concepts that span multiple modalities."""
        unified_concepts = []

        # Find concepts that appear in multiple modalities
        concept_occurrences = defaultdict(lambda: defaultdict(list))

        for modality_name, result in modality_results.items():
            if 'error' not in result:
                for concept in result.get('concepts', []):
                    concept_name = concept.get('concept', '').lower()
                    if concept_name:
                        concept_occurrences[concept_name][modality_name].append(concept)

        # Create unified concepts for those appearing in multiple modalities
        for concept_name, modality_data in concept_occurrences.items():
            if len(modality_data) > 1:  # Appears in multiple modalities
                # Create unified concept
                modalities = list(modality_data.keys())
                all_instances = [inst for instances in modality_data.values() for inst in instances]

                unified_concept = {
                    'concept': concept_name,
                    'type': 'unified_multi_modal',
                    'modalities': modalities,
                    'modality_count': len(modalities),
                    'avg_confidence': np.mean([c.get('confidence', 0.5) for c in all_instances]),
                    'max_confidence': max(c.get('confidence', 0.5) for c in all_instances),
                    'category': all_instances[0].get('category', 'unknown'),
                    'unified': True,
                    'cross_modal_strength': len(modalities) / len(modality_results)  # Fraction of modalities
                }

                unified_concepts.append(unified_concept)

        return unified_concepts

    def _generate_cross_modal_links(self, modality_results: Dict[str, Dict]) -> List[Dict]:
        """Generate links between concepts from different modalities."""
        links = []

        # Get concepts from each modality
        modality_concepts = {}
        for modality_name, result in modality_results.items():
            if 'error' not in result:
                modality_concepts[modality_name] = [
                    c.get('concept', '').lower() for c in result.get('concepts', [])
                    if c.get('concept')
                ]

        # Find overlapping concepts
        all_concepts = set()
        for concepts in modality_concepts.values():
            all_concepts.update(concepts)

        for concept in all_concepts:
            modalities_with_concept = [
                modality for modality, concepts in modality_concepts.items()
                if concept in concepts
            ]

            if len(modalities_with_concept) > 1:
                links.append({
                    'concept': concept,
                    'modalities': modalities_with_concept,
                    'link_type': 'shared_concept',
                    'strength': len(modalities_with_concept) / len(modality_results),
                    'timestamp': datetime.now().isoformat()
                })

        return links

    def _store_multi_modal_memory(self, fused_result: Dict, unified_concepts: List[Dict],
                                 context: Optional[Dict] = None) -> Dict[str, Any]:
        """Store multi-modal information in memory."""
        # Create a combined text representation
        memory_text = self._create_memory_text(fused_result, unified_concepts)

        # Store in memory
        memory_id = self.memory_manager.add_memory(
            content=memory_text,
            memory_type='multi_modal',
            metadata={
                'modality_info': fused_result.get('modality_contributions', {}),
                'unified_concepts': len(unified_concepts),
                'fusion_confidence': fused_result.get('confidence', 0.5),
                'context': context
            }
        )

        # Link concepts to memory
        for concept in unified_concepts:
            concept_id = self.concept_extractor._get_concept_id(concept['concept'])
            self.concept_extractor._update_cross_references(memory_id, [concept_id])

        return {
            'memory_id': memory_id,
            'stored_content': memory_text,
            'linked_concepts': len(unified_concepts)
        }

    def _create_memory_text(self, fused_result: Dict, unified_concepts: List[Dict]) -> str:
        """Create a text representation of multi-modal content for memory storage."""
        text_parts = []

        # Add fused concepts
        if unified_concepts:
            concept_names = [c['concept'] for c in unified_concepts[:5]]
            text_parts.append(f"Multi-modal concepts: {', '.join(concept_names)}")

        # Add modality information
        modality_contributions = fused_result.get('modality_contributions', {})
        if modality_contributions:
            modality_info = []
            for modality, info in modality_contributions.items():
                modality_info.append(f"{modality} ({info.get('concept_count', 0)} concepts)")
            text_parts.append(f"Modalities: {', '.join(modality_info)}")

        return ". ".join(text_parts)

    def _generate_fusion_signature(self, inputs: Dict[str, Any]) -> str:
        """Generate a signature for fusion caching."""
        # Create a hash based on input types and content
        import hashlib

        signature_parts = []
        for modality, content in inputs.items():
            if content:
                if isinstance(content, str):
                    signature_parts.append(f"{modality}:{content[:100]}")
                else:
                    signature_parts.append(f"{modality}:{type(content).__name__}")

        signature_string = "|".join(signature_parts)
        return hashlib.md5(signature_string.encode()).hexdigest()[:16]

    def _cache_fusion_result(self, signature: str, result: Dict):
        """Cache fusion result."""
        if len(self.fusion_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.fusion_cache))
            del self.fusion_cache[oldest_key]

        self.fusion_cache[signature] = result.copy()

    def search_multi_modal(self, query: str, modalities: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search across multi-modal memories.

        Args:
            query: Search query
            modalities: Optional list of modalities to search

        Returns:
            Search results with multi-modal information
        """
        # Search text memories
        text_results = self.memory_manager.search_memory(query)

        # Filter for multi-modal memories
        multi_modal_results = [
            result for result in text_results
            if result.get('type') == 'multi_modal'
        ]

        # Enhance results with modality information
        enhanced_results = []
        for result in multi_modal_results:
            metadata = result.get('metadata', {})
            modality_info = metadata.get('modality_info', {})

            enhanced_result = result.copy()
            enhanced_result['modality_info'] = modality_info
            enhanced_result['unified_concepts'] = metadata.get('unified_concepts', 0)
            enhanced_result['fusion_confidence'] = metadata.get('fusion_confidence', 0.5)

            enhanced_results.append(enhanced_result)

        return {
            'query': query,
            'results': enhanced_results,
            'total_results': len(enhanced_results),
            'search_timestamp': datetime.now().isoformat()
        }

    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get statistics about multi-modal fusion operations."""
        return {
            'cached_fusions': len(self.fusion_cache),
            'max_cache_size': self.cache_max_size,
            'cache_utilization': len(self.fusion_cache) / self.cache_max_size,
            'modality_weights': self.modality_weights,
            'cross_modal_links': len(self.cross_modal_links)
        }
