import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class MemoryTransformer:
    """
    Memory Transformer for Brein AI - generates internal thought traces from memory embeddings.
    Uses a small transformer model to process memory context and produce reasoning traces.
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-small", device: str = "cpu"):
        """
        Initialize the Memory Transformer.

        Args:
            model_name: HuggingFace model name for the transformer
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_name = model_name

        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Memory Transformer loaded: {model_name} on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Memory Transformer model: {e}")
            # Fallback to a simple mock implementation
            self.tokenizer = None
            self.model = None
            logger.info("Using mock Memory Transformer implementation")

    def generate_thought_trace(self, memory_embeddings: List[np.ndarray],
                              query: str, max_length: int = 50) -> Dict:
        """
        Generate internal thought trace from memory embeddings and query.

        Args:
            memory_embeddings: List of memory embedding vectors
            query: User query string
            max_length: Maximum length of generated thought trace

        Returns:
            Dictionary with thought trace and metadata
        """
        if not memory_embeddings:
            return {
                "thought_trace": "No memory context available for reasoning.",
                "confidence": 0.0,
                "activated_nodes": 0,
                "reasoning_type": "empty_memory"
            }

        # If model is not loaded, use mock implementation
        if self.model is None or self.tokenizer is None:
            return self._mock_thought_generation(memory_embeddings, query)

        try:
            # Prepare input by combining query with memory context summary
            memory_summary = self._summarize_memory_embeddings(memory_embeddings)
            input_text = f"Query: {query}\nMemory Context: {memory_summary}\nInternal Thought:"

            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(self.device)

            # Generate thought trace
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=inputs.input_ids.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Calculate confidence based on memory coherence
            confidence = self._calculate_memory_coherence(memory_embeddings)

            return {
                "thought_trace": generated_text,
                "confidence": confidence,
                "activated_nodes": len(memory_embeddings),
                "reasoning_type": "transformer_generated",
                "model_used": self.model_name
            }

        except Exception as e:
            logger.error(f"Error generating thought trace: {e}")
            return self._mock_thought_generation(memory_embeddings, query)

    def _summarize_memory_embeddings(self, embeddings: List[np.ndarray],
                                   max_summaries: int = 3) -> str:
        """
        Create a text summary from memory embeddings.
        Since we don't have the original text, we create a semantic summary.
        """
        if not embeddings:
            return "No memories available."

        # Calculate centroid of embeddings
        centroid = np.mean(embeddings, axis=0)

        # Find most representative embeddings (closest to centroid)
        distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
        closest_indices = np.argsort(distances)[:max_summaries]

        # Create summary based on embedding patterns
        summary_parts = []
        for i, idx in enumerate(closest_indices):
            distance = distances[idx]
            coherence = 1.0 / (1.0 + distance)  # Convert distance to coherence score

            if coherence > 0.8:
                summary_parts.append(f"Strong memory pattern {i+1}")
            elif coherence > 0.6:
                summary_parts.append(f"Related memory concept {i+1}")
            else:
                summary_parts.append(f"Distant memory association {i+1}")

        return " | ".join(summary_parts)

    def _calculate_memory_coherence(self, embeddings: List[np.ndarray]) -> float:
        """
        Calculate coherence score based on embedding similarity.
        """
        if len(embeddings) < 2:
            return 0.5

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        if not similarities:
            return 0.5

        # Return average similarity as coherence score
        return float(np.mean(similarities))

    def _mock_thought_generation(self, memory_embeddings: List[np.ndarray], query: str) -> Dict:
        """
        Mock implementation when transformer model is not available.
        """
        num_memories = len(memory_embeddings)

        if num_memories == 0:
            thought = "No relevant memories found. This appears to be a new topic."
            confidence = 0.0
        elif num_memories == 1:
            thought = f"Recalling one relevant memory. The query '{query}' seems related to stored information."
            confidence = 0.6
        else:
            coherence = self._calculate_memory_coherence(memory_embeddings)
            if coherence > 0.7:
                thought = f"Multiple coherent memories activated ({num_memories} total). Strong pattern recognition for query: {query}"
                confidence = min(0.9, coherence)
            else:
                thought = f"Several loosely related memories found ({num_memories} total). Making connections between different concepts."
                confidence = coherence

        return {
            "thought_trace": thought,
            "confidence": confidence,
            "activated_nodes": num_memories,
            "reasoning_type": "mock_implementation",
            "model_used": "none"
        }

    def analyze_memory_patterns(self, embeddings: List[np.ndarray]) -> Dict:
        """
        Analyze patterns in memory embeddings for debugging/insights.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Analysis results
        """
        if not embeddings:
            return {"analysis": "No embeddings to analyze"}

        embeddings_array = np.array(embeddings)

        # Calculate basic statistics
        centroid = np.mean(embeddings_array, axis=0)
        distances = [np.linalg.norm(emb - centroid) for emb in embeddings_array]

        analysis = {
            "num_embeddings": len(embeddings),
            "embedding_dim": embeddings_array.shape[1],
            "avg_distance_from_centroid": float(np.mean(distances)),
            "max_distance_from_centroid": float(np.max(distances)),
            "min_distance_from_centroid": float(np.min(distances)),
            "std_distance_from_centroid": float(np.std(distances)),
            "coherence_score": self._calculate_memory_coherence(embeddings)
        }

        return analysis

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "mock_mode", "model_name": "none"}

        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }