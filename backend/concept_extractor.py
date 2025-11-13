"""
Semantic Concept Extractor for Brein AI
Implements advanced NLP processing for semantic concept extraction, ontology building,
and context-aware concept analysis.
"""

import os
import re
import json
import pickle
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SemanticConceptExtractor:
    """
    Advanced semantic concept extraction using sentence transformers and graph-based clustering.
    Implements ontology building, context-aware extraction, and concept importance scoring.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "memory/concepts"):
        """
        Initialize the semantic concept extractor with persistent embedding caching.

        Args:
            model_name: SentenceTransformer model for semantic embeddings
            cache_dir: Directory to cache concept data
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedding_model = SentenceTransformer(model_name)

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Concept storage
        self.concepts: Dict[str, Dict] = {}  # concept_id -> concept_data
        self.concept_embeddings: Dict[str, np.ndarray] = {}  # concept_id -> embedding
        self.concept_hierarchy = nx.DiGraph()  # Hierarchical relationships
        self.concept_graph = nx.Graph()  # Semantic relationships

        # Context and temporal tracking
        self.concept_contexts: Dict[str, List[Dict]] = defaultdict(list)  # concept_id -> context_history
        self.concept_timeline: Dict[str, List[datetime]] = defaultdict(list)  # concept_id -> timestamps

        # Cross-reference mappings
        self.memory_to_concepts: Dict[str, Set[str]] = defaultdict(set)  # memory_id -> concept_ids
        self.concept_to_memories: Dict[str, Set[str]] = defaultdict(set)  # concept_id -> memory_ids

        # Persistent embedding cache for text-to-embedding mappings
        self.text_embedding_cache: Dict[str, np.ndarray] = {}
        self.embedding_cache_file = os.path.join(cache_dir, "text_embeddings.pkl")
        self.cache_max_size = 5000  # Maximum cached text embeddings

        # Load existing data
        self._load_concept_data()
        self._load_embedding_cache()

    def _load_embedding_cache(self):
        """Load persistent text embedding cache."""
        try:
            if os.path.exists(self.embedding_cache_file):
                with open(self.embedding_cache_file, 'rb') as f:
                    self.text_embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.text_embedding_cache)} cached text embeddings")
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}. Starting fresh.")
            self.text_embedding_cache = {}

    def _save_embedding_cache(self):
        """Save persistent text embedding cache."""
        try:
            # Limit cache size before saving
            if len(self.text_embedding_cache) > self.cache_max_size:
                # Keep most recently used items
                items = list(self.text_embedding_cache.items())
                # Simple LRU approximation - keep first N items
                self.text_embedding_cache = dict(items[:self.cache_max_size])

            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.text_embedding_cache, f)
        except Exception as e:
            logger.error(f"Could not save embedding cache: {e}")

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text, or None if not cached."""
        cache_key = text.lower().strip()
        return self.text_embedding_cache.get(cache_key)

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding for text."""
        cache_key = text.lower().strip()
        self.text_embedding_cache[cache_key] = embedding

        # Periodically save cache to disk (every 100 new embeddings)
        if len(self.text_embedding_cache) % 100 == 0:
            self._save_embedding_cache()

    def _load_concept_data(self):
        """Load existing concept data from cache."""
        try:
            # Load concepts
            concepts_file = os.path.join(self.cache_dir, "concepts.json")
            if os.path.exists(concepts_file):
                with open(concepts_file, 'r') as f:
                    self.concepts = json.load(f)

            # Load embeddings
            embeddings_file = os.path.join(self.cache_dir, "concept_embeddings.pkl")
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    self.concept_embeddings = pickle.load(f)

            # Load hierarchy
            hierarchy_file = os.path.join(self.cache_dir, "concept_hierarchy.json")
            if os.path.exists(hierarchy_file):
                with open(hierarchy_file, 'r') as f:
                    hierarchy_data = json.load(f)
                    self.concept_hierarchy = nx.node_link_graph(hierarchy_data, edges="links")

            # Load semantic graph
            graph_file = os.path.join(self.cache_dir, "concept_graph.json")
            if os.path.exists(graph_file):
                with open(graph_file, 'r') as f:
                    graph_data = json.load(f)
                    self.concept_graph = nx.node_link_graph(graph_data, edges="links")

            # Load cross-references
            xref_file = os.path.join(self.cache_dir, "cross_references.json")
            if os.path.exists(xref_file):
                with open(xref_file, 'r') as f:
                    xref_data = json.load(f)
                    self.memory_to_concepts = {k: set(v) for k, v in xref_data.get('memory_to_concepts', {}).items()}
                    self.concept_to_memories = {k: set(v) for k, v in xref_data.get('concept_to_memories', {}).items()}

            logger.info(f"Loaded {len(self.concepts)} concepts from cache")

        except Exception as e:
            logger.warning(f"Could not load concept data: {e}. Starting fresh.")

    def _save_concept_data(self):
        """Save concept data to cache."""
        try:
            # Save concepts
            with open(os.path.join(self.cache_dir, "concepts.json"), 'w') as f:
                json.dump(self.concepts, f, indent=2, default=str)

            # Save embeddings
            with open(os.path.join(self.cache_dir, "concept_embeddings.pkl"), 'wb') as f:
                pickle.dump(self.concept_embeddings, f)

            # Save hierarchy
            with open(os.path.join(self.cache_dir, "concept_hierarchy.json"), 'w') as f:
                json.dump(nx.node_link_data(self.concept_hierarchy), f, indent=2, default=str)

            # Save semantic graph
            with open(os.path.join(self.cache_dir, "concept_graph.json"), 'w') as f:
                json.dump(nx.node_link_data(self.concept_graph), f, indent=2, default=str)

            # Save cross-references
            xref_data = {
                'memory_to_concepts': {k: list(v) for k, v in self.memory_to_concepts.items()},
                'concept_to_memories': {k: list(v) for k, v in self.concept_to_memories.items()}
            }
            with open(os.path.join(self.cache_dir, "cross_references.json"), 'w') as f:
                json.dump(xref_data, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save concept data: {e}")

    def extract_concepts_from_text(self, text: str, context: Optional[Dict] = None,
                                  memory_id: Optional[str] = None) -> List[Dict]:
        """
        Extract semantic concepts from text using advanced NLP processing.

        Args:
            text: Input text to extract concepts from
            context: Optional context information (conversation history, user intent, etc.)
            memory_id: Optional memory ID to link concepts to

        Returns:
            List of extracted concepts with metadata
        """
        # Preprocess text
        cleaned_text = self._preprocess_text(text)

        # Extract candidate concepts using multiple strategies
        candidates = self._extract_candidate_concepts(cleaned_text)

        # Generate embeddings for candidates with caching
        if candidates:
            embeddings = []
            uncached_texts = []

            for candidate in candidates:
                cached_emb = self._get_cached_embedding(candidate)
                if cached_emb is not None:
                    embeddings.append(cached_emb)
                else:
                    uncached_texts.append(candidate)
                    embeddings.append(None)  # Placeholder

            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.embedding_model.encode(uncached_texts)

                # Cache new embeddings and fill placeholders
                uncached_idx = 0
                for i, emb in enumerate(embeddings):
                    if emb is None:
                        embeddings[i] = new_embeddings[uncached_idx]
                        self._cache_embedding(uncached_texts[uncached_idx], new_embeddings[uncached_idx])
                        uncached_idx += 1

            embeddings = np.array(embeddings)
        else:
            return []

        # Perform semantic clustering
        concept_clusters = self._cluster_semantic_concepts(candidates, embeddings)

        # Create or update concepts
        extracted_concepts = []
        for cluster in concept_clusters:
            concept_data = self._create_or_update_concept(cluster, context, memory_id)
            if concept_data:
                extracted_concepts.append(concept_data)

        # Update cross-references
        if memory_id:
            self._update_cross_references(memory_id, [c['id'] for c in extracted_concepts])

        # Save updated data
        self._save_concept_data()

        return extracted_concepts

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for concept extraction."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        return text

    def _extract_candidate_concepts(self, text: str) -> List[str]:
        """Extract candidate concepts using multiple NLP strategies."""
        candidates = set()

        # Strategy 1: Noun phrase extraction using simple POS-like patterns
        # Look for sequences that might be concepts
        words = text.lower().split()

        # Single words (filter common words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }

        for word in words:
            word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
            if len(word) >= 3 and word not in stop_words:
                candidates.add(word)

        # Strategy 2: Multi-word phrases (2-4 words)
        for i in range(len(words) - 1):
            if len(words[i]) >= 3 and len(words[i+1]) >= 3:
                phrase = f"{words[i]} {words[i+1]}"
                if not any(stop in phrase for stop in stop_words):
                    candidates.add(phrase)

        for i in range(len(words) - 2):
            if all(len(words[i+j]) >= 3 for j in range(3)):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if not any(stop in phrase for stop in stop_words):
                    candidates.add(phrase)

        return list(candidates)

    def _cluster_semantic_concepts(self, candidates: List[str], embeddings: np.ndarray) -> List[Dict]:
        """
        Cluster concepts semantically using embeddings and density-based clustering.

        Args:
            candidates: List of candidate concept strings
            embeddings: Corresponding embeddings

        Returns:
            List of concept clusters with representative concepts
        """
        if len(candidates) <= 1:
            return [{'concepts': candidates, 'embeddings': embeddings, 'centroid': embeddings[0] if len(embeddings) > 0 else None}]

        # Use DBSCAN for density-based clustering
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(normalized_embeddings)

        # DBSCAN clustering - convert similarity to distance
        distance_matrix = 1 - np.clip(similarity_matrix, 0, 1)  # Ensure values are in [0, 1]
        clustering = DBSCAN(eps=0.7, min_samples=1, metric='precomputed').fit(distance_matrix)

        # Group concepts by cluster
        clusters = defaultdict(list)
        cluster_embeddings = defaultdict(list)

        for idx, label in enumerate(clustering.labels_):
            clusters[label].append(candidates[idx])
            cluster_embeddings[label].append(embeddings[idx])

        # Create cluster summaries
        concept_clusters = []
        for label, concepts in clusters.items():
            cluster_embeds = np.array(cluster_embeddings[label])
            centroid = np.mean(cluster_embeds, axis=0)

            concept_clusters.append({
                'concepts': concepts,
                'embeddings': cluster_embeds,
                'centroid': centroid,
                'size': len(concepts)
            })

        # Sort by cluster size (largest first)
        concept_clusters.sort(key=lambda x: x['size'], reverse=True)

        return concept_clusters

    def _create_or_update_concept(self, cluster: Dict, context: Optional[Dict] = None,
                                 memory_id: Optional[str] = None) -> Optional[Dict]:
        """
        Create or update a concept from a semantic cluster.

        Args:
            cluster: Semantic cluster data
            context: Optional context information
            memory_id: Optional memory ID

        Returns:
            Concept data dictionary
        """
        # Find representative concept (most central in cluster)
        if len(cluster['concepts']) == 1:
            representative = cluster['concepts'][0]
        else:
            # Use the concept closest to centroid
            centroid = cluster['centroid']
            distances = [np.linalg.norm(emb - centroid) for emb in cluster['embeddings']]
            min_idx = np.argmin(distances)
            representative = cluster['concepts'][min_idx]

        # Check if concept already exists
        concept_id = self._get_concept_id(representative)

        if concept_id in self.concepts:
            # Update existing concept
            self._update_existing_concept(concept_id, cluster, context, memory_id)
        else:
            # Create new concept
            concept_id = self._create_new_concept(representative, cluster, context, memory_id)

        # Calculate importance score
        importance_score = self._calculate_concept_importance(concept_id)

        # Update concept data
        self.concepts[concept_id].update({
            'importance_score': importance_score,
            'last_updated': datetime.now(),
            'cluster_size': cluster['size']
        })

        return {
            'id': concept_id,
            'name': representative,
            'importance_score': importance_score,
            'cluster_size': cluster['size'],
            'related_concepts': list(self.concept_graph.neighbors(concept_id))[:5] if concept_id in self.concept_graph else []
        }

    def _get_concept_id(self, concept_name: str) -> str:
        """Generate a consistent concept ID from concept name."""
        import hashlib
        return hashlib.md5(concept_name.lower().encode()).hexdigest()[:16]

    def _create_new_concept(self, representative: str, cluster: Dict, context: Optional[Dict] = None,
                           memory_id: Optional[str] = None) -> str:
        """Create a new concept."""
        concept_id = self._get_concept_id(representative)

        self.concepts[concept_id] = {
            'name': representative,
            'variants': cluster['concepts'],
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'frequency': 1,
            'importance_score': 0.5,
            'cluster_size': cluster['size'],
            'context_history': [],
            'temporal_pattern': []
        }

        # Store embedding
        self.concept_embeddings[concept_id] = cluster['centroid']

        # Add to semantic graph
        self.concept_graph.add_node(concept_id, name=representative, importance=0.5)

        # Add context if provided
        if context:
            self.concept_contexts[concept_id].append({
                'context': context,
                'timestamp': datetime.now(),
                'memory_id': memory_id
            })

        # Add timestamp
        self.concept_timeline[concept_id].append(datetime.now())

        logger.info(f"Created new concept: {representative} (ID: {concept_id})")

        return concept_id

    def _update_existing_concept(self, concept_id: str, cluster: Dict, context: Optional[Dict] = None,
                                memory_id: Optional[str] = None):
        """Update an existing concept."""
        concept = self.concepts[concept_id]

        # Update frequency
        concept['frequency'] += 1

        # Merge variants
        existing_variants = set(concept['variants'])
        new_variants = set(cluster['concepts'])
        concept['variants'] = list(existing_variants.union(new_variants))

        # Update embedding (moving average)
        current_embedding = self.concept_embeddings[concept_id]
        new_centroid = cluster['centroid']
        updated_embedding = (current_embedding * concept['frequency'] + new_centroid) / (concept['frequency'] + 1)
        self.concept_embeddings[concept_id] = updated_embedding

        # Add context
        if context:
            self.concept_contexts[concept_id].append({
                'context': context,
                'timestamp': datetime.now(),
                'memory_id': memory_id
            })

        # Add timestamp
        self.concept_timeline[concept_id].append(datetime.now())

        # Update graph node
        if concept_id in self.concept_graph:
            self.concept_graph.nodes[concept_id]['importance'] = concept.get('importance_score', 0.5)

    def _calculate_concept_importance(self, concept_id: str) -> float:
        """
        Calculate concept importance based on frequency, recency, and contextual relevance.

        Args:
            concept_id: Concept ID to score

        Returns:
            Importance score between 0 and 1
        """
        if concept_id not in self.concepts:
            return 0.0

        concept = self.concepts[concept_id]

        # Frequency score (logarithmic scaling)
        frequency_score = min(1.0, np.log(concept['frequency'] + 1) / np.log(100))

        # Recency score (exponential decay from last access)
        if self.concept_timeline[concept_id]:
            last_access = max(self.concept_timeline[concept_id])
            days_since_access = (datetime.now() - last_access).days
            recency_score = np.exp(-days_since_access / 30)  # 30-day half-life
        else:
            recency_score = 0.5

        # Contextual relevance score
        context_score = 0.0
        if self.concept_contexts[concept_id]:
            # Score based on context diversity and recency
            contexts = self.concept_contexts[concept_id]
            recent_contexts = [c for c in contexts if (datetime.now() - c['timestamp']).days <= 7]
            context_score = min(1.0, len(recent_contexts) / 5)  # Max score for 5+ recent contexts

        # Network centrality score
        if concept_id in self.concept_graph:
            degree = self.concept_graph.degree(concept_id)
            centrality_score = min(1.0, degree / 10)  # Max score for 10+ connections
        else:
            centrality_score = 0.0

        # Weighted combination
        importance_score = (
            frequency_score * 0.3 +
            recency_score * 0.3 +
            context_score * 0.2 +
            centrality_score * 0.2
        )

        return importance_score

    def _update_cross_references(self, memory_id: str, concept_ids: List[str]):
        """Update cross-references between memories and concepts."""
        for concept_id in concept_ids:
            self.memory_to_concepts[memory_id].add(concept_id)
            self.concept_to_memories[concept_id].add(memory_id)

    def build_concept_ontology(self):
        """Build hierarchical concept relationships (ontology)."""
        # Create hierarchy based on semantic similarity and co-occurrence
        concept_ids = list(self.concepts.keys())

        if len(concept_ids) < 2:
            return

        # Compute pairwise similarities
        embeddings = np.array([self.concept_embeddings[cid] for cid in concept_ids])
        similarities = cosine_similarity(embeddings)

        # Build hierarchy using agglomerative clustering approach
        self.concept_hierarchy.clear()

        # Add all concepts as nodes
        for cid in concept_ids:
            self.concept_hierarchy.add_node(cid, name=self.concepts[cid]['name'])

        # Create hierarchical relationships based on similarity thresholds
        for i, cid1 in enumerate(concept_ids):
            for j, cid2 in enumerate(concept_ids):
                if i != j and similarities[i, j] > 0.7:  # High similarity threshold
                    # Determine direction based on concept generality (simplified)
                    name1 = self.concepts[cid1]['name']
                    name2 = self.concepts[cid2]['name']

                    # Simple heuristic: shorter names are more general
                    if len(name1.split()) < len(name2.split()):
                        self.concept_hierarchy.add_edge(cid1, cid2, weight=similarities[i, j])
                    elif len(name2.split()) < len(name1.split()):
                        self.concept_hierarchy.add_edge(cid2, cid1, weight=similarities[i, j])
                    else:
                        # Same length - add bidirectional relationship
                        self.concept_hierarchy.add_edge(cid1, cid2, weight=similarities[i, j])

    def update_semantic_relationships(self):
        """Update semantic relationships in the concept graph."""
        concept_ids = list(self.concepts.keys())

        if len(concept_ids) < 2:
            return

        # Compute pairwise similarities
        embeddings = np.array([self.concept_embeddings[cid] for cid in concept_ids])
        similarities = cosine_similarity(embeddings)

        # Update graph edges
        for i, cid1 in enumerate(concept_ids):
            for j, cid2 in enumerate(concept_ids):
                if i < j:  # Avoid duplicate edges
                    similarity = similarities[i, j]
                    if similarity > 0.3:  # Similarity threshold for connection
                        if self.concept_graph.has_edge(cid1, cid2):
                            # Update existing edge
                            self.concept_graph[cid1][cid2]['weight'] = similarity
                        else:
                            # Add new edge
                            self.concept_graph.add_edge(cid1, cid2, weight=similarity)

        # Remove weak connections
        edges_to_remove = []
        for u, v, data in self.concept_graph.edges(data=True):
            if data['weight'] < 0.2:
                edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            self.concept_graph.remove_edge(u, v)

    def get_concept_context(self, concept_id: str, max_contexts: int = 5) -> List[Dict]:
        """Get context history for a concept."""
        if concept_id not in self.concept_contexts:
            return []

        contexts = self.concept_contexts[concept_id]
        # Sort by recency
        contexts.sort(key=lambda x: x['timestamp'], reverse=True)

        return contexts[:max_contexts]

    def find_related_concepts(self, concept_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find semantically related concepts."""
        if concept_id not in self.concept_graph:
            return []

        neighbors = []
        for neighbor in self.concept_graph.neighbors(concept_id):
            weight = self.concept_graph[concept_id][neighbor]['weight']
            neighbors.append((neighbor, weight))

        # Sort by weight
        neighbors.sort(key=lambda x: x[1], reverse=True)

        return neighbors[:top_k]

    def search_concepts_by_similarity(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search concepts by semantic similarity to query."""
        if not self.concept_embeddings:
            return []

        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]

        # Compute similarities
        similarities = []
        for concept_id, embedding in self.concept_embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((concept_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_concept_evolution(self, concept_id: str) -> Dict:
        """Get temporal evolution data for a concept."""
        if concept_id not in self.concepts:
            return {}

        concept = self.concepts[concept_id]
        timeline = self.concept_timeline[concept_id]

        # Calculate evolution metrics
        if timeline:
            first_appearance = min(timeline)
            last_appearance = max(timeline)
            timespan_days = (last_appearance - first_appearance).days

            # Frequency over time (group by week)
            weekly_counts = defaultdict(int)
            for timestamp in timeline:
                week_key = timestamp.strftime('%Y-%W')
                weekly_counts[week_key] += 1

            evolution_data = {
                'first_appearance': first_appearance,
                'last_appearance': last_appearance,
                'timespan_days': timespan_days,
                'total_occurrences': len(timeline),
                'weekly_frequency': dict(weekly_counts),
                'current_importance': concept.get('importance_score', 0.0)
            }
        else:
            evolution_data = {}

        return evolution_data

    def get_concept_stats(self) -> Dict:
        """Get comprehensive statistics about the concept system."""
        total_concepts = len(self.concepts)

        if total_concepts == 0:
            return {'total_concepts': 0}

        # Importance distribution
        importance_scores = [c.get('importance_score', 0) for c in self.concepts.values()]
        avg_importance = np.mean(importance_scores)
        max_importance = max(importance_scores)

        # Frequency distribution
        frequencies = [c.get('frequency', 0) for c in self.concepts.values()]
        avg_frequency = np.mean(frequencies)
        max_frequency = max(frequencies)

        # Graph statistics
        if self.concept_graph.number_of_nodes() > 0:
            avg_degree = np.mean([d for n, d in self.concept_graph.degree()])
            clustering_coeff = nx.average_clustering(self.concept_graph)
        else:
            avg_degree = 0
            clustering_coeff = 0

        # Hierarchy statistics
        hierarchy_depth = 0
        if self.concept_hierarchy.number_of_nodes() > 0:
            try:
                hierarchy_depth = nx.dag_longest_path_length(self.concept_hierarchy)
            except:
                hierarchy_depth = 0

        return {
            'total_concepts': total_concepts,
            'avg_importance': avg_importance,
            'max_importance': max_importance,
            'avg_frequency': avg_frequency,
            'max_frequency': max_frequency,
            'graph_nodes': self.concept_graph.number_of_nodes(),
            'graph_edges': self.concept_graph.number_of_edges(),
            'avg_degree': avg_degree,
            'clustering_coefficient': clustering_coeff,
            'hierarchy_depth': hierarchy_depth,
            'total_cross_references': sum(len(memories) for memories in self.concept_to_memories.values())
        }

    def cleanup_old_concepts(self, days_threshold: int = 90):
        """Remove concepts that haven't been accessed recently."""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)

        concepts_to_remove = []
        for concept_id, concept in self.concepts.items():
            if self.concept_timeline[concept_id]:
                last_access = max(self.concept_timeline[concept_id])
                if last_access < cutoff_date and concept.get('importance_score', 0) < 0.3:
                    concepts_to_remove.append(concept_id)

        # Remove concepts
        for concept_id in concepts_to_remove:
            if concept_id in self.concepts:
                del self.concepts[concept_id]
            if concept_id in self.concept_embeddings:
                del self.concept_embeddings[concept_id]
            if concept_id in self.concept_contexts:
                del self.concept_contexts[concept_id]
            if concept_id in self.concept_timeline:
                del self.concept_timeline[concept_id]

            # Remove from graphs
            if self.concept_graph.has_node(concept_id):
                self.concept_graph.remove_node(concept_id)
            if self.concept_hierarchy.has_node(concept_id):
                self.concept_hierarchy.remove_node(concept_id)

            # Clean up cross-references
            if concept_id in self.concept_to_memories:
                for memory_id in self.concept_to_memories[concept_id]:
                    self.memory_to_concepts[memory_id].discard(concept_id)
                del self.concept_to_memories[concept_id]

        if concepts_to_remove:
            self._save_concept_data()
            logger.info(f"Cleaned up {len(concepts_to_remove)} old concepts")

        return len(concepts_to_remove)
