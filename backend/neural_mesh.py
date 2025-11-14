import os
import json
import pickle
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime

class NeuralMesh:
    """
    Neural Mesh for Brein AI - implements graph-based memory connections with Hebbian learning.
    Stores nodes (concepts/memory items) and weighted edges representing associations.
    """

    def __init__(self, mesh_file: str = "memory/neural_mesh.json"):
        """
        Initialize the Neural Mesh with optimized data structures.

        Args:
            mesh_file: Path to store the mesh data
        """
        self.mesh_file = mesh_file
        self.nodes: Dict[str, Dict] = {}  # node_id -> node_data
        self.edges: Dict[Tuple[str, str], Dict] = {}  # (node_a, node_b) -> edge_data
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # node_id -> connected_nodes

        # Performance optimizations
        self.node_cache: Dict[str, Dict] = {}  # Recently accessed nodes
        self.cache_size = 1000  # Maximum cached nodes
        self.dirty_nodes = set()  # Track modified nodes for selective saving
        self.dirty_edges = set()  # Track modified edges for selective saving

        # Ensure memory directory exists
        os.makedirs(os.path.dirname(mesh_file), exist_ok=True)

        # Load existing mesh if available
        self._load_mesh()

    def _load_mesh(self):
        """Load existing mesh data from file."""
        if os.path.exists(self.mesh_file):
            try:
                with open(self.mesh_file, 'r') as f:
                    data = json.load(f)

                self.nodes = data.get('nodes', {})
                self.edges = {}

                # Reconstruct edges from stored data
                for edge_key, edge_data in data.get('edges', {}).items():
                    node_a, node_b = edge_key.split('::')
                    self.edges[(node_a, node_b)] = edge_data
                    self.adjacency[node_a].add(node_b)
                    self.adjacency[node_b].add(node_a)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load mesh data: {e}. Starting fresh.")

    def _save_mesh(self):
        """Save mesh data to file."""
        # Convert edges dict to serializable format
        edges_serializable = {}
        for (node_a, node_b), edge_data in self.edges.items():
            edge_key = f"{node_a}::{node_b}"
            edges_serializable[edge_key] = edge_data

        data = {
            'nodes': self.nodes,
            'edges': edges_serializable,
            'last_updated': datetime.now().isoformat()
        }

        with open(self.mesh_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_node(self, node_id: str, node_type: str = "memory",
                 metadata: Optional[Dict] = None) -> None:
        """
        Add a new node to the mesh.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (memory, concept, etc.)
            metadata: Additional node metadata
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                'type': node_type,
                'activation_level': 0.5,
                'created_at': datetime.now().isoformat(),
                'last_activated': datetime.now().isoformat(),
                'activation_count': 0,
                'metadata': metadata or {}
            }
            self._save_mesh()

    def add_edge(self, node_a: str, node_b: str, initial_weight: float = 0.1,
                 edge_type: str = "association") -> None:
        """
        Add or update an edge between two nodes.

        Args:
            node_a: First node ID
            node_b: Second node ID
            initial_weight: Initial edge weight
            edge_type: Type of connection
        """
        # Ensure both nodes exist
        if node_a not in self.nodes:
            self.add_node(node_a)
        if node_b not in self.nodes:
            self.add_node(node_b)

        # Create edge (undirected, so store in consistent order)
        edge_key = tuple(sorted([node_a, node_b]))

        if edge_key not in self.edges:
            self.edges[edge_key] = {
                'weight': initial_weight,
                'reinforcement_count': 1,
                'last_reinforced': datetime.now().isoformat(),
                'edge_type': edge_type,
                'created_at': datetime.now().isoformat()
            }
        else:
            # Update existing edge
            self.edges[edge_key]['weight'] = min(1.0, self.edges[edge_key]['weight'] + initial_weight)
            self.edges[edge_key]['reinforcement_count'] += 1
            self.edges[edge_key]['last_reinforced'] = datetime.now().isoformat()

        # Update adjacency
        self.adjacency[node_a].add(node_b)
        self.adjacency[node_b].add(node_a)

        self._save_mesh()

    def reinforce_connection(self, node_a: str, node_b: str, delta: float = 0.1) -> None:
        """
        Apply Hebbian reinforcement to strengthen the connection between two nodes.
        Creates a connection if one doesn't exist (neurons that fire together wire together).

        Args:
            node_a: First node ID
            node_b: Second node ID
            delta: Amount to increase connection strength
        """
        # Ensure both nodes exist
        if node_a not in self.nodes:
            self.add_node(node_a)
        if node_b not in self.nodes:
            self.add_node(node_b)

        edge_key = tuple(sorted([node_a, node_b]))

        if edge_key not in self.edges:
            # Create new connection (Hebbian learning principle)
            self.edges[edge_key] = {
                'weight': delta,
                'reinforcement_count': 1,
                'last_reinforced': datetime.now().isoformat(),
                'edge_type': 'hebbian',
                'created_at': datetime.now().isoformat()
            }
            # Update adjacency
            self.adjacency[node_a].add(node_b)
            self.adjacency[node_b].add(node_a)
        else:
            # Strengthen existing connection
            self.edges[edge_key]['weight'] = min(1.0, self.edges[edge_key]['weight'] + delta)
            self.edges[edge_key]['reinforcement_count'] += 1
            self.edges[edge_key]['last_reinforced'] = datetime.now().isoformat()

        self._save_mesh()

    def activate_node(self, node_id: str) -> None:
        """
        Activate a node, updating its activation level and timestamp.

        Args:
            node_id: Node to activate
        """
        if node_id in self.nodes:
            self.nodes[node_id]['activation_level'] = min(1.0, self.nodes[node_id]['activation_level'] + 0.1)
            self.nodes[node_id]['last_activated'] = datetime.now().isoformat()
            self.nodes[node_id]['activation_count'] += 1
            self._save_mesh()

    def get_neighbors(self, node_id: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get neighboring nodes sorted by edge weight.

        Args:
            node_id: Node to get neighbors for
            top_k: Limit number of results

        Returns:
            List of (neighbor_id, weight) tuples
        """
        if node_id not in self.adjacency:
            return []

        neighbors = []
        for neighbor_id in self.adjacency[node_id]:
            edge_key = tuple(sorted([node_id, neighbor_id]))
            weight = self.edges[edge_key]['weight']
            neighbors.append((neighbor_id, weight))

        # Sort by weight descending
        neighbors.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            neighbors = neighbors[:top_k]

        return neighbors

    def traverse_mesh(self, start_node: str, max_depth: int = 2,
                     min_weight: float = 0.1) -> Dict[str, List]:
        """
        Traverse the mesh from a starting node using breadth-first search.

        Args:
            start_node: Node to start traversal from
            max_depth: Maximum traversal depth
            min_weight: Minimum edge weight to traverse

        Returns:
            Dictionary with traversal results
        """
        if start_node not in self.nodes:
            return {'nodes': [], 'edges': [], 'depths': []}

        visited = set()
        queue = [(start_node, 0)]  # (node_id, depth)
        traversal_nodes = []
        traversal_edges = []
        node_depths = []

        while queue:
            current_node, depth = queue.pop(0)

            if current_node in visited or depth > max_depth:
                continue

            visited.add(current_node)
            traversal_nodes.append(current_node)
            node_depths.append(depth)

            # Get neighbors above minimum weight
            neighbors = [(nid, w) for nid, w in self.get_neighbors(current_node) if w >= min_weight]

            for neighbor_id, weight in neighbors:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))
                    edge_key = tuple(sorted([current_node, neighbor_id]))
                    traversal_edges.append({
                        'source': current_node,
                        'target': neighbor_id,
                        'weight': weight,
                        'depth': depth
                    })

        return {
            'nodes': traversal_nodes,
            'edges': traversal_edges,
            'depths': node_depths,
            'max_depth_reached': max(node_depths) if node_depths else 0
        }

    def decay_weights(self, decay_rate: float = 0.01) -> None:
        """
        Apply decay to all edge weights over time.

        Args:
            decay_rate: Rate at which weights decay
        """
        for edge_key in self.edges:
            self.edges[edge_key]['weight'] = max(0.0, self.edges[edge_key]['weight'] - decay_rate)

        self._save_mesh()

    def get_mesh_stats(self) -> Dict:
        """Get statistics about the neural mesh."""
        total_edges = len(self.edges)
        total_nodes = len(self.nodes)

        if total_edges > 0:
            weights = [edge_data['weight'] for edge_data in self.edges.values()]
            avg_weight = sum(weights) / len(weights)
            max_weight = max(weights)
            min_weight = min(weights)
        else:
            avg_weight = max_weight = min_weight = 0.0

        # Node type distribution
        node_types = {}
        for node_data in self.nodes.values():
            node_type = node_data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'avg_edge_weight': avg_weight,
            'max_edge_weight': max_weight,
            'min_edge_weight': min_weight,
            'node_types': node_types,
            'avg_degree': (2 * total_edges / total_nodes) if total_nodes > 0 else 0
        }

    def find_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """
        Find densely connected clusters in the mesh.

        Args:
            min_cluster_size: Minimum cluster size to return

        Returns:
            List of node clusters
        """
        # Simple clustering based on connected components
        visited = set()
        clusters = []

        for node_id in self.nodes:
            if node_id not in visited:
                # Find connected component
                cluster = set()
                queue = [node_id]

                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        queue.extend(self.adjacency[current] - visited)

                if len(cluster) >= min_cluster_size:
                    clusters.append(list(cluster))

        return clusters

    def cleanup_weak_connections(self, min_weight: float = 0.05) -> int:
        """
        Remove edges below minimum weight threshold.

        Args:
            min_weight: Minimum weight to keep

        Returns:
            Number of edges removed
        """
        edges_to_remove = []

        for edge_key, edge_data in self.edges.items():
            if edge_data['weight'] < min_weight:
                edges_to_remove.append(edge_key)

        for edge_key in edges_to_remove:
            node_a, node_b = edge_key
            self.adjacency[node_a].discard(node_b)
            self.adjacency[node_b].discard(node_a)
            del self.edges[edge_key]

        if edges_to_remove:
            self._save_mesh()

        return len(edges_to_remove)
