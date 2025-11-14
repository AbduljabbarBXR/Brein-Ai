"""
Test Hebbian Learning in Isolation
"""

import asyncio
import os
import logging
import time
from memory_manager import MemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_hebbian_learning_isolated():
    """Test Hebbian learning with proper isolation"""
    print("🧠 Testing Hebbian Learning in Isolation")
    print("=" * 50)

    # Use timestamp-based database name to avoid conflicts
    timestamp = int(time.time())
    db_path = f"memory/test_hebbian_{timestamp}.db"

    print(f"Using isolated database: {db_path}")

    # Clean up any existing test data with this timestamp
    test_files = [
        db_path,
        f"{db_path}-journal",
        f"memory/test_hebbian_{timestamp}_neural_mesh.json"
    ]
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed existing file: {file_path}")
        except Exception as e:
            print(f"Could not remove {file_path}: {e}")

    try:
        # Create isolated neural mesh with unique file to avoid loading main system data
        from neural_mesh import NeuralMesh
        mesh_file = f"memory/test_hebbian_mesh_{timestamp}.json"
        neural_mesh = NeuralMesh(mesh_file=mesh_file)

        # Create memory manager with isolated database
        memory_manager = MemoryManager(
            db_path=db_path,
            embedding_model="all-MiniLM-L6-v2"
        )

        # Replace the neural mesh with our isolated one
        memory_manager.neural_mesh = neural_mesh

        neural_mesh = memory_manager.neural_mesh

        print(f"Initial neural mesh connections: {len(neural_mesh.edges)}")

        # Create test nodes
        print("Creating test memories...")
        node_a = memory_manager.add_memory("Machine learning algorithms", "stable")
        node_b = memory_manager.add_memory("Neural network training", "stable")
        node_c = memory_manager.add_memory("Deep learning models", "stable")

        print(f"Created nodes: {node_a[:8]}..., {node_b[:8]}..., {node_c[:8]}...")

        # Check connections after creation
        connections_after_creation = len(neural_mesh.edges)
        print(f"Connections after creation: {connections_after_creation}")

        if connections_after_creation > 0:
            print("❌ FAILED: Unexpected connections after node creation")
            return False

        # Activate nodes together multiple times (Hebbian learning)
        print("Activating nodes together (Hebbian learning)...")
        for i in range(5):
            neural_mesh.activate_node(node_a)
            neural_mesh.activate_node(node_b)
            # Reinforce connection between co-activated nodes
            neural_mesh.reinforce_connection(node_a, node_b, 0.1)
            print(f"  Iteration {i+1}: reinforced connection between co-activated nodes")

        # Check if connection was formed
        connections = neural_mesh.get_neighbors(node_a)
        connection_to_b = next((conn for conn in connections if conn[0] == node_b), None)

        if not connection_to_b:
            print("❌ FAILED: No connection formed between co-activated nodes")
            return False

        connection_strength = connection_to_b[1]
        if connection_strength <= 0:
            print(f"❌ FAILED: Connection strength is {connection_strength}, expected > 0")
            return False

        # Test that non-co-activated nodes don't get connected
        connections_to_c = [conn for conn in connections if conn[0] == node_c]
        if connections_to_c:
            print("❌ FAILED: Unexpected connection to non-co-activated node")
            return False

        print("✅ PASSED: Hebbian learning working correctly")
        print(f"  Connection strength: {connection_strength}")
        print(f"  Reinforcement count: {neural_mesh.edges.get((node_a, node_b), {}).get('reinforcement_count', 0)}")
        print(f"  Total connections in mesh: {len(neural_mesh.edges)}")

        return True

    except Exception as e:
        print(f"💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_hebbian_learning_isolated())
    if success:
        print("\n🎉 Hebbian learning test PASSED")
        exit(0)
    else:
        print("\n❌ Hebbian learning test FAILED")
        exit(1)
