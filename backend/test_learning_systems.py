"""
Comprehensive Test Suite for Brein AI Learning Systems
Tests memory consolidation, reinforcement learning, neural mesh, and database synchronization
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

from memory_manager import MemoryManager, MemoryConsolidator, NeuralMeshDatabaseBridge
from neural_mesh import NeuralMesh
from user_behavior_analyzer import UserBehaviorAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningSystemsTestSuite:
    """
    Comprehensive test suite for all learning systems in Brein AI.
    Tests Hebbian learning, memory consolidation, reinforcement learning, and synchronization.
    """

    def __init__(self):
        self.memory_manager = None
        self.neural_mesh = None
        self.consolidator = None
        self.bridge = None
        self.test_results = {}

    async def setup_test_environment(self):
        """Set up test environment with clean database and neural mesh"""
        logger.info("Setting up test environment...")

        # Use timestamp-based names for complete isolation
        import time
        self.timestamp = int(time.time())
        db_path = f"memory/test_learning_{self.timestamp}.db"
        mesh_file = f"memory/test_learning_mesh_{self.timestamp}.json"

        # Clean up any existing test data
        self._cleanup_test_data(db_path, mesh_file)

        # Create isolated neural mesh first
        from neural_mesh import NeuralMesh
        neural_mesh = NeuralMesh(mesh_file=mesh_file)

        # Initialize memory manager with isolated components
        self.memory_manager = MemoryManager(
            db_path=db_path,
            embedding_model="all-MiniLM-L6-v2"
        )

        # Replace with isolated neural mesh
        self.memory_manager.neural_mesh = neural_mesh
        # Update bridge reference
        self.memory_manager.neural_mesh_bridge.neural_mesh = neural_mesh

        self.neural_mesh = neural_mesh
        self.consolidator = self.memory_manager.consolidator
        self.bridge = self.memory_manager.neural_mesh_bridge

        logger.info("Test environment initialized with isolated components")

    def _cleanup_test_data(self, db_path: str, mesh_file: str):
        """Clean up test data files"""
        test_files = [
            db_path,
            f"{db_path}-journal",
            mesh_file
        ]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Removed test file: {file_path}")

    async def run_complete_test_suite(self) -> Dict[str, Any]:
        """
        Run the complete learning systems test suite

        Returns:
            Dict with test results and status
        """
        logger.info("🧠 Starting Brein AI Learning Systems Test Suite")
        print("=" * 70)

        try:
            # Setup
            await self.setup_test_environment()

            # Run all tests
            test_results = await self._run_all_tests()

            # Generate final report
            final_report = self._generate_final_report(test_results)

            print("\n" + "=" * 70)
            print("🎯 LEARNING SYSTEMS TEST SUITE COMPLETED")
            print("=" * 70)

            return final_report

        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _run_all_tests(self) -> Dict[str, Any]:
        """Run all learning system tests"""
        tests = [
            ("hebbian_learning", self.test_hebbian_learning),
            ("memory_consolidation", self.test_memory_consolidation),
            ("reinforcement_learning", self.test_reinforcement_learning),
            ("database_synchronization", self.test_database_synchronization),
            ("learning_health_monitoring", self.test_learning_health_monitoring),
            ("end_to_end_learning", self.test_end_to_end_learning)
        ]

        results = {}
        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n🔬 Running test: {test_name}")
            try:
                start_time = time.time()
                test_result = await test_func()
                end_time = time.time()

                test_result["duration"] = end_time - start_time
                results[test_name] = test_result

                if test_result["status"] == "PASSED":
                    passed += 1
                    print(f"  ✅ PASSED ({test_result['duration']:.2f}s)")
                else:
                    print(f"  ❌ FAILED ({test_result['duration']:.2f}s)")
                    if "error" in test_result:
                        print(f"     Error: {test_result['error']}")

            except Exception as e:
                results[test_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "duration": 0
                }
                print(f"  💥 ERROR: {e}")

        results["summary"] = {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "success_rate": passed / total if total > 0 else 0
        }

        return results

    async def test_hebbian_learning(self) -> Dict[str, Any]:
        """Test Hebbian learning: neurons that fire together wire together"""
        try:
            # Create test nodes
            node_a = self.memory_manager.add_memory("Machine learning algorithms", "stable")
            node_b = self.memory_manager.add_memory("Neural network training", "stable")
            node_c = self.memory_manager.add_memory("Deep learning models", "stable")

            # Initially should have no connections
            initial_connections = len(self.neural_mesh.edges)
            if initial_connections > 0:
                return {"status": "FAILED", "error": f"Expected 0 initial connections, got {initial_connections}"}

            # Activate nodes together multiple times (Hebbian learning)
            for _ in range(5):
                self.neural_mesh.activate_node(node_a)
                self.neural_mesh.activate_node(node_b)
                # Reinforce connection between co-activated nodes
                self.neural_mesh.reinforce_connection(node_a, node_b, 0.1)

            # Check if connection was formed
            connections = self.neural_mesh.get_neighbors(node_a)
            connection_to_b = next((conn for conn in connections if conn[0] == node_b), None)

            if not connection_to_b:
                return {"status": "FAILED", "error": "No connection formed between co-activated nodes"}

            connection_strength = connection_to_b[1]
            if connection_strength <= 0:
                return {"status": "FAILED", "error": f"Connection strength is {connection_strength}, expected > 0"}

            # Test that non-co-activated nodes don't get connected
            connections_to_c = [conn for conn in connections if conn[0] == node_c]
            if connections_to_c:
                return {"status": "FAILED", "error": "Unexpected connection to non-co-activated node"}

            return {
                "status": "PASSED",
                "connection_strength": connection_strength,
                "reinforcement_count": self.neural_mesh.edges.get((node_a, node_b), {}).get('reinforcement_count', 0)
            }

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    async def test_memory_consolidation(self) -> Dict[str, Any]:
        """Test memory consolidation system"""
        try:
            # Add test memories
            memories = []
            for i in range(5):
                node_id = self.memory_manager.add_memory(f"Test memory content {i}", "conversational")
                memories.append(node_id)

            # Check initial state
            initial_health = self.consolidator.get_memory_health_report()
            initial_nodes = initial_health.get("total_nodes", 0)

            if initial_nodes < 5:
                return {"status": "FAILED", "error": f"Expected at least 5 nodes, got {initial_nodes}"}

            # Apply memory decay
            decay_result = self.consolidator.apply_memory_decay(24)  # 24 hours
            decayed_count = decay_result.get("nodes_decayed", 0)

            if decayed_count == 0:
                return {"status": "FAILED", "error": "Memory decay did not affect any nodes"}

            # Test consolidation
            consolidation_result = self.consolidator.consolidate_similar_memories()
            consolidation_actions = consolidation_result.get("groups_consolidated", 0)

            # Check final health
            final_health = self.consolidator.get_memory_health_report()
            final_health_score = final_health.get("health_score", 0)

            return {
                "status": "PASSED",
                "initial_nodes": initial_nodes,
                "nodes_decayed": decayed_count,
                "consolidation_actions": consolidation_actions,
                "final_health_score": final_health_score
            }

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    async def test_reinforcement_learning(self) -> Dict[str, Any]:
        """Test reinforcement learning persistence"""
        try:
            # Create test scenario
            node_ids = []
            for i in range(3):
                node_id = self.memory_manager.add_memory(f"Reinforcement test content {i}", "conversational")
                node_ids.append(node_id)

            # Apply reinforcement
            reinforcement_result = self.consolidator.reinforce_memory(
                node_ids, 0.8, "test_reinforcement"
            )

            reinforced_count = reinforcement_result.get("nodes_reinforced", 0)
            if reinforced_count != 3:
                return {"status": "FAILED", "error": f"Expected 3 reinforced nodes, got {reinforced_count}"}

            # Check if reinforcement persisted in database
            conn = sqlite3.connect(self.memory_manager.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT AVG(activation_level), AVG(importance_score), AVG(consolidation_strength)
                FROM nodes WHERE node_id IN (?, ?, ?)
            """, node_ids)

            avg_activation, avg_importance, avg_consolidation = cursor.fetchone()
            conn.close()

            # Verify values increased
            if avg_activation < 0.5:
                return {"status": "FAILED", "error": f"Average activation {avg_activation} too low"}

            if avg_importance < 0.5:
                return {"status": "FAILED", "error": f"Average importance {avg_importance} too low"}

            if avg_consolidation <= 0.0:
                return {"status": "FAILED", "error": f"Average consolidation {avg_consolidation} not increased"}

            return {
                "status": "PASSED",
                "reinforced_nodes": reinforced_count,
                "avg_activation": avg_activation,
                "avg_importance": avg_importance,
                "avg_consolidation": avg_consolidation
            }

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    async def test_database_synchronization(self) -> Dict[str, Any]:
        """Test neural mesh to database synchronization"""
        try:
            # Create test data in database first
            test_node_id = self.memory_manager.add_memory("Sync test content A", "test")
            connected_node_id = self.memory_manager.add_memory("Sync test content B", "test")

            # Add to neural mesh and create connections
            self.neural_mesh.add_node(test_node_id, "test")
            self.neural_mesh.activate_node(test_node_id)

            self.neural_mesh.add_node(connected_node_id, "test")
            self.neural_mesh.add_edge(test_node_id, connected_node_id, 0.5)

            # Perform synchronization
            sync_result = self.bridge.sync_mesh_to_database()

            synced_nodes = sync_result.get("nodes_synced", 0)
            synced_connections = sync_result.get("connections_synced", 0)

            if synced_nodes == 0:
                return {"status": "FAILED", "error": "No nodes were synchronized"}

            # Verify data persisted in database
            conn = sqlite3.connect(self.memory_manager.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM nodes WHERE node_id IN (?, ?)", (test_node_id, connected_node_id))
            db_nodes = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM memory_consolidation_log WHERE node_id IN (?, ?)", (test_node_id, connected_node_id))
            log_entries = cursor.fetchone()[0]

            conn.close()

            if db_nodes != 2:
                return {"status": "FAILED", "error": f"Expected 2 nodes, found {db_nodes}"}

            return {
                "status": "PASSED",
                "synced_nodes": synced_nodes,
                "synced_connections": synced_connections,
                "db_nodes_found": db_nodes,
                "log_entries": log_entries
            }

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    async def test_learning_health_monitoring(self) -> Dict[str, Any]:
        """Test learning system health monitoring"""
        try:
            # Get health report
            health_report = self.consolidator.get_memory_health_report()

            required_fields = [
                "total_nodes", "average_activation", "average_importance",
                "average_consolidation", "health_score"
            ]

            for field in required_fields:
                if field not in health_report:
                    return {"status": "FAILED", "error": f"Missing field: {field}"}

            health_score = health_report.get("health_score", 0)
            if not (0 <= health_score <= 1):
                return {"status": "FAILED", "error": f"Invalid health score: {health_score}"}

            # Test sync status
            sync_status = self.bridge.get_sync_status()

            required_sync_fields = ["last_sync", "needs_sync", "sync_health"]
            for field in required_sync_fields:
                if field not in sync_status:
                    return {"status": "FAILED", "error": f"Missing sync field: {field}"}

            return {
                "status": "PASSED",
                "health_score": health_score,
                "total_nodes": health_report.get("total_nodes", 0),
                "sync_health": sync_status.get("sync_health", 0),
                "needs_sync": sync_status.get("needs_sync", False)
            }

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    async def test_end_to_end_learning(self) -> Dict[str, Any]:
        """Test complete end-to-end learning pipeline"""
        try:
            # Step 1: Add learning content
            content_nodes = []
            for i in range(3):
                node_id = self.memory_manager.add_memory(
                    f"End-to-end learning test content {i}",
                    "conversational"
                )
                content_nodes.append(node_id)

            # Step 2: Simulate user interaction and learning
            for node_id in content_nodes:
                # Activate nodes (simulating memory retrieval)
                self.neural_mesh.activate_node(node_id)

            # Step 3: Apply reinforcement learning
            reinforcement_result = self.consolidator.reinforce_memory(
                content_nodes, 0.9, "end_to_end_test"
            )

            # Step 4: Synchronize learning
            sync_result = self.bridge.sync_mesh_to_database()

            # Step 5: Verify learning persisted
            conn = sqlite3.connect(self.memory_manager.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT AVG(activation_level), AVG(importance_score), AVG(consolidation_strength)
                FROM nodes WHERE node_id IN (?, ?, ?)
            """, content_nodes)

            final_metrics = cursor.fetchone()
            conn.close()

            avg_activation, avg_importance, avg_consolidation = final_metrics

            # Verify learning occurred
            if avg_activation < 0.5:
                return {"status": "FAILED", "error": "Activation levels not sufficiently increased"}

            if avg_importance < 0.5:
                return {"status": "FAILED", "error": "Importance scores not sufficiently increased"}

            if avg_consolidation <= 0.0:
                return {"status": "FAILED", "error": "Consolidation not applied"}

            return {
                "status": "PASSED",
                "content_nodes_created": len(content_nodes),
                "reinforcement_applied": reinforcement_result.get("nodes_reinforced", 0),
                "sync_completed": sync_result.get("nodes_synced", 0),
                "final_activation": avg_activation,
                "final_importance": avg_importance,
                "final_consolidation": avg_consolidation
            }

        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    def _generate_final_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final test report"""
        summary = test_results.get("summary", {})

        report = {
            "status": "PASSED" if summary.get("success_rate", 0) == 1.0 else "PARTIAL" if summary.get("success_rate", 0) > 0.5 else "FAILED",
            "timestamp": datetime.now().isoformat(),
            "test_summary": summary,
            "individual_tests": {k: v for k, v in test_results.items() if k != "summary"},
            "system_health": self._assess_overall_system_health(test_results)
        }

        # Print detailed report
        self._print_detailed_report(report)

        return report

    def _assess_overall_system_health(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall learning system health"""
        summary = test_results.get("summary", {})
        success_rate = summary.get("success_rate", 0)

        if success_rate == 1.0:
            health_status = "EXCELLENT"
            description = "All learning systems functioning perfectly"
        elif success_rate >= 0.8:
            health_status = "GOOD"
            description = "Most learning systems working well"
        elif success_rate >= 0.6:
            health_status = "FAIR"
            description = "Some learning systems need attention"
        elif success_rate >= 0.4:
            health_status = "POOR"
            description = "Multiple learning systems failing"
        else:
            health_status = "CRITICAL"
            description = "Learning systems severely compromised"

        return {
            "status": health_status,
            "description": description,
            "success_rate": success_rate,
            "recommendations": self._generate_health_recommendations(test_results)
        }

    def _generate_health_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on test results"""
        recommendations = []

        for test_name, result in test_results.items():
            if test_name == "summary":
                continue

            if result.get("status") != "PASSED":
                error = result.get("error", "Unknown error")
                recommendations.append(f"Fix {test_name}: {error}")

        if not recommendations:
            recommendations.append("All learning systems healthy - continue monitoring")

        return recommendations

    def _print_detailed_report(self, report: Dict[str, Any]):
        """Print detailed test report"""
        print(f"\n📊 FINAL TEST REPORT")
        print(f"Status: {report['status']}")
        print(f"Timestamp: {report['timestamp']}")

        summary = report.get("test_summary", {})
        print(f"\nTest Summary:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('passed_tests', 0)}")
        print(f"  Failed: {summary.get('failed_tests', 0)}")
        print(".1%")

        health = report.get("system_health", {})
        print(f"\nSystem Health: {health.get('status', 'UNKNOWN')}")
        print(f"Description: {health.get('description', '')}")

        if health.get("recommendations"):
            print(f"\nRecommendations:")
            for rec in health["recommendations"]:
                print(f"  • {rec}")

async def main():
    """Main test execution function"""
    test_suite = LearningSystemsTestSuite()

    try:
        results = await test_suite.run_complete_test_suite()

        # Save results to file
        with open("test_results/learning_systems_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: test_results/learning_systems_test_results.json")

        # Exit with appropriate code
        if results.get("status") == "PASSED":
            print("🎉 ALL TESTS PASSED!")
            return 0
        else:
            print("⚠️  SOME TESTS FAILED - Check recommendations above")
            return 1

    except Exception as e:
        print(f"💥 TEST SUITE CRASHED: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
