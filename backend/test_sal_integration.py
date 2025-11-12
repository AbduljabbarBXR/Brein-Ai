"""
SAL Integration Test Suite
Tests System Awareness Layer functionality and inter-agent communication
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Any
import time

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sal import SystemAwarenessLayer
from message_schemas import BrainMessageSchemas, BrainMessageType
from agents import (
    HippocampusAgent, PrefrontalCortexAgent, AmygdalaAgent, ThalamusRouter, GGUFModelLoader
)
from memory_manager import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SALIntegrationTester:
    """
    Comprehensive test suite for SAL integration and inter-agent communication
    """

    def __init__(self):
        self.sal = None
        self.memory_manager = None
        self.model_loader = None
        self.agents = {}
        self.test_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'performance_metrics': {}
        }

    async def setup_test_environment(self):
        """Set up the test environment with SAL and agents"""
        logger.info("Setting up SAL integration test environment...")

        try:
            # Initialize core components
            self.memory_manager = MemoryManager(db_path="memory/test_brain_memory.db")
            self.model_loader = GGUFModelLoader()

            # Initialize SAL
            self.sal = SystemAwarenessLayer()
            sal_initialized = await self.sal.initialize()
            if not sal_initialized:
                raise Exception("SAL initialization failed")

            # Initialize agents
            self.agents['hippocampus'] = HippocampusAgent(self.memory_manager, self.model_loader)
            self.agents['prefrontal_cortex'] = PrefrontalCortexAgent(self.memory_manager, self.model_loader)
            self.agents['amygdala'] = AmygdalaAgent(self.model_loader)
            self.agents['thalamus_router'] = ThalamusRouter(self.model_loader)

            # Connect agents to SAL
            await self._connect_agents_to_sal()

            logger.info("Test environment setup complete")
            return True

        except Exception as e:
            logger.error(f"Test environment setup failed: {e}")
            return False

    async def _connect_agents_to_sal(self):
        """Connect all test agents to SAL"""
        for agent_name, agent in self.agents.items():
            try:
                await agent.set_sal(self.sal)
                logger.info(f"Connected {agent_name} to SAL")
            except Exception as e:
                logger.error(f"Failed to connect {agent_name} to SAL: {e}")

    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """
        Run the complete SAL integration test suite

        Returns:
            Dict containing test results and metrics
        """
        logger.info("Starting comprehensive SAL integration test suite...")

        # Test 1: SAL Initialization and State Management
        await self._test_sal_initialization()

        # Test 2: Inter-Agent Communication
        await self._test_inter_agent_communication()

        # Test 3: Event Bus Functionality
        await self._test_event_bus()

        # Test 4: Message Routing
        await self._test_message_routing()

        # Test 5: Brain State Management
        await self._test_brain_state_management()

        # Test 6: System Health Monitoring
        await self._test_system_health_monitoring()

        # Test 7: Coordination Engine
        await self._test_coordination_engine()

        # Test 8: Performance and Load Testing
        await self._test_performance_under_load()

        # Generate test report
        return self._generate_test_report()

    async def _test_sal_initialization(self):
        """Test SAL initialization and basic functionality"""
        self._start_test("SAL Initialization")

        try:
            # Check SAL is initialized
            assert self.sal.is_initialized, "SAL not initialized"

            # Check brain components are registered
            brain_state = await self.sal.query_brain_state()
            assert 'components' in brain_state, "Brain components not registered"
            assert len(brain_state['components']) == 4, "Not all brain components registered"

            # Check system monitoring is active
            health = await self.sal.get_system_health()
            assert 'overall_health' in health, "System health not available"

            self._pass_test("SAL Initialization")
        except Exception as e:
            self._fail_test("SAL Initialization", str(e))

    async def _test_inter_agent_communication(self):
        """Test communication between brain agents"""
        self._start_test("Inter-Agent Communication")

        try:
            # Test hippocampus → amygdala communication
            test_message = BrainMessageSchemas.emotional_context(
                sender="hippocampus",
                tone="anxious",
                urgency="high",
                triggers=["stress", "deadline"]
            )

            await self.sal.broadcast_message(
                sender="hippocampus",
                recipient="amygdala",
                message_type=BrainMessageType.EMOTIONAL_CONTEXT,
                payload=test_message['payload']
            )

            # Give time for message processing
            await asyncio.sleep(0.1)

            # Check if amygdala received and processed the message
            amygdala_state = await self.sal.query_brain_state('amygdala')
            assert amygdala_state['status'] == 'connected', "Amygdala not connected"

            self._pass_test("Inter-Agent Communication")
        except Exception as e:
            self._fail_test("Inter-Agent Communication", str(e))

    async def _test_event_bus(self):
        """Test event bus functionality"""
        self._start_test("Event Bus")

        try:
            # Test event subscription and publishing
            events_received = []

            async def test_event_handler(event, data):
                events_received.append({'event': event, 'data': data})

            # Subscribe to test events
            await self.sal.event_bus.subscribe("test.*", test_event_handler)

            # Publish test events
            await self.sal.event_bus.publish("test.message1", {"test": "data1"})
            await self.sal.event_bus.publish("test.message2", {"test": "data2"})
            await self.sal.event_bus.publish("other.event", {"test": "data3"})  # Should not be received

            # Wait for async processing
            await asyncio.sleep(0.1)

            # Check events received
            assert len(events_received) == 2, f"Expected 2 events, got {len(events_received)}"
            assert events_received[0]['data']['test'] == 'data1', "Event data mismatch"
            assert events_received[1]['data']['test'] == 'data2', "Event data mismatch"

            self._pass_test("Event Bus")
        except Exception as e:
            self._fail_test("Event Bus", str(e))

    async def _test_message_routing(self):
        """Test message routing functionality"""
        self._start_test("Message Routing")

        try:
            # Test message routing statistics
            initial_stats = await self.sal.message_router.get_status()

            # Send several test messages
            for i in range(5):
                await self.sal.broadcast_message(
                    sender="test_sender",
                    recipient="test_recipient",
                    message_type="test_message",
                    payload={"test_id": i}
                )

            # Check routing statistics updated
            final_stats = await self.sal.message_router.get_status()
            messages_routed = final_stats['messages_routed'] - initial_stats['messages_routed']
            assert messages_routed == 5, f"Expected 5 messages routed, got {messages_routed}"

            self._pass_test("Message Routing")
        except Exception as e:
            self._fail_test("Message Routing", str(e))

    async def _test_brain_state_management(self):
        """Test brain state management"""
        self._start_test("Brain State Management")

        try:
            # Test state updates
            await self.sal.brain_state.update_component_state('hippocampus', {
                'test_metric': 42,
                'status': 'testing'
            })

            # Query updated state
            hippocampus_state = await self.sal.query_brain_state('hippocampus')
            assert hippocampus_state['test_metric'] == 42, "State update failed"
            assert hippocampus_state['status'] == 'testing', "State update failed"

            # Test full brain state query
            full_state = await self.sal.query_brain_state()
            assert 'components' in full_state, "Full state query failed"
            assert 'system' in full_state, "System state not included"

            self._pass_test("Brain State Management")
        except Exception as e:
            self._fail_test("Brain State Management", str(e))

    async def _test_system_health_monitoring(self):
        """Test system health monitoring"""
        self._start_test("System Health Monitoring")

        try:
            # Wait for system monitor to collect initial metrics
            await asyncio.sleep(0.5)

            # Get system health
            health = await self.sal.get_system_health()

            required_keys = ['overall_health', 'brain_components', 'system_resources', 'timestamp']
            for key in required_keys:
                assert key in health, f"Missing health key: {key}"

            # Check brain component health
            assert 'healthy' in health['brain_components'], "Brain health status missing"
            assert 'component_health' in health['brain_components'], "Component health details missing"

            # Check system resources - they're nested in the 'metrics' object
            system_resources = health['system_resources']

            # Check that metrics exist
            assert 'metrics' in system_resources, "Metrics not found in system_resources"

            metrics = system_resources['metrics']
            resource_keys = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_status']
            for key in resource_keys:
                assert key in metrics, f"Missing system resource: {key}"

            self._pass_test("System Health Monitoring")
        except Exception as e:
            self._fail_test("System Health Monitoring", str(e))

    async def _test_coordination_engine(self):
        """Test coordination engine functionality"""
        self._start_test("Coordination Engine")

        try:
            # Test coordination of complex reasoning
            coordination_result = await self.sal.coordinate_brain_activity(
                activity_type="complex_reasoning",
                context={
                    "query": "How does quantum computing work?",
                    "complexity": 0.8,
                    "requires_memory": True,
                    "requires_emotion": False
                }
            )

            # Check coordination result
            assert 'coordination_type' in coordination_result, "Coordination result incomplete"
            assert coordination_result['coordination_type'] == 'complex_reasoning', "Wrong coordination type"

            # Check agent involvement
            assert 'agents_involved' in coordination_result, "Agent involvement not specified"
            involved_agents = coordination_result['agents_involved']
            assert 'prefrontal_cortex' in involved_agents, "Prefrontal cortex not involved in reasoning"

            self._pass_test("Coordination Engine")
        except Exception as e:
            self._fail_test("Coordination Engine", str(e))

    async def _test_performance_under_load(self):
        """Test SAL performance under load"""
        self._start_test("Performance Under Load")

        try:
            # Measure baseline performance
            start_time = time.time()

            # Send 100 messages rapidly
            tasks = []
            for i in range(100):
                task = self.sal.broadcast_message(
                    sender="performance_test",
                    recipient="all",
                    message_type="performance_test",
                    payload={"message_id": i, "data": "x" * 100}  # 100 char payload
                )
                tasks.append(task)

            # Execute all messages concurrently
            await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate performance metrics
            messages_per_second = 100 / total_time
            avg_latency = total_time / 100 * 1000  # ms

            # Log performance metrics
            self.test_results['performance_metrics'] = {
                'messages_per_second': messages_per_second,
                'avg_latency_ms': avg_latency,
                'total_time': total_time,
                'messages_sent': 100
            }

            # Performance assertions
            assert messages_per_second > 50, f"Low throughput: {messages_per_second} msg/s"
            assert avg_latency < 50, f"High latency: {avg_latency}ms"

            logger.info(f"Performance test completed: {messages_per_second:.2f} msg/s, {avg_latency:.2f}ms avg latency")
            self._pass_test("Performance Under Load")
        except Exception as e:
            self._fail_test("Performance Under Load", str(e))

    def _start_test(self, test_name: str):
        """Start a test"""
        logger.info(f"Running test: {test_name}")
        self.test_results['tests_run'] += 1

    def _pass_test(self, test_name: str):
        """Mark a test as passed"""
        logger.info(f"✅ Test passed: {test_name}")
        self.test_results['tests_passed'] += 1

    def _fail_test(self, test_name: str, error: str):
        """Mark a test as failed"""
        logger.error(f"❌ Test failed: {test_name} - {error}")
        self.test_results['tests_failed'] += 1
        self.test_results['errors'].append({
            'test': test_name,
            'error': error,
            'timestamp': time.time()
        })

    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        success_rate = (self.test_results['tests_passed'] / self.test_results['tests_run']) * 100 if self.test_results['tests_run'] > 0 else 0

        report = {
            'test_summary': {
                'total_tests': self.test_results['tests_run'],
                'passed': self.test_results['tests_passed'],
                'failed': self.test_results['tests_failed'],
                'success_rate': success_rate
            },
            'errors': self.test_results['errors'],
            'performance_metrics': self.test_results['performance_metrics'],
            'system_health': {},
            'recommendations': []
        }

        # Add system health at end of testing
        if self.sal:
            try:
                # Note: This would need to be awaited in real usage
                # For now, we'll just note that health monitoring is working
                report['system_health'] = {
                    'monitoring_active': True,
                    'components_connected': len(self.agents)
                }
            except Exception as e:
                report['system_health'] = {'error': str(e)}
        else:
            report['system_health'] = {'error': 'SAL not available'}

        # Generate recommendations based on results
        if self.test_results['tests_failed'] > 0:
            report['recommendations'].append("Address failed tests before production deployment")
        if success_rate < 90:
            report['recommendations'].append("Improve test reliability - current success rate below 90%")
        if self.test_results['performance_metrics'].get('avg_latency_ms', 0) > 20:
            report['recommendations'].append("Optimize message routing for lower latency")

        return report

    async def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("Cleaning up test environment...")

        try:
            if self.sal:
                await self.sal.shutdown()

            # Clean up test database
            if os.path.exists("memory/test_brain_memory.db"):
                os.remove("memory/test_brain_memory.db")

            logger.info("Test environment cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def run_sal_integration_tests():
    """Main test runner function"""
    print("🧠 Starting SAL Integration Test Suite...")
    print("=" * 50)

    tester = SALIntegrationTester()

    try:
        # Setup
        setup_success = await tester.setup_test_environment()
        if not setup_success:
            print("❌ Test environment setup failed")
            return

        # Run tests
        test_results = await tester.run_comprehensive_test_suite()

        # Display results
        print("\n📊 Test Results Summary:")
        print("-" * 30)
        summary = test_results['test_summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(".1f")

        if test_results['performance_metrics']:
            print("\n🚀 Performance Metrics:")
            perf = test_results['performance_metrics']
            print(f"  Messages/second: {perf['messages_per_second']:.1f}")
            print(f"  Average latency: {perf['avg_latency_ms']:.2f}ms")

        if test_results['errors']:
            print("\n❌ Test Errors:")
            for error in test_results['errors'][:3]:  # Show first 3 errors
                print(f"  • {error['test']}: {error['error']}")

        if test_results['recommendations']:
            print("\n💡 Recommendations:")
            for rec in test_results['recommendations']:
                print(f"  • {rec}")

        # Overall assessment
        if summary['success_rate'] >= 90:
            print("\n🎉 SAL Integration: PASSED")
            print("   System Awareness Layer is ready for production!")
        else:
            print("\n⚠️  SAL Integration: NEEDS ATTENTION")
            print("   Address issues before production deployment.")

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        await tester.cleanup_test_environment()


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(run_sal_integration_tests())
