"""
Communication Protocols Test Suite
Tests enhanced message routing, coordination patterns, and error recovery
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

from sal import SystemAwarenessLayer, MessageRouter, CoordinationEngine
from message_schemas import BrainMessageSchemas, BrainMessageType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CommunicationProtocolsTester:
    """
    Comprehensive test suite for enhanced communication protocols
    """

    def __init__(self):
        self.sal = None
        self.test_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'performance_metrics': {}
        }

    async def setup_test_environment(self):
        """Set up the test environment"""
        logger.info("Setting up communication protocols test environment...")

        try:
            # Initialize SAL
            self.sal = SystemAwarenessLayer()
            sal_initialized = await self.sal.initialize()
            if not sal_initialized:
                raise Exception("SAL initialization failed")

            logger.info("Test environment setup complete")
            return True

        except Exception as e:
            logger.error(f"Test environment setup failed: {e}")
            return False

    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """
        Run the complete communication protocols test suite

        Returns:
            Dict containing test results and metrics
        """
        logger.info("Starting comprehensive communication protocols test suite...")

        # Test 1: Intelligent Message Routing
        await self._test_intelligent_routing()

        # Test 2: Alternative Route Discovery
        await self._test_alternative_routing()

        # Test 3: Error Recovery and Retry Logic
        await self._test_error_recovery()

        # Test 4: Advanced Coordination Patterns
        await self._test_advanced_coordination()

        # Test 5: Multi-Modal Coordination
        await self._test_multi_modal_coordination()

        # Test 6: Crisis Response Coordination
        await self._test_crisis_response()

        # Test 7: Performance Under Load
        await self._test_performance_under_load()

        # Generate test report
        return await self._generate_test_report()

    async def _test_intelligent_routing(self):
        """Test intelligent message routing with rules"""
        self._start_test("Intelligent Message Routing")

        try:
            # Test high-priority emotional message broadcasting
            emotional_schema = BrainMessageSchemas.emotional_context(
                sender="amygdala",
                tone="anxious",
                urgency="high",
                triggers=["stress", "deadline"]
            )

            # Create proper message format
            emotional_message = {
                'id': 'test-emotional-001',
                'sender': 'amygdala',
                'recipient': 'all',  # Should trigger broadcast due to high urgency
                'type': emotional_schema['type'],
                'payload': emotional_schema['payload'],
                'timestamp': '2025-01-01T00:00:00'
            }

            # This should trigger broadcast routing due to high urgency
            success = await self.sal.message_router.route_message(emotional_message)
            assert success, "High-priority emotional message should route successfully"

            # Test complexity-based routing
            complexity_schema = BrainMessageSchemas.complexity_assessment(
                sender="thalamus_router",
                query_complexity=0.9,
                recommended_agent="prefrontal_cortex"
            )

            complexity_message = {
                'id': 'test-complexity-001',
                'sender': 'thalamus_router',
                'recipient': 'prefrontal_cortex',
                'type': complexity_schema['type'],
                'payload': complexity_schema['payload'],
                'timestamp': '2025-01-01T00:00:00'
            }

            success = await self.sal.message_router.route_message(complexity_message)
            assert success, "Complexity assessment should route successfully"

            # Check routing analytics
            analytics = await self.sal.message_router.get_routing_analytics()
            assert 'success_rate' in analytics, "Routing analytics should be available"
            assert analytics['success_rate'] > 0.8, "Routing success rate should be high"

            self._pass_test("Intelligent Message Routing")
        except Exception as e:
            self._fail_test("Intelligent Message Routing", str(e))

    async def _test_alternative_routing(self):
        """Test alternative route discovery when primary routes fail"""
        self._start_test("Alternative Route Discovery")

        try:
            # Test with a valid message type that has fallback routes
            test_message = {
                'id': 'test-alt-route-001',
                'sender': 'hippocampus',  # Valid sender
                'recipient': 'invalid_recipient',  # Invalid recipient
                'type': 'memory_boost',  # Has fallback routes
                'payload': {'test': 'data'},
                'timestamp': '2025-01-01T00:00:00'
            }

            # Test fallback route discovery - should find alternative routes
            alternative = await self.sal.message_router._find_alternative_route(test_message)
            assert alternative in ['hippocampus', 'prefrontal_cortex'], f"Should find valid alternative route, got: {alternative}"

            # Test with emotional context message
            emotional_message = {
                'id': 'test-emotional-alt-001',
                'sender': 'amygdala',
                'recipient': 'invalid_recipient',
                'type': 'emotional_context',
                'payload': {'test': 'data'},
                'timestamp': '2025-01-01T00:00:00'
            }

            alternative_emotional = await self.sal.message_router._find_alternative_route(emotional_message)
            assert alternative_emotional in ['amygdala', 'prefrontal_cortex'], f"Should find valid emotional alternative route, got: {alternative_emotional}"

            # Test with message type that has no fallback routes
            unknown_message = {
                'id': 'test-unknown-001',
                'sender': 'unknown_sender',
                'recipient': 'invalid_recipient',
                'type': 'unknown_message_type',
                'payload': {'test': 'data'},
                'timestamp': '2025-01-01T00:00:00'
            }

            no_alternative = await self.sal.message_router._find_alternative_route(unknown_message)
            assert no_alternative is None, "Should return None for message types with no fallback routes"

            self._pass_test("Alternative Route Discovery")
        except Exception as e:
            self._fail_test("Alternative Route Discovery", str(e))

    async def _test_error_recovery(self):
        """Test error recovery and retry mechanisms"""
        self._start_test("Error Recovery and Retry Logic")

        try:
            # Test coordination with error recovery
            crisis_context = {
                'crisis_type': 'resource_exhaustion',
                'severity': 'high',
                'affected_agents': ['hippocampus']
            }

            # This should trigger error recovery patterns
            result = await self.sal.coordinate_brain_activity('crisis_response', crisis_context)

            # Check if recovery was attempted
            assert 'recovery_type' in result or result.get('status') == 'success', "Error recovery should be attempted"

            # Test multiple coordination attempts
            complex_context = {
                'complexity': 0.9,
                'emotional_context': True,
                'time_constraint': 'strict'
            }

            result = await self.sal.coordinate_brain_activity('complex_reasoning', complex_context)
            assert result.get('status') == 'success', "Complex reasoning should succeed with recovery"

            self._pass_test("Error Recovery and Retry Logic")
        except Exception as e:
            self._fail_test("Error Recovery and Retry Logic", str(e))

    async def _test_advanced_coordination(self):
        """Test advanced coordination patterns"""
        self._start_test("Advanced Coordination Patterns")

        try:
            # Test creative problem solving coordination
            creative_context = {
                'domain': 'mathematical',
                'constraints': ['innovative', 'efficient'],
                'time_limit': 300
            }

            result = await self.sal.coordinate_brain_activity('creative_problem_solving', creative_context)
            assert result.get('coordination_type') == 'creative_problem_solving', "Should coordinate creative problem solving"
            assert 'prefrontal_cortex' in result.get('agents_involved', []), "Should involve prefrontal cortex"
            assert result.get('thinking_mode') == 'divergent', "Should use divergent thinking"

            # Test learning adaptation coordination
            learning_context = {
                'focus': 'performance_optimization',
                'objective': 'reduce_response_time',
                'scope': 'system_wide'
            }

            result = await self.sal.coordinate_brain_activity('learning_adaptation', learning_context)
            assert result.get('agents_involved') == ['all'], "Learning should involve all agents"
            assert result.get('adaptation_focus') == 'performance_optimization', "Should focus on performance"

            self._pass_test("Advanced Coordination Patterns")
        except Exception as e:
            self._fail_test("Advanced Coordination Patterns", str(e))

    async def _test_multi_modal_coordination(self):
        """Test multi-modal processing coordination"""
        self._start_test("Multi-Modal Coordination")

        try:
            # Test coordination of multiple input modalities
            multi_modal_context = {
                'modalities': ['text', 'emotional', 'memory'],
                'integration_required': True,
                'priority': 'high'
            }

            result = await self.sal.coordinate_brain_activity('multi_modal_processing', multi_modal_context)
            assert result.get('coordination_type') == 'multi_modal_processing', "Should coordinate multi-modal processing"
            assert 'prefrontal_cortex' in result.get('agents_involved', []), "Should involve prefrontal cortex for integration"

            modalities_processed = result.get('modalities_processed', [])
            assert 'emotional' in modalities_processed, "Should process emotional modality"
            assert 'memory' in modalities_processed, "Should process memory modality"

            self._pass_test("Multi-Modal Coordination")
        except Exception as e:
            self._fail_test("Multi-Modal Coordination", str(e))

    async def _test_crisis_response(self):
        """Test crisis response coordination"""
        self._start_test("Crisis Response Coordination")

        try:
            # Test critical crisis response
            crisis_context = {
                'crisis_type': 'system_overload',
                'severity': 'critical',
                'impact': 'all_operations',
                'response_time_required': 'immediate'
            }

            result = await self.sal.coordinate_brain_activity('crisis_response', crisis_context, priority='critical')
            assert result.get('coordination_type') == 'crisis_response', "Should coordinate crisis response"
            assert result.get('agents_involved') == ['all'], "Crisis should involve all agents"
            assert result.get('response_priority') == 'critical', "Should have critical priority"
            assert result.get('resource_allocation') == 'maximum', "Should allocate maximum resources"

            self._pass_test("Crisis Response Coordination")
        except Exception as e:
            self._fail_test("Crisis Response Coordination", str(e))

    async def _test_performance_under_load(self):
        """Test communication performance under load"""
        self._start_test("Performance Under Load")

        try:
            start_time = time.time()

            # Send multiple complex coordination requests
            tasks = []
            for i in range(10):
                context = {
                    'complexity': 0.5 + (i * 0.05),  # Increasing complexity
                    'emotional_context': i % 2 == 0,  # Alternate emotional context
                    'request_id': f'perf_test_{i}'
                }
                task = self.sal.coordinate_brain_activity('complex_reasoning', context)
                tasks.append(task)

            # Execute all coordinations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            total_time = end_time - start_time

            # Analyze results
            successful_coordinations = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
            success_rate = successful_coordinations / len(results)

            # Performance assertions
            assert success_rate >= 0.8, f"Coordination success rate too low: {success_rate}"
            assert total_time < 30, f"Coordination took too long: {total_time}s"

            # Log performance metrics
            self.test_results['performance_metrics'] = {
                'total_coordinations': len(results),
                'successful_coordinations': successful_coordinations,
                'success_rate': success_rate,
                'total_time': total_time,
                'avg_time_per_coordination': total_time / len(results)
            }

            logger.info(f"Performance test completed: {successful_coordinations}/{len(results)} successful in {total_time:.2f}s")
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

    async def _generate_test_report(self) -> Dict[str, Any]:
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
            'routing_analytics': {},
            'coordination_analytics': {},
            'recommendations': []
        }

        # Add routing and coordination analytics
        if self.sal:
            try:
                routing_analytics = await self.sal.message_router.get_routing_analytics()
                coordination_analytics = await self.sal.coordination_engine.get_coordination_analytics()

                report['routing_analytics'] = routing_analytics
                report['coordination_analytics'] = coordination_analytics
            except Exception as e:
                logger.error(f"Failed to get analytics: {e}")

        # Generate recommendations based on results
        if self.test_results['tests_failed'] > 0:
            report['recommendations'].append("Address failed tests before production deployment")
        if success_rate < 90:
            report['recommendations'].append("Improve protocol reliability - current success rate below 90%")

        perf_metrics = self.test_results.get('performance_metrics', {})
        if perf_metrics.get('avg_time_per_coordination', 0) > 5:
            report['recommendations'].append("Optimize coordination performance - average time too high")

        return report

    async def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("Cleaning up communication protocols test environment...")

        try:
            if self.sal:
                await self.sal.shutdown()

            logger.info("Test environment cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def run_communication_protocols_tests():
    """Main test runner function"""
    print("🔄 Starting Communication Protocols Test Suite...")
    print("=" * 55)

    tester = CommunicationProtocolsTester()

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
            print(f"  Total Coordinations: {perf['total_coordinations']}")
            print(f"  Success Rate: {perf['success_rate']:.1%}")
            print(".2f")

        if test_results['routing_analytics']:
            print("\n📡 Routing Analytics:")
            routing = test_results['routing_analytics']
            if 'success_rate' in routing:
                print(".1%")

        if test_results['coordination_analytics']:
            print("\n🎭 Coordination Analytics:")
            coord = test_results['coordination_analytics']
            if 'success_rate' in coord:
                print(".1%")
                print(".2f")

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
            print("\n🎉 Communication Protocols: PASSED")
            print("   Advanced communication protocols are ready for production!")
        else:
            print("\n⚠️  Communication Protocols: NEEDS ATTENTION")
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
    asyncio.run(run_communication_protocols_tests())
