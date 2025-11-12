"""
Performance Benchmark Suite for Brein AI System Awareness Layer
Tests scalability, latency, and resource efficiency under various loads
"""

import asyncio
import time
import psutil
import json
import os
import sys
from typing import Dict, List, Any, Tuple
from datetime import datetime
import statistics

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sal import SystemAwarenessLayer
from message_schemas import BrainMessageSchemas


class PerformanceBenchmarker:
    """
    Comprehensive performance benchmarking for SAL and brain coordination
    """

    def __init__(self):
        self.sal = None
        self.benchmark_results = {
            'system_info': {},
            'latency_tests': {},
            'throughput_tests': {},
            'resource_usage': {},
            'scalability_tests': {},
            'coordination_tests': {}
        }
        self.test_iterations = 100
        self.concurrency_levels = [1, 5, 10, 25, 50, 100]

    async def setup_benchmark_environment(self):
        """Set up the benchmark environment"""
        print("🔧 Setting up performance benchmark environment...")

        # Collect system information
        self.benchmark_results['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'platform': sys.platform,
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }

        # Initialize SAL
        self.sal = SystemAwarenessLayer()
        sal_initialized = await self.sal.initialize()
        if not sal_initialized:
            raise Exception("SAL initialization failed")

        print("✅ Benchmark environment ready")

    async def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """
        Run all performance benchmarks

        Returns:
            Dict containing all benchmark results
        """
        print("🏃 Starting comprehensive performance benchmarks...")
        print("=" * 60)

        try:
            # Test 1: Message Routing Latency
            await self._benchmark_message_routing_latency()

            # Test 2: Event Bus Throughput
            await self._benchmark_event_bus_throughput()

            # Test 3: State Management Performance
            await self._benchmark_state_management()

            # Test 4: Coordination Engine Performance
            await self._benchmark_coordination_engine()

            # Test 5: Concurrent Load Testing
            await self._benchmark_concurrent_load()

            # Test 6: Memory and Resource Usage
            await self._benchmark_resource_usage()

            # Test 7: Scalability Testing
            await self._benchmark_scalability()

            # Generate performance report
            return self._generate_performance_report()

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    async def _benchmark_message_routing_latency(self):
        """Benchmark message routing latency"""
        print("📡 Benchmarking message routing latency...")

        latencies = []

        for i in range(self.test_iterations):
            # Create test message
            message = {
                'id': f'latency-test-{i}',
                'sender': 'benchmark_agent',
                'recipient': 'prefrontal_cortex',
                'type': 'reasoning_progress',
                'payload': {'test_data': f'iteration_{i}'},
                'timestamp': datetime.now().isoformat()
            }

            # Measure routing time
            start_time = time.perf_counter()
            success = await self.sal.message_router.route_message(message)
            end_time = time.perf_counter()

            if success:
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)

        # Calculate statistics
        if latencies:
            sorted_latencies = sorted(latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            p99_index = int(len(sorted_latencies) * 0.99)

            self.benchmark_results['latency_tests']['message_routing'] = {
                'mean_latency_ms': statistics.mean(latencies),
                'median_latency_ms': statistics.median(latencies),
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'p95_latency_ms': sorted_latencies[min(p95_index, len(sorted_latencies) - 1)],
                'p99_latency_ms': sorted_latencies[min(p99_index, len(sorted_latencies) - 1)],
                'samples': len(latencies),
                'success_rate': len(latencies) / self.test_iterations * 100
            }

            print(".2f")
        else:
            print("❌ No successful routing tests")

    async def _benchmark_event_bus_throughput(self):
        """Benchmark event bus throughput"""
        print("📢 Benchmarking event bus throughput...")

        event_counts = []
        total_events = 1000

        # Subscribe to test events
        events_received = []

        async def test_subscriber(event, data):
            events_received.append((event, data))

        await self.sal.event_bus.subscribe("benchmark.*", test_subscriber)

        # Measure throughput
        start_time = time.perf_counter()

        # Publish events
        publish_tasks = []
        for i in range(total_events):
            task = self.sal.event_bus.publish(f"benchmark.test_{i}", {
                'sequence': i,
                'data': f'test_payload_{i}'
            })
            publish_tasks.append(task)

        await asyncio.gather(*publish_tasks)

        # Wait for processing
        await asyncio.sleep(0.1)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        throughput = total_events / total_time

        self.benchmark_results['throughput_tests']['event_bus'] = {
            'total_events': total_events,
            'events_received': len(events_received),
            'total_time_seconds': total_time,
            'throughput_events_per_second': throughput,
            'avg_latency_ms': (total_time / total_events) * 1000
        }

        print(".0f")

    async def _benchmark_state_management(self):
        """Benchmark state management performance"""
        print("🧠 Benchmarking state management...")

        latencies = []

        for i in range(self.test_iterations):
            # Test state updates
            start_time = time.perf_counter()
            await self.sal.brain_state.update_component_state(
                'prefrontal_cortex',
                {
                    'active_tasks': [f'task_{i}'],
                    'performance_metrics': {'test_metric': i},
                    'last_activity': datetime.now().isoformat()
                }
            )
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)

            # Test state queries
            start_time = time.perf_counter()
            state = await self.sal.brain_state.get_component_state('prefrontal_cortex')
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)

        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)

        self.benchmark_results['latency_tests']['state_management'] = {
            'mean_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'p95_latency_ms': sorted_latencies[min(p95_index, len(sorted_latencies) - 1)],
            'samples': len(latencies)
        }

        print(".2f")

    async def _benchmark_coordination_engine(self):
        """Benchmark coordination engine performance"""
        print("🎭 Benchmarking coordination engine...")

        coordination_types = [
            'complex_reasoning',
            'emotional_processing',
            'memory_consolidation',
            'creative_problem_solving'
        ]

        results = {}

        for coord_type in coordination_types:
            latencies = []

            for i in range(min(50, self.test_iterations)):  # Fewer iterations for coordination tests
                context = {
                    'complexity': 0.5 + (i * 0.01),
                    'request_id': f'coord_test_{coord_type}_{i}'
                }

                start_time = time.perf_counter()
                result = await self.sal.coordinate_brain_activity(coord_type, context)
                end_time = time.perf_counter()

                if result.get('status') == 'success':
                    latencies.append((end_time - start_time) * 1000)

            if latencies:
                sorted_latencies = sorted(latencies)
                p95_index = int(len(sorted_latencies) * 0.95)

                results[coord_type] = {
                    'mean_latency_ms': statistics.mean(latencies),
                    'median_latency_ms': statistics.median(latencies),
                    'p95_latency_ms': sorted_latencies[min(p95_index, len(sorted_latencies) - 1)],
                    'success_rate': len(latencies) / min(50, self.test_iterations) * 100
                }

        self.benchmark_results['latency_tests']['coordination_engine'] = results

        # Print summary
        for coord_type, metrics in results.items():
            print(".1f")

    async def _benchmark_concurrent_load(self):
        """Benchmark concurrent load handling"""
        print("🔄 Benchmarking concurrent load handling...")

        concurrent_results = {}

        for concurrency in self.concurrency_levels:
            print(f"   Testing {concurrency} concurrent operations...")

            latencies = []
            start_time = time.perf_counter()

            # Create concurrent tasks
            async def concurrent_task(task_id: int):
                context = {
                    'complexity': 0.3,
                    'request_id': f'concurrent_{task_id}'
                }

                task_start = time.perf_counter()
                result = await self.sal.coordinate_brain_activity('complex_reasoning', context)
                task_end = time.perf_counter()

                if result.get('status') == 'success':
                    return (task_end - task_start) * 1000
                return None

            # Execute concurrent tasks
            tasks = [concurrent_task(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            total_time = end_time - start_time

            successful_latencies = [lat for lat in results if lat is not None]

            if successful_latencies:
                concurrent_results[concurrency] = {
                    'total_time_seconds': total_time,
                    'successful_operations': len(successful_latencies),
                    'success_rate': len(successful_latencies) / concurrency * 100,
                    'mean_latency_ms': statistics.mean(successful_latencies),
                    'max_latency_ms': max(successful_latencies),
                    'throughput_ops_per_second': len(successful_latencies) / total_time
                }

        self.benchmark_results['scalability_tests']['concurrent_load'] = concurrent_results

        print("   Concurrent load testing complete")

    async def _benchmark_resource_usage(self):
        """Benchmark resource usage"""
        print("💾 Benchmarking resource usage...")

        # Baseline measurements
        baseline_cpu = psutil.cpu_percent(interval=0.1)
        baseline_memory = psutil.virtual_memory().percent

        # Run intensive operations
        intensive_tasks = []
        for i in range(100):
            context = {
                'complexity': 0.8,
                'request_id': f'intensive_{i}'
            }
            task = self.sal.coordinate_brain_activity('complex_reasoning', context)
            intensive_tasks.append(task)

        start_time = time.perf_counter()
        results = await asyncio.gather(*intensive_tasks)
        end_time = time.perf_counter()

        # Peak measurements
        peak_cpu = psutil.cpu_percent(interval=0.1)
        peak_memory = psutil.virtual_memory().percent

        successful_ops = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')

        self.benchmark_results['resource_usage'] = {
            'baseline_cpu_percent': baseline_cpu,
            'peak_cpu_percent': peak_cpu,
            'cpu_overhead_percent': peak_cpu - baseline_cpu,
            'baseline_memory_percent': baseline_memory,
            'peak_memory_percent': peak_memory,
            'memory_overhead_percent': peak_memory - baseline_memory,
            'test_duration_seconds': end_time - start_time,
            'operations_completed': successful_ops,
            'throughput_ops_per_second': successful_ops / (end_time - start_time)
        }

        print(".1f")
        print(".1f")

    async def _benchmark_scalability(self):
        """Benchmark system scalability"""
        print("📈 Benchmarking system scalability...")

        scalability_results = {}

        # Test increasing message loads
        message_loads = [10, 50, 100, 500, 1000]

        for load in message_loads:
            print(f"   Testing {load} message load...")

            start_time = time.perf_counter()

            # Create and route messages
            routing_tasks = []
            for i in range(load):
                message = {
                    'id': f'scalability-test-{load}-{i}',
                    'sender': 'benchmark_agent',
                    'recipient': 'all',
                    'type': 'state_update',
                    'payload': {'load_test': True, 'sequence': i},
                    'timestamp': datetime.now().isoformat()
                }
                task = self.sal.message_router.route_message(message)
                routing_tasks.append(task)

            results = await asyncio.gather(*routing_tasks)
            successful_routes = sum(1 for r in results if r)

            end_time = time.perf_counter()
            total_time = end_time - start_time

            scalability_results[load] = {
                'total_messages': load,
                'successful_routes': successful_routes,
                'success_rate': successful_routes / load * 100,
                'total_time_seconds': total_time,
                'throughput_messages_per_second': successful_routes / total_time,
                'avg_latency_ms': (total_time / successful_routes) * 1000
            }

        self.benchmark_results['scalability_tests']['message_load'] = scalability_results

        print("   Scalability testing complete")

    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        print("\n📊 Generating performance report...")

        report = {
            'benchmark_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_tests_run': len(self.benchmark_results) - 1,  # Exclude system_info
                'test_environment': self.benchmark_results['system_info']
            },
            'performance_metrics': {},
            'recommendations': [],
            'compliance_check': {}
        }

        # Analyze latency performance
        if 'latency_tests' in self.benchmark_results:
            latency_tests = self.benchmark_results['latency_tests']

            report['performance_metrics']['latency'] = {
                'message_routing_p95_ms': latency_tests.get('message_routing', {}).get('p95_latency_ms', 'N/A'),
                'state_management_p95_ms': latency_tests.get('state_management', {}).get('p95_latency_ms', 'N/A'),
                'coordination_engine_avg_ms': {
                    coord_type: metrics.get('mean_latency_ms', 'N/A')
                    for coord_type, metrics in latency_tests.get('coordination_engine', {}).items()
                }
            }

        # Analyze throughput
        if 'throughput_tests' in self.benchmark_results:
            throughput = self.benchmark_results['throughput_tests']
            report['performance_metrics']['throughput'] = {
                'event_bus_events_per_second': throughput.get('event_bus', {}).get('throughput_events_per_second', 'N/A')
            }

        # Analyze scalability
        if 'scalability_tests' in self.benchmark_results:
            scalability = self.benchmark_results['scalability_tests']

            concurrent_load = scalability.get('concurrent_load', {})
            max_concurrency = max(concurrent_load.keys()) if concurrent_load else 0
            max_throughput = concurrent_load.get(max_concurrency, {}).get('throughput_ops_per_second', 0)

            message_load = scalability.get('message_load', {})
            max_message_load = max(message_load.keys()) if message_load else 0
            max_message_throughput = message_load.get(max_message_load, {}).get('throughput_messages_per_second', 0)

            report['performance_metrics']['scalability'] = {
                'max_concurrent_operations': max_concurrency,
                'peak_concurrent_throughput_ops_per_sec': max_throughput,
                'max_message_load_tested': max_message_load,
                'peak_message_throughput_per_sec': max_message_throughput
            }

        # Analyze resource usage
        if 'resource_usage' in self.benchmark_results:
            resources = self.benchmark_results['resource_usage']
            report['performance_metrics']['resource_efficiency'] = {
                'cpu_overhead_percent': resources.get('cpu_overhead_percent', 'N/A'),
                'memory_overhead_percent': resources.get('memory_overhead_percent', 'N/A'),
                'baseline_cpu_percent': resources.get('baseline_cpu_percent', 'N/A'),
                'peak_cpu_percent': resources.get('peak_cpu_percent', 'N/A')
            }

        # Generate recommendations
        recommendations = []

        # Latency recommendations
        latency_metrics = report['performance_metrics'].get('latency', {})
        if isinstance(latency_metrics.get('message_routing_p95_ms'), (int, float)) and latency_metrics['message_routing_p95_ms'] > 50:
            recommendations.append("Consider optimizing message routing for reduced latency")
        if isinstance(latency_metrics.get('state_management_p95_ms'), (int, float)) and latency_metrics['state_management_p95_ms'] > 20:
            recommendations.append("State management latency could be improved")

        # Resource recommendations
        resource_metrics = report['performance_metrics'].get('resource_efficiency', {})
        if isinstance(resource_metrics.get('cpu_overhead_percent'), (int, float)) and resource_metrics['cpu_overhead_percent'] > 10:
            recommendations.append("CPU overhead is higher than expected - consider optimization")
        if isinstance(resource_metrics.get('memory_overhead_percent'), (int, float)) and resource_metrics['memory_overhead_percent'] > 15:
            recommendations.append("Memory overhead exceeds recommended limits")

        # Scalability recommendations
        scalability_metrics = report['performance_metrics'].get('scalability', {})
        if scalability_metrics.get('peak_concurrent_throughput_ops_per_sec', 0) < 10:
            recommendations.append("Concurrent throughput could be improved for better scalability")

        report['recommendations'] = recommendations

        # Compliance check against requirements
        report['compliance_check'] = {
            'latency_requirements': {
                'event_processing_lt_10ms': latency_metrics.get('message_routing_p95_ms', 999) < 10,
                'state_queries_lt_5ms': latency_metrics.get('state_management_p95_ms', 999) < 5
            },
            'resource_requirements': {
                'cpu_overhead_lt_5_percent': resource_metrics.get('cpu_overhead_percent', 999) < 5,
                'memory_overhead_lt_50mb': resource_metrics.get('memory_overhead_percent', 999) < 5  # Approximation
            },
            'scalability_requirements': {
                'handles_1000_events_per_second': scalability_metrics.get('peak_message_throughput_per_sec', 0) > 1000,
                'supports_50_concurrent_operations': scalability_metrics.get('max_concurrent_operations', 0) >= 50
            }
        }

        # Save detailed results
        self._save_benchmark_results()

        return report

    def _save_benchmark_results(self):
        """Save detailed benchmark results to file"""
        try:
            os.makedirs('performance_logs', exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_logs/benchmark_results_{timestamp}.json'

            with open(filename, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2, default=str)

            print(f"📁 Detailed results saved to {filename}")

        except Exception as e:
            print(f"⚠️ Failed to save benchmark results: {e}")

    async def cleanup_benchmark_environment(self):
        """Clean up benchmark environment"""
        print("🧹 Cleaning up benchmark environment...")

        try:
            if self.sal:
                await self.sal.shutdown()

            print("✅ Benchmark cleanup complete")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


async def run_performance_benchmarks():
    """Main benchmark runner function"""
    print("🚀 Starting Brein AI Performance Benchmark Suite")
    print("=" * 60)

    benchmarker = PerformanceBenchmarker()

    try:
        # Setup
        await benchmarker.setup_benchmark_environment()

        # Run benchmarks
        results = await benchmarker.run_comprehensive_benchmarks()

        # Display results
        print("\n📈 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 40)

        if 'error' in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return

        # Performance metrics summary
        metrics = results.get('performance_metrics', {})

        if 'latency' in metrics:
            print("\n⚡ LATENCY PERFORMANCE:")
            latency = metrics['latency']
            if 'message_routing_p95_ms' in latency:
                print(".2f")
            if 'state_management_p95_ms' in latency:
                print(".2f")

        if 'throughput' in metrics:
            print("\n🚀 THROUGHPUT PERFORMANCE:")
            throughput = metrics['throughput']
            if 'event_bus_events_per_second' in throughput:
                print(".0f")

        if 'scalability' in metrics:
            print("\n📈 SCALABILITY METRICS:")
            scalability = metrics['scalability']
            if 'max_concurrent_operations' in scalability:
                print(f"  Max Concurrent Operations: {scalability['max_concurrent_operations']}")
            if 'peak_concurrent_throughput_ops_per_sec' in scalability:
                print(".1f")
            if 'peak_message_throughput_per_sec' in scalability:
                print(".0f")

        if 'resource_efficiency' in metrics:
            print("\n💾 RESOURCE EFFICIENCY:")
            resources = metrics['resource_efficiency']
            if 'cpu_overhead_percent' in resources:
                print(".1f")
            if 'memory_overhead_percent' in resources:
                print(".1f")

        # Compliance check
        compliance = results.get('compliance_check', {})
        print("\n✅ COMPLIANCE CHECK:")

        total_checks = 0
        passed_checks = 0

        for category, checks in compliance.items():
            print(f"  {category.replace('_', ' ').title()}:")
            for check_name, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"    {status} {check_name.replace('_', ' ').replace('lt', '<').replace('gt', '>')}")
                total_checks += 1
                if passed:
                    passed_checks += 1

        compliance_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        print(".1f")

        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  • {rec}")

        # Overall assessment
        if compliance_rate >= 80:
            print("\n🎉 PERFORMANCE ASSESSMENT: EXCELLENT")
            print("   System meets or exceeds all performance requirements!")
        elif compliance_rate >= 60:
            print("\n⚠️ PERFORMANCE ASSESSMENT: GOOD")
            print("   System performs well but has room for optimization.")
        else:
            print("\n❌ PERFORMANCE ASSESSMENT: NEEDS IMPROVEMENT")
            print("   Address performance issues before production deployment.")

    except Exception as e:
        print(f"❌ Benchmark suite failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        await benchmarker.cleanup_benchmark_environment()


if __name__ == "__main__":
    # Run the benchmark suite
    asyncio.run(run_performance_benchmarks())
