import asyncio
import time
import json
import os
from typing import Dict, List, Any
import logging
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
from orchestrator import Orchestrator
from web_fetcher import WebFetcher
import numpy as np

logger = logging.getLogger(__name__)

class TestHarness:
    """
    Comprehensive test harness for Brein AI system benchmarking and validation.
    """

    def __init__(self, memory_manager: MemoryManager, orchestrator: Orchestrator):
        self.memory = memory_manager
        self.orchestrator = orchestrator
        self.results_dir = "test_results/"
        os.makedirs(self.results_dir, exist_ok=True)

        # Sample documents for testing
        self.sample_docs = self._generate_sample_documents()

    def _generate_sample_documents(self) -> List[Dict[str, Any]]:
        """Generate 1000+ sample documents for testing."""
        docs = []

        # Technical documentation samples
        tech_docs = [
            {
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns. The field encompasses supervised learning, unsupervised learning, and reinforcement learning.",
                "type": "educational"
            },
            {
                "title": "Neural Networks Architecture",
                "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes called neurons that process and transmit information. Modern neural networks use backpropagation for training and can have multiple layers including input, hidden, and output layers.",
                "type": "technical"
            },
            {
                "title": "Vector Databases Overview",
                "content": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They use similarity search algorithms like FAISS or Annoy to find vectors that are closest to a query vector. These databases are essential for modern AI applications including semantic search and recommendation systems.",
                "type": "technical"
            },
            {
                "title": "Memory Systems in AI",
                "content": "Memory systems in artificial intelligence are crucial for maintaining context and learning from experience. Different types of memory include working memory for immediate tasks, long-term memory for persistent knowledge, and episodic memory for specific events. Modern AI systems often implement hierarchical memory architectures.",
                "type": "research"
            },
            {
                "title": "Natural Language Processing",
                "content": "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves tasks such as text analysis, language translation, sentiment analysis, and text generation. Modern NLP systems use transformer architectures and large language models.",
                "type": "technical"
            }
        ]

        # Generate variations and expansions
        base_docs = tech_docs * 50  # 250 docs

        # Add more diverse content
        additional_docs = [
            {
                "title": f"Advanced Topic {i}",
                "content": f"This document covers advanced concepts in field {i}. It includes detailed explanations of complex algorithms, mathematical foundations, and practical implementations. The content explores various approaches to solving problems in this domain, including both theoretical and applied aspects.",
                "type": "research"
            } for i in range(100)
        ]

        # Add conversational content
        conversational_docs = [
            {
                "title": f"Discussion Thread {i}",
                "content": f"In this conversation, participants discuss topic {i} from multiple perspectives. They share experiences, ask questions, and provide insights about various aspects of the subject. The discussion covers both practical applications and theoretical considerations.",
                "type": "conversational"
            } for i in range(200)
        ]

        # Add functional content
        functional_docs = [
            {
                "title": f"API Documentation {i}",
                "content": f"This API endpoint handles {i} operations. It accepts various parameters and returns structured responses. The endpoint includes error handling, authentication requirements, and usage examples. Rate limiting and caching strategies are also documented.",
                "type": "functional"
            } for i in range(200)
        ]

        # Add news-style content
        news_docs = [
            {
                "title": f"Industry Update {i}",
                "content": f"Recent developments in technology sector {i} have led to significant changes. Companies are adopting new approaches to solve emerging challenges. Industry experts predict continued growth and innovation in this area over the coming months.",
                "type": "news"
            } for i in range(200)
        ]

        # Add stable/reference content
        stable_docs = [
            {
                "title": f"Reference Guide {i}",
                "content": f"This comprehensive reference covers fundamental concepts in area {i}. It provides definitions, examples, and best practices that remain relatively stable over time. The guide serves as a foundation for understanding more advanced topics in this field.",
                "type": "stable"
            } for i in range(200)
        ]

        docs.extend(base_docs)
        docs.extend(additional_docs)
        docs.extend(conversational_docs)
        docs.extend(functional_docs)
        docs.extend(news_docs)
        docs.extend(stable_docs)

        return docs

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive performance and functionality tests.
        """
        logger.info("Starting comprehensive Brein AI test suite...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "test_phases": {},
            "overall_metrics": {}
        }

        # Phase 1: Memory Ingestion Test
        logger.info("Phase 1: Testing memory ingestion performance...")
        ingestion_results = await self._test_memory_ingestion()
        results["test_phases"]["memory_ingestion"] = ingestion_results

        # Phase 2: Memory Retrieval Test
        logger.info("Phase 2: Testing memory retrieval performance...")
        retrieval_results = await self._test_memory_retrieval()
        results["test_phases"]["memory_retrieval"] = retrieval_results

        # Phase 3: Query Processing Test
        logger.info("Phase 3: Testing query processing performance...")
        query_results = await self._test_query_processing()
        results["test_phases"]["query_processing"] = query_results

        # Phase 4: Neural Mesh Test
        logger.info("Phase 4: Testing neural mesh functionality...")
        mesh_results = await self._test_neural_mesh()
        results["test_phases"]["neural_mesh"] = mesh_results

        # Phase 5: Concurrent Load Test
        logger.info("Phase 5: Testing concurrent load handling...")
        load_results = await self._test_concurrent_load()
        results["test_phases"]["concurrent_load"] = load_results

        # Calculate overall metrics
        results["overall_metrics"] = self._calculate_overall_metrics(results)

        # Save results
        self._save_test_results(results)

        logger.info("Comprehensive test suite completed!")
        return results

    async def _test_memory_ingestion(self) -> Dict[str, Any]:
        """Test memory ingestion performance."""
        results = {
            "total_docs": len(self.sample_docs),
            "ingestion_times": [],
            "throughput": 0,
            "memory_stats": {}
        }

        start_time = time.time()

        for i, doc in enumerate(self.sample_docs):
            doc_start = time.time()

            # Ingest document
            node_id = self.memory.add_memory(
                content=doc["content"],
                memory_type=doc["type"]
            )

            doc_time = time.time() - doc_start
            results["ingestion_times"].append(doc_time)

            if (i + 1) % 100 == 0:
                logger.info(f"Ingested {i + 1}/{len(self.sample_docs)} documents...")

        total_time = time.time() - start_time
        results["total_time"] = total_time
        results["throughput"] = len(self.sample_docs) / total_time
        results["avg_ingestion_time"] = np.mean(results["ingestion_times"])
        results["memory_stats"] = self.memory.get_memory_stats()

        return results

    async def _test_memory_retrieval(self) -> Dict[str, Any]:
        """Test memory retrieval performance."""
        test_queries = [
            "machine learning algorithms",
            "neural network architecture",
            "vector database performance",
            "memory systems in AI",
            "natural language processing",
            "artificial intelligence applications",
            "data processing techniques",
            "algorithm optimization"
        ]

        results = {
            "queries_tested": len(test_queries),
            "retrieval_times": [],
            "avg_results_per_query": 0,
            "total_results": 0
        }

        total_results = 0

        for query in test_queries:
            start_time = time.time()

            # Perform search
            search_results = self.memory.search_memory(query, top_k=10, use_mesh_expansion=True)

            retrieval_time = time.time() - start_time
            results["retrieval_times"].append(retrieval_time)
            total_results += len(search_results)

        results["avg_retrieval_time"] = np.mean(results["retrieval_times"])
        results["total_results"] = total_results
        results["avg_results_per_query"] = total_results / len(test_queries)

        return results

    async def _test_query_processing(self) -> Dict[str, Any]:
        """Test full query processing pipeline."""
        test_queries = [
            "Explain machine learning concepts",
            "How do neural networks work?",
            "What are vector databases used for?",
            "Describe memory systems in AI",
            "What is natural language processing?"
        ]

        results = {
            "queries_tested": len(test_queries),
            "processing_times": [],
            "responses_generated": 0,
            "avg_confidence": 0
        }

        total_confidence = 0

        for query in test_queries:
            start_time = time.time()

            # Process query through orchestrator
            result = await self.orchestrator.process_query(query, session_id="test_session")

            processing_time = time.time() - start_time
            results["processing_times"].append(processing_time)

            if result.get("response"):
                results["responses_generated"] += 1
                total_confidence += result.get("confidence", 0)

        results["avg_processing_time"] = np.mean(results["processing_times"])
        results["avg_confidence"] = total_confidence / results["responses_generated"] if results["responses_generated"] > 0 else 0

        return results

    async def _test_neural_mesh(self) -> Dict[str, Any]:
        """Test neural mesh functionality."""
        results = {
            "mesh_stats": self.memory.neural_mesh.get_mesh_stats(),
            "reinforcement_tests": 0,
            "expansion_tests": 0
        }

        # Test reinforcement
        test_node_ids = ["test_node_1", "test_node_2", "test_node_3"]
        for node_id in test_node_ids:
            self.memory.neural_mesh.add_node(node_id, "test")
            results["reinforcement_tests"] += 1

        # Test connections
        self.memory.neural_mesh.reinforce_connection("test_node_1", "test_node_2", 0.5)
        self.memory.neural_mesh.reinforce_connection("test_node_2", "test_node_3", 0.3)

        # Test expansion
        neighbors = self.memory.neural_mesh.get_neighbors("test_node_1", top_k=5)
        results["expansion_tests"] = len(neighbors)

        return results

    async def _test_concurrent_load(self) -> Dict[str, Any]:
        """Test concurrent query processing."""
        import asyncio

        async def concurrent_query(query_num: int):
            query = f"Test query {query_num} about AI and machine learning"
            start_time = time.time()
            result = await self.orchestrator.process_query(query, session_id=f"concurrent_{query_num}")
            end_time = time.time()
            return {
                "query_num": query_num,
                "processing_time": end_time - start_time,
                "success": bool(result.get("response"))
            }

        # Run 10 concurrent queries
        tasks = [concurrent_query(i) for i in range(10)]
        concurrent_results = await asyncio.gather(*tasks)

        results = {
            "concurrent_queries": len(concurrent_results),
            "successful_queries": sum(1 for r in concurrent_results if r["success"]),
            "avg_processing_time": np.mean([r["processing_time"] for r in concurrent_results]),
            "max_processing_time": max(r["processing_time"] for r in concurrent_results),
            "min_processing_time": min(r["processing_time"] for r in concurrent_results)
        }

        return results

    def _calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall test metrics."""
        metrics = {
            "total_test_duration": 0,
            "overall_throughput": 0,
            "memory_efficiency": 0,
            "query_success_rate": 0
        }

        # Calculate total duration
        phase_durations = []
        for phase_name, phase_results in results["test_phases"].items():
            if "total_time" in phase_results:
                phase_durations.append(phase_results["total_time"])
            elif "avg_processing_time" in phase_results:
                # Estimate for query phases
                phase_durations.append(phase_results["avg_processing_time"] * phase_results.get("queries_tested", 1))

        metrics["total_test_duration"] = sum(phase_durations)

        # Calculate throughput
        ingestion_phase = results["test_phases"].get("memory_ingestion", {})
        if ingestion_phase.get("throughput"):
            metrics["overall_throughput"] = ingestion_phase["throughput"]

        # Calculate query success rate
        query_phase = results["test_phases"].get("query_processing", {})
        if query_phase.get("responses_generated", 0) > 0:
            metrics["query_success_rate"] = query_phase["responses_generated"] / query_phase["queries_tested"]

        # Memory efficiency (simplified)
        memory_stats = results["test_phases"].get("memory_ingestion", {}).get("memory_stats", {})
        total_nodes = memory_stats.get("total_nodes", 0)
        if total_nodes > 0:
            metrics["memory_efficiency"] = total_nodes / (metrics["total_test_duration"] or 1)

        return metrics

    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Test results saved to {filepath}")

    async def run_performance_benchmark(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Run focused performance benchmark.
        """
        logger.info(f"Running performance benchmark with {iterations} iterations...")

        benchmark_results = {
            "iterations": iterations,
            "query_times": [],
            "memory_usage": [],
            "throughput": 0
        }

        start_time = time.time()

        for i in range(iterations):
            query = f"Benchmark query {i}: performance testing AI system capabilities"

            query_start = time.time()
            result = await self.orchestrator.process_query(query, session_id="benchmark")
            query_time = time.time() - query_start

            benchmark_results["query_times"].append(query_time)

            if (i + 1) % 20 == 0:
                logger.info(f"Benchmark iteration {i + 1}/{iterations}...")

        total_time = time.time() - start_time
        benchmark_results["total_time"] = total_time
        benchmark_results["throughput"] = iterations / total_time
        benchmark_results["avg_query_time"] = np.mean(benchmark_results["query_times"])
        benchmark_results["p95_query_time"] = np.percentile(benchmark_results["query_times"], 95)
        benchmark_results["p99_query_time"] = np.percentile(benchmark_results["query_times"], 99)

        # Save benchmark results
        benchmark_file = os.path.join(self.results_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, default=str)

        logger.info(f"Performance benchmark completed. Results saved to {benchmark_file}")
        return benchmark_results