# Testing Guide

This comprehensive testing guide covers all aspects of testing Brein AI, from unit tests to integration testing and performance benchmarking.

## 🧪 Testing Overview

### Testing Pyramid
```
End-to-End Tests (E2E)     ┌─────────────┐  Few tests, high confidence
Integration Tests         ┌─┴─────────────┴┐  Medium tests, good coverage
Unit Tests               ┌─┴─────────────┴┐  Many tests, fast feedback
                       Code
```

### Test Categories
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete user workflows
- **Performance Tests**: Test speed, scalability, and resource usage
- **Security Tests**: Test vulnerability resistance

## 🛠️ Testing Setup

### Development Dependencies
```bash
# Install testing dependencies
pip install -r requirements-dev.txt

# Core testing libraries
pytest>=7.0.0          # Test framework
pytest-cov>=4.0.0      # Coverage reporting
pytest-asyncio>=0.21.0 # Async test support
pytest-mock>=3.10.0    # Mocking utilities
hypothesis>=6.0.0      # Property-based testing
```

### Test Directory Structure
```
tests/
├── __init__.py
├── conftest.py              # Test configuration and fixtures
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_memory.py
│   ├── test_api.py
│   └── test_utils.py
├── integration/             # Integration tests
│   ├── __init__.py
│   ├── test_agent_communication.py
│   ├── test_memory_pipeline.py
│   └── test_api_endpoints.py
├── e2e/                     # End-to-end tests
│   ├── __init__.py
│   ├── test_user_workflows.py
│   └── test_system_integration.py
├── performance/             # Performance tests
│   ├── __init__.py
│   ├── test_query_performance.py
│   ├── test_memory_scaling.py
│   └── test_concurrent_users.py
├── security/                # Security tests
│   ├── __init__.py
│   ├── test_input_validation.py
│   ├── test_authentication.py
│   └── test_access_control.py
├── fixtures/                # Test data and fixtures
│   ├── sample_queries.json
│   ├── mock_responses.json
│   └── test_database.db
└── utils/                   # Test utilities
    ├── __init__.py
    ├── test_helpers.py
    └── performance_utils.py
```

### Test Configuration
```python
# conftest.py
import pytest
import asyncio
from typing import AsyncGenerator
from backend.main import app
from backend.memory_manager import MemoryManager

@pytest.fixture
async def memory_manager() -> AsyncGenerator[MemoryManager, None]:
    """Provide a test memory manager instance."""
    manager = MemoryManager()
    await manager.initialize()
    yield manager
    await manager.cleanup()

@pytest.fixture
async def test_client():
    """Provide a test client for API testing."""
    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client

@pytest.fixture
def sample_queries():
    """Provide sample test queries."""
    return [
        "What is machine learning?",
        "Explain quantum computing",
        "Write a Python function to calculate fibonacci",
        "What are the benefits of renewable energy?"
    ]

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

## 🧪 Writing Unit Tests

### Basic Unit Test Structure
```python
import pytest
from unittest.mock import Mock, AsyncMock
from backend.agents.hippocampus import HippocampusAgent

class TestHippocampusAgent:
    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        return HippocampusAgent()

    @pytest.fixture
    def mock_memory_manager(self):
        """Mock memory manager for testing."""
        mock = Mock()
        mock.store.return_value = "test_doc_id"
        mock.retrieve.return_value = [{"content": "test", "similarity": 0.9}]
        return mock

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.model_name == "llama-3.2-3b"
        assert agent.max_tokens == 2048
        assert agent.temperature == 0.7

    def test_process_valid_query(self, agent):
        """Test processing a valid query."""
        query = "What is machine learning?"
        result = agent.process(query)

        assert "response" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_process_empty_query(self, agent):
        """Test handling of empty queries."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            agent.process("")

    @pytest.mark.asyncio
    async def test_memory_integration(self, agent, mock_memory_manager):
        """Test integration with memory system."""
        agent.memory_manager = mock_memory_manager

        query = "What do you remember about Python?"
        result = await agent.process_with_memory(query)

        mock_memory_manager.retrieve.assert_called_once()
        assert "response" in result

    @pytest.mark.parametrize("query,expected_min_confidence", [
        ("Simple question", 0.5),
        ("Complex technical question", 0.3),
        ("Ambiguous query", 0.1),
    ])
    def test_confidence_scoring(self, agent, query, expected_min_confidence):
        """Test confidence scoring for different query types."""
        result = agent.process(query)
        assert result["confidence"] >= expected_min_confidence
```

### Testing Async Functions
```python
import pytest
from unittest.mock import AsyncMock, patch

class TestAsyncAgent:
    @pytest.mark.asyncio
    async def test_async_processing(self):
        """Test async processing functionality."""
        agent = AsyncAgent()

        # Mock async dependencies
        with patch('backend.external_api.call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"result": "success"}

            result = await agent.process_async("test query")

            assert result["status"] == "success"
            mock_call.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling for long-running operations."""
        agent = AsyncAgent()

        with patch('backend.external_api.call', new_callable=AsyncMock) as mock_call:
            # Simulate timeout
            mock_call.side_effect = asyncio.TimeoutError()

            with pytest.raises(asyncio.TimeoutError):
                await agent.process_with_timeout("slow query", timeout=1.0)
```

### Property-Based Testing
```python
import pytest
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule

class TestQueryProcessor:
    @given(
        query=st.text(min_size=1, max_size=1000),
        temperature=st.floats(min_value=0.0, max_value=2.0),
        max_tokens=st.integers(min_value=1, max_value=4096)
    )
    def test_query_processing_properties(self, query, temperature, max_tokens):
        """Property-based test for query processing."""
        processor = QueryProcessor()

        result = processor.process(
            query=query,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Properties that should always hold
        assert isinstance(result, dict)
        assert "response" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["response"]) <= max_tokens

class QueryProcessorStateMachine(RuleBasedStateMachine):
    """Stateful testing for query processor."""

    def __init__(self):
        super().__init__()
        self.processor = QueryProcessor()
        self.memory = []

    @rule(query=st.text(min_size=1, max_size=500))
    def process_query(self, query):
        """Process a query and store result."""
        result = self.processor.process(query)
        self.memory.append(result)

    @rule()
    def check_memory_consistency(self):
        """Check that memory operations are consistent."""
        if self.memory:
            # Verify all results have required fields
            for result in self.memory:
                assert "response" in result
                assert "confidence" in result

TestStateMachine = QueryProcessorStateMachine.TestCase
```

## 🔗 Integration Testing

### API Integration Tests
```python
import pytest
from httpx import AsyncClient

class TestAPIIntegration:
    @pytest.mark.asyncio
    async def test_query_endpoint_integration(self, test_client: AsyncClient):
        """Test complete query processing through API."""
        request_data = {
            "query": "What is the capital of France?",
            "session_id": "test_session_123"
        }

        response = await test_client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "response" in data
        assert "session_id" in data
        assert "processing_time" in data
        assert "agents_used" in data
        assert "confidence_score" in data

        # Verify session continuity
        assert data["session_id"] == "test_session_123"

    @pytest.mark.asyncio
    async def test_memory_operations_integration(self, test_client: AsyncClient):
        """Test memory operations through API."""
        # Store content
        store_response = await test_client.post("/api/memory/ingest", json={
            "content": "Paris is the capital of France",
            "metadata": {"source": "test"}
        })
        assert store_response.status_code == 200
        doc_id = store_response.json()["document_id"]

        # Retrieve content
        search_response = await test_client.get("/api/memory/search?q=capital")
        assert search_response.status_code == 200

        results = search_response.json()["results"]
        assert len(results) > 0
        assert any("Paris" in result["content"] for result in results)

    @pytest.mark.asyncio
    async def test_agent_coordination_integration(self, test_client: AsyncClient):
        """Test multi-agent coordination through API."""
        response = await test_client.post("/api/query", json={
            "query": "Explain machine learning and create a simple Python example",
            "enable_web_access": False
        })

        assert response.status_code == 200
        data = response.json()

        # Should involve multiple agents
        agents_used = data["agents_used"]
        assert len(agents_used) >= 2  # At least reasoning + memory
        assert "prefrontal_cortex" in agents_used  # For reasoning
        assert any(agent in ["hippocampus", "amygdala"] for agent in agents_used)
```

### Database Integration Tests
```python
import pytest
import aiosqlite

class TestDatabaseIntegration:
    @pytest.fixture
    async def db_connection(self):
        """Provide a test database connection."""
        conn = await aiosqlite.connect(":memory:")
        # Set up test schema
        await conn.execute("""
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()
        yield conn
        await conn.close()

    @pytest.mark.asyncio
    async def test_document_storage_and_retrieval(self, db_connection):
        """Test complete document storage and retrieval cycle."""
        # Store document
        doc_id = "test_doc_123"
        content = "This is a test document about machine learning."

        await db_connection.execute(
            "INSERT INTO documents (id, content) VALUES (?, ?)",
            (doc_id, content)
        )
        await db_connection.commit()

        # Retrieve document
        cursor = await db_connection.execute(
            "SELECT content FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = await cursor.fetchone()

        assert result is not None
        assert result[0] == content

    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, db_connection):
        """Test concurrent database operations."""
        import asyncio

        async def store_document(doc_id: str, content: str):
            await db_connection.execute(
                "INSERT INTO documents (id, content) VALUES (?, ?)",
                (doc_id, content)
            )

        # Create multiple concurrent operations
        tasks = [
            store_document(f"doc_{i}", f"Content {i}")
            for i in range(10)
        ]

        await asyncio.gather(*tasks)
        await db_connection.commit()

        # Verify all documents were stored
        cursor = await db_connection.execute("SELECT COUNT(*) FROM documents")
        count = await cursor.fetchone()
        assert count[0] == 10
```

## 🌐 End-to-End Testing

### User Workflow Tests
```python
import pytest
from playwright.async_api import Page

class TestUserWorkflows:
    @pytest.mark.asyncio
    async def test_complete_user_journey(self, page: Page):
        """Test complete user interaction workflow."""
        # Navigate to application
        await page.goto("http://localhost:8000")

        # Wait for page to load
        await page.wait_for_selector("#chat-input")

        # Send a query
        await page.fill("#chat-input", "Hello, what can you do?")
        await page.click("#send-button")

        # Wait for response
        await page.wait_for_selector(".message.response")

        # Verify response appears
        response_element = page.locator(".message.response").last
        response_text = await response_element.text_content()
        assert len(response_text) > 0
        assert "brein" in response_text.lower() or "ai" in response_text.lower()

    @pytest.mark.asyncio
    async def test_memory_persistence(self, page: Page):
        """Test that memory persists across page reloads."""
        await page.goto("http://localhost:8000")

        # Teach the system something
        await page.fill("#chat-input", "Remember that I love hiking")
        await page.click("#send-button")
        await page.wait_for_selector(".message.response")

        # Reload page
        await page.reload()
        await page.wait_for_selector("#chat-input")

        # Ask about the remembered information
        await page.fill("#chat-input", "What do I love doing?")
        await page.click("#send-button")
        await page.wait_for_selector(".message.response")

        # Check if memory was retained
        response_element = page.locator(".message.response").last
        response_text = await response_element.text_content()
        assert "hiking" in response_text.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, page: Page):
        """Test error handling in UI."""
        await page.goto("http://localhost:8000")

        # Try to send empty message
        await page.click("#send-button")

        # Should show error message
        error_element = page.locator(".error-message")
        await error_element.wait_for()
        error_text = await error_element.text_content()
        assert "empty" in error_text.lower() or "message" in error_text.lower()
```

## ⚡ Performance Testing

### Benchmarking Tests
```python
import pytest
import time
import statistics
from typing import List, Dict

class TestPerformanceBenchmarking:
    def test_query_response_time(self, benchmark):
        """Benchmark query response time."""
        processor = QueryProcessor()

        def run_query():
            return processor.process("What is the meaning of life?")

        result = benchmark(run_query)

        # Assert performance requirements
        assert result["mean"] < 2.0  # Average < 2 seconds
        assert result["max"] < 5.0   # Max < 5 seconds

    def test_memory_operations_performance(self):
        """Test memory operation performance."""
        memory_manager = MemoryManager()

        # Test ingestion performance
        documents = [f"Document {i} content" for i in range(100)]

        start_time = time.time()
        for doc in documents:
            memory_manager.ingest(doc)
        ingestion_time = time.time() - start_time

        # Test retrieval performance
        start_time = time.time()
        results = memory_manager.search("Document")
        search_time = time.time() - start_time

        # Performance assertions
        assert ingestion_time < 30.0  # 30 seconds for 100 docs
        assert search_time < 1.0      # 1 second for search
        assert len(results) == 100    # All documents found

    def test_concurrent_user_load(self):
        """Test system performance under concurrent load."""
        import concurrent.futures
        import threading

        processor = QueryProcessor()
        results = []
        errors = []

        def process_query(query_id: int):
            try:
                start_time = time.time()
                result = processor.process(f"Query {query_id}: What is AI?")
                end_time = time.time()

                results.append({
                    "query_id": query_id,
                    "response_time": end_time - start_time,
                    "success": True
                })
            except Exception as e:
                errors.append({
                    "query_id": query_id,
                    "error": str(e)
                })

        # Run 50 concurrent queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_query, i) for i in range(50)]
            concurrent.futures.wait(futures)

        # Analyze results
        successful_queries = len(results)
        failed_queries = len(errors)
        response_times = [r["response_time"] for r in results]

        # Performance assertions
        assert successful_queries >= 45  # At least 90% success rate
        assert failed_queries <= 5
        assert statistics.mean(response_times) < 3.0  # Average < 3 seconds
        assert statistics.quantiles(response_times, n=20)[18] < 5.0  # P95 < 5 seconds
```

### Load Testing
```python
import locust
from locust import HttpUser, task, between

class BreinAIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def query_ai(self):
        """Simulate user querying the AI."""
        self.client.post("/api/query", json={
            "query": "Explain a complex topic simply",
            "session_id": f"user_{self.user_id}"
        })

    @task(1)
    def search_memory(self):
        """Simulate memory search."""
        self.client.get("/api/memory/search?q=machine+learning")

    @task(1)
    def check_health(self):
        """Simulate health checks."""
        self.client.get("/health")

# Run with: locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

## 🔒 Security Testing

### Input Validation Tests
```python
import pytest

class TestInputValidation:
    @pytest.mark.parametrize("malicious_input", [
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "UNION SELECT * FROM users--",
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
    ])
    def test_malicious_input_rejection(self, malicious_input):
        """Test that malicious inputs are properly rejected."""
        processor = QueryProcessor()

        # Should either sanitize or reject malicious input
        result = processor.process(malicious_input)

        # Response should not contain malicious content
        assert "<script>" not in result["response"]
        assert "alert(" not in result["response"]
        assert "../../../" not in result["response"]

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        # This should not execute any SQL
        dangerous_input = "'; DROP TABLE users; --"

        # Should not crash or expose database errors
        result = processor.process(dangerous_input)
        assert "error" not in result["response"].lower()
        assert "drop" not in result["response"].lower()

    def test_large_input_handling(self):
        """Test handling of very large inputs."""
        large_input = "A" * 100000  # 100KB input

        # Should handle gracefully without crashing
        result = processor.process(large_input)
        assert "response" in result
        assert len(result["response"]) < 10000  # Reasonable response length
```

### Authentication Tests
```python
class TestAuthentication:
    def test_api_key_validation(self):
        """Test API key authentication."""
        client = APIClient()

        # Valid key
        client.set_api_key("valid_key_123")
        response = client.query("test")
        assert response.status_code == 200

        # Invalid key
        client.set_api_key("invalid_key")
        response = client.query("test")
        assert response.status_code == 401

        # No key
        client.set_api_key(None)
        response = client.query("test")
        assert response.status_code == 401

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        client = APIClient(api_key="valid_key")

        # Make many requests quickly
        responses = []
        for i in range(150):  # Exceed rate limit
            response = client.query(f"Query {i}")
            responses.append(response.status_code)

        # Should have some rate limited responses
        rate_limited_count = responses.count(429)
        assert rate_limited_count > 0

        successful_count = responses.count(200)
        assert successful_count > 0
```

## 📊 Test Reporting and Coverage

### Coverage Configuration
```ini
# .coveragerc
[run]
source = backend
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod

[html]
directory = htmlcov
```

### Running Tests with Coverage
```bash
# Run tests with coverage
pytest --cov=backend --cov-report=html --cov-report=term-missing

# Generate coverage badge
coverage-badge -o coverage.svg

# View detailed HTML report
open htmlcov/index.html
```

### CI/CD Integration
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Run tests
      run: pytest --cov=backend --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 🎯 Best Practices

### Test Organization
1. **One Test Per Behavior**: Each test should verify one specific behavior
2. **Descriptive Names**: Test names should clearly indicate what they're testing
3. **Independent Tests**: Tests should not depend on each other
4. **Fast Tests**: Keep unit tests fast, integration tests reasonable
5. **Realistic Data**: Use realistic test data that reflects production usage

### Test-Driven Development
```python
# Red: Write failing test first
def test_user_registration():
    user_service = UserService()
    with pytest.raises(UserNotFoundError):
        user_service.get_user("nonexistent@example.com")

# Green: Implement minimal code to pass
class UserService:
    def get_user(self, email: str):
        # Minimal implementation
        raise UserNotFoundError(f"User {email} not found")

# Refactor: Improve implementation while keeping tests passing
class UserService:
    def __init__(self):
        self.users = {}

    def register_user(self, email: str, name: str):
        if email in self.users:
            raise UserAlreadyExistsError()
        self.users[email] = {"name": name, "email": email}

    def get_user(self, email: str):
        if email not in self.users:
            raise UserNotFoundError(f"User {email} not found")
        return self.users[email]
```

### Continuous Testing
- **Pre-commit Hooks**: Run tests before commits
- **CI/CD Pipeline**: Automated testing on every push
- **Nightly Builds**: Comprehensive testing overnight
- **Performance Regression**: Automated performance monitoring

## 📞 Support and Resources

### Testing Resources
- [[Contributing|Contributing]] - Development and testing guidelines
- [[API Reference|API-Reference]] - API testing examples
- [[Troubleshooting|Troubleshooting]] - Debugging test failures

### Getting Help
- **Test Failures**: Check CI/CD logs for detailed error messages
- **Coverage Issues**: Use coverage reports to identify untested code
- **Performance Problems**: Profile code to identify bottlenecks

---

*Testing Guide - Last updated: November 2025*
