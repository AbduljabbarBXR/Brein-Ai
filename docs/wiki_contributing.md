# Contributing to Brein AI

Welcome! We're excited that you're interested in contributing to Brein AI. This guide covers everything you need to know to contribute effectively to the project.

## 🚀 Ways to Contribute

### Code Contributions
- **Bug Fixes**: Identify and fix issues in the codebase
- **New Features**: Implement new functionality following our architecture
- **Performance Improvements**: Optimize existing code for better performance
- **Security Enhancements**: Improve security measures and practices

### Non-Code Contributions
- **Documentation**: Improve docs, write tutorials, create examples
- **Testing**: Write tests, improve test coverage, find edge cases
- **Design**: UI/UX improvements, accessibility enhancements
- **Research**: Explore new AI techniques, benchmark approaches
- **Community**: Help users, moderate discussions, spread awareness

### Content Contributions
- **Educational Content**: Create learning materials and guides
- **Use Cases**: Document real-world applications and examples
- **Tutorials**: Step-by-step guides for specific tasks
- **Translations**: Help localize the project

## 🛠️ Development Setup

### Prerequisites
- **Python 3.8+**: Main development language
- **Git**: Version control
- **Docker** (optional): For containerized development
- **GPU** (optional): For model training and testing

### Local Development Environment
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/Brein-Ai.git
cd Brein-Ai

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # Windows: dev_env\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Install pre-commit hooks
pre-commit install

# Run initial tests
python -m pytest tests/ -v
```

### Development Workflow
```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Write tests for new functionality
# Update documentation if needed

# Run tests
python -m pytest tests/ -v

# Format code
black .
isort .

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request
```

## 📝 Code Style and Standards

### Python Code Style
We follow PEP 8 with some modifications:

```python
# Good: Clear, readable code
def process_user_query(query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Process a user query with optional context.

    Args:
        query: The user's question or request
        context: Optional context dictionary

    Returns:
        Dictionary containing response and metadata
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    # Process query with context
    result = self._analyze_query(query, context)
    response = self._generate_response(result)

    return {
        "response": response,
        "confidence": result.confidence,
        "processing_time": result.processing_time
    }

# Bad: Unclear, non-standard code
def proc(q,c=None):
    if not q:return{}
    r=self._anal(q,c)
    return{"resp":r,"conf":r.conf,"time":r.time}
```

### Key Standards
- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Google-style docstrings for all public functions
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Line Length**: 88 characters maximum (Black formatter default)
- **Imports**: Group imports (stdlib, third-party, local) with isort

### Code Quality Tools
```bash
# Format code
black .
isort .

# Lint code
flake8 .
mypy .  # Type checking

# Run tests with coverage
pytest --cov=backend --cov-report=html

# Security scanning
bandit -r backend/
safety check
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── fixtures/      # Test data and fixtures
└── conftest.py    # Test configuration
```

### Writing Tests
```python
import pytest
from unittest.mock import Mock, patch

class TestQueryProcessor:
    def setup_method(self):
        self.processor = QueryProcessor()
        self.sample_query = "What is machine learning?"

    def test_process_valid_query(self):
        """Test processing a valid query."""
        result = self.processor.process(self.sample_query)

        assert "response" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_process_empty_query_raises_error(self):
        """Test that empty queries raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            self.processor.process("")

    @patch('backend.agents.hippocampus.HippocampusAgent.query')
    def test_memory_agent_integration(self, mock_agent):
        """Test integration with memory agent."""
        mock_agent.return_value = {"response": "From memory"}

        result = self.processor.process(
            "What do you remember?",
            use_memory=True
        )

        mock_agent.assert_called_once()
        assert "From memory" in result["response"]

    @pytest.mark.parametrize("query,expected_contains", [
        ("Hello", "greeting"),
        ("Calculate 2+2", "mathematical"),
        ("Write a poem", "creative"),
    ])
    def test_query_classification(self, query, expected_contains):
        """Test query classification for different types."""
        category = self.processor.classify_query(query)
        assert expected_contains in category
```

### Test Coverage Requirements
- **Unit Tests**: 80% minimum coverage
- **Integration Tests**: All major workflows covered
- **Edge Cases**: Error conditions and boundary values
- **Performance Tests**: Response time benchmarks

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/unit/test_agents.py

# Run tests matching pattern
pytest -k "test_memory"

# Run tests in parallel
pytest -n auto

# Debug failing test
pytest --pdb tests/unit/test_query_processor.py::TestQueryProcessor::test_process_valid_query
```

## 📋 Pull Request Process

### Before Submitting
1. **Update Tests**: Ensure all tests pass and add new tests for new features
2. **Update Documentation**: Update docs for any user-facing changes
3. **Code Review**: Self-review your code before requesting review
4. **Branch Status**: Ensure your branch is up-to-date with main

### PR Template
```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] No breaking changes
- [ ] Ready for review

## Screenshots (if applicable)
Add screenshots of UI changes or test results.
```

### PR Review Process
1. **Automated Checks**: CI/CD runs tests, linting, and security scans
2. **Peer Review**: At least one maintainer reviews the code
3. **Testing**: Reviewer may request additional tests or changes
4. **Approval**: PR approved and merged by maintainer
5. **Deployment**: Changes automatically deployed to staging

## 🏗️ Architecture Guidelines

### Adding New Agents
```python
from backend.agents.base import BaseAgent
from backend.sal import SAL

class CustomAgent(BaseAgent):
    def __init__(self, sal: SAL):
        super().__init__(sal)
        self.specialization = "custom_functionality"
        self.model = self.load_model()

    async def process(self, message: AgentMessage) -> AgentResponse:
        """Process messages for custom functionality."""
        # Implement custom logic
        result = await self.model.generate(message.content)

        return AgentResponse(
            content=result,
            confidence=self.calculate_confidence(result),
            metadata={"agent": "custom"}
        )

    def calculate_confidence(self, result: str) -> float:
        """Calculate confidence score for response."""
        # Implement confidence calculation
        return 0.85
```

### Extending Memory System
```python
from backend.memory.base import MemoryBackend

class CustomMemoryBackend(MemoryBackend):
    def __init__(self, config: Dict):
        self.config = config
        self.connection = self.initialize_connection()

    async def store(self, content: str, metadata: Dict) -> str:
        """Store content and return unique identifier."""
        doc_id = self.generate_id()
        await self.connection.store(doc_id, content, metadata)
        return doc_id

    async def retrieve(self, query: str, limit: int = 10) -> List[Dict]:
        """Retrieve relevant content for query."""
        results = await self.connection.search(query, limit)
        return self.format_results(results)
```

### Adding API Endpoints
```python
from fastapi import APIRouter, HTTPException
from backend.schemas import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/custom-endpoint", response_model=QueryResponse)
async def custom_endpoint(request: QueryRequest):
    """Custom API endpoint."""
    try:
        # Validate request
        if not request.query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Process request
        result = await process_custom_query(request.query)

        return QueryResponse(
            response=result.response,
            confidence=result.confidence,
            processing_time=result.processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 🔒 Security Considerations

### Secure Coding Practices
- **Input Validation**: Always validate and sanitize inputs
- **Authentication**: Use secure authentication methods
- **Authorization**: Implement proper access controls
- **Data Protection**: Encrypt sensitive data at rest and in transit
- **Error Handling**: Don't expose internal system details in errors

### Security Testing
```bash
# Run security scans
bandit -r backend/
safety check

# Test for common vulnerabilities
python -m pytest tests/security/ -v

# Check dependencies for vulnerabilities
pip-audit
```

## 📊 Performance Optimization

### Profiling Code
```python
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    """Profile a function's performance."""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    return result

# Usage
result = profile_function(my_expensive_function, arg1, arg2)
```

### Memory Optimization
```python
# Use generators for large datasets
def process_large_dataset(data_stream):
    """Process large datasets efficiently."""
    for chunk in data_stream:
        processed_chunk = yield from process_chunk(chunk)
        yield processed_chunk

# Optimize data structures
from collections import deque

# Use deque for frequent appends/removals
recent_items = deque(maxlen=1000)

# Use sets for membership testing
unique_items = set()
```

## 📚 Documentation Standards

### Code Documentation
```python
def complex_function(param1: str, param2: int = 42) -> Dict[str, Any]:
    """Perform complex operation with multiple steps.

    This function implements a sophisticated algorithm that processes
    input data through several transformation stages. It's designed
    to handle various edge cases and provide robust error handling.

    Args:
        param1: Description of first parameter including type constraints
        param2: Description of second parameter with default value explanation

    Returns:
        Dictionary containing:
        - 'result': The main computation result
        - 'metadata': Additional processing information
        - 'confidence': Confidence score between 0.0 and 1.0

    Raises:
        ValueError: When input parameters are invalid
        RuntimeError: When computation cannot be completed

    Examples:
        >>> result = complex_function("input", 42)
        >>> print(result['result'])
        'processed_output'

    Note:
        This function may take several seconds for large inputs.
        Consider using async version for better performance.
    """
```

### API Documentation
```python
@router.post(
    "/advanced-query",
    summary="Process Advanced Query",
    description="""
    Process a complex query with multiple options and parameters.

    This endpoint supports:
    - Multi-turn conversations
    - Web content integration
    - Custom agent selection
    - Advanced filtering options
    """,
    response_model=AdvancedQueryResponse,
    responses={
        200: {"description": "Query processed successfully"},
        400: {"description": "Invalid query parameters"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
```

## 🎯 Best Practices

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Functions have proper type hints
- [ ] Public functions have docstrings
- [ ] Tests added for new functionality
- [ ] No hardcoded secrets or credentials
- [ ] Error handling implemented
- [ ] Performance considerations addressed
- [ ] Security implications reviewed

### Git Workflow
```bash
# Keep branch updated
git fetch origin
git rebase origin/main

# Write clear commit messages
git commit -m "feat: add user authentication system

- Implement JWT token-based authentication
- Add user registration and login endpoints
- Include password hashing with bcrypt
- Add rate limiting for auth endpoints"

# Use meaningful branch names
git checkout -b feature/user-authentication
git checkout -b bugfix/memory-leak-fix
git checkout -b docs/api-documentation-update
```

## 🌟 Recognition

### Contributor Recognition
- **Contributors**: Listed in CONTRIBUTORS.md
- **Commit History**: All contributions tracked in git
- **Issues/PRs**: Recognition in GitHub interface
- **Community**: Featured in release notes

### Getting Help
- **GitHub Discussions**: Ask questions and get help
- **Issues**: Report bugs or request features
- **Discord/Slack**: Real-time community support
- **Documentation**: Comprehensive wiki and guides

## 📞 Contact Information

- **Project Maintainers**: abdijabarboxer2009@gmail.com
- **Community**: GitHub Discussions
- **Issues**: GitHub Issues
- **Security**: security@brein.ai (for security-related issues)

---

*Contributing Guide - Last updated: November 2025*
