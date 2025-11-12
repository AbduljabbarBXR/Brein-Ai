"""
Test script for the Brein AI prompt engineering system.
Tests PromptManager functionality and prompt loading/integration.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_manager import PromptManager
from sal import SystemAwarenessLayer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_prompt_manager_initialization():
    """Test that PromptManager initializes correctly"""
    print("Testing PromptManager initialization...")

    # Initialize SAL
    sal = SystemAwarenessLayer()
    await sal.initialize()

    # Initialize PromptManager
    prompt_manager = PromptManager(prompts_dir="backend/prompts", sal=sal)
    await prompt_manager.set_sal(sal)

    # Check that prompts were loaded
    available_prompts = prompt_manager.list_available_prompts()
    print(f"Loaded {len(available_prompts)} prompt categories: {available_prompts}")

    assert len(available_prompts) > 0, "No prompts were loaded"
    assert "core.identity" in available_prompts, "Identity prompts not loaded"
    assert "core.awareness" in available_prompts, "Awareness prompts not loaded"

    print("✓ PromptManager initialization test passed")
    return prompt_manager

async def run_all_tests_async():
    """Run all prompt system tests asynchronously"""
    print("🚀 Starting Brein AI Prompt System Tests")
    print("=" * 50)

    try:
        # Test initialization
        prompt_manager = await test_prompt_manager_initialization()

        # Test prompt categories
        test_identity_prompts(prompt_manager)
        test_agent_prompts(prompt_manager)
        test_system_prompts(prompt_manager)

        # Test functionality
        test_prompt_substitution(prompt_manager)
        test_no_model_disclosure(prompt_manager)

        print("\n" + "=" * 50)
        print("🎉 All prompt system tests passed!")
        print("✓ PromptManager working correctly")
        print("✓ All prompts loaded and accessible")
        print("✓ Identity and awareness properly defined")
        print("✓ No model implementation details leaked")
        print("✓ Variable substitution working")
        print("✓ SAL integration ready")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_identity_prompts(prompt_manager):
    """Test identity and self-awareness prompts"""
    print("\nTesting identity prompts...")

    # Test who_am_i prompt
    who_am_i = prompt_manager.get_prompt("core.identity.who_am_i")
    print(f"Who am I: {who_am_i[:100]}...")
    assert "Brein AI" in who_am_i, "Identity prompt doesn't mention Brein AI"
    assert "brain-inspired" in who_am_i, "Identity prompt doesn't mention brain-inspired architecture"

    # Test architecture overview
    architecture = prompt_manager.get_prompt("core.identity.architecture_overview")
    print(f"Architecture: {architecture[:100]}...")
    assert "Hippocampus" in architecture, "Architecture prompt missing Hippocampus"
    assert "Prefrontal Cortex" in architecture, "Architecture prompt missing Prefrontal Cortex"

    # Test awareness description
    awareness = prompt_manager.get_prompt("core.identity.awareness_description")
    print(f"Awareness: {awareness[:100]}...")
    assert "System Awareness Layer" in awareness, "Awareness prompt missing SAL"
    assert "self-awareness" in awareness, "Awareness prompt missing self-awareness"

    print("✓ Identity prompts test passed")

def test_agent_prompts(prompt_manager):
    """Test agent-specific prompts"""
    print("\nTesting agent prompts...")

    # Test hippocampus prompts
    summary_prompt = prompt_manager.get_prompt("agents.hippocampus.content_summary", content="Test content")
    print(f"Hippocampus summary prompt: {summary_prompt[:50]}...")
    assert "Summarize" in summary_prompt, "Hippocampus summary prompt incorrect"

    # Test prefrontal cortex prompts
    reasoning_prompt = prompt_manager.get_prompt("agents.prefrontal_cortex.complex_reasoning",
                                                query="Test query", context="Test context")
    print(f"Prefrontal cortex reasoning prompt: {reasoning_prompt[:50]}...")
    assert "Analyze this query" in reasoning_prompt, "Prefrontal cortex reasoning prompt incorrect"

    # Test amygdala prompts
    personality_prompt = prompt_manager.get_prompt("agents.amygdala.personality_response",
                                                  query="Hello", emotional_context="neutral",
                                                  relevant_memories="")
    print(f"Amygdala personality prompt: {personality_prompt[:50]}...")
    assert "helpful, empathetic" in personality_prompt, "Amygdala personality prompt incorrect"

    print("✓ Agent prompts test passed")

def test_system_prompts(prompt_manager):
    """Test system-level prompts"""
    print("\nTesting system prompts...")

    # Test chat management prompts
    title_prompt = prompt_manager.get_prompt("system.chat_management.smart_title_generation",
                                           first_query="How does AI work?")
    print(f"Chat title prompt: {title_prompt[:50]}...")
    assert "Generate a very short" in title_prompt, "Chat title prompt incorrect"

    # Test orchestrator prompts
    reasoning_prompt = prompt_manager.get_prompt("system.orchestrator.internal_reasoning", query="Test")
    print(f"Orchestrator reasoning prompt: {reasoning_prompt[:50]}...")
    assert "Think step by step" in reasoning_prompt, "Orchestrator reasoning prompt incorrect"

    print("✓ System prompts test passed")

def test_prompt_substitution(prompt_manager):
    """Test variable substitution in prompts"""
    print("\nTesting prompt variable substitution...")

    # Test with variables
    error_prompt = prompt_manager.get_prompt("agents.hippocampus.memory_ingestion_error", error="Test error")
    print(f"Error prompt with substitution: {error_prompt}")
    assert "Test error" in error_prompt, "Variable substitution failed"

    # Test complex substitution
    thought_trace = prompt_manager.get_prompt("system.orchestrator.thought_trace_complex",
                                            analysis="Test analysis",
                                            reasoning_steps="Test steps",
                                            complexity_score=0.85)
    print(f"Complex thought trace: {thought_trace[:100]}...")
    assert "Test analysis" in thought_trace, "Complex substitution failed"
    assert "0.85" in thought_trace, "Numeric substitution failed"

    print("✓ Prompt substitution test passed")

def test_no_model_disclosure(prompt_manager):
    """Test that prompts don't disclose model information"""
    print("\nTesting model disclosure prevention...")

    # Check that identity prompts don't mention specific models
    identity_prompts = ["who_am_i", "architecture_overview", "awareness_description", "capabilities_summary"]

    for prompt_key in identity_prompts:
        prompt = prompt_manager.get_prompt(f"core.identity.{prompt_key}")
        # Should not contain specific model names (be more specific to avoid false positives)
        forbidden_terms = ["llama-3.2", "phi-3.1", "tinyllama", "gguf", "gpt", "transformer model", "neural network model"]
        prompt_lower = prompt.lower()
        for term in forbidden_terms:
            assert term not in prompt_lower, f"Prompt {prompt_key} contains forbidden term: {term}"

        # Should not contain generic "model" in model context
        assert "using model" not in prompt_lower, f"Prompt {prompt_key} mentions using models"
        assert "powered by" not in prompt_lower, f"Prompt {prompt_key} mentions being powered by models"

    print("✓ Model disclosure prevention test passed")

if __name__ == "__main__":
    success = asyncio.run(run_all_tests_async())
    sys.exit(0 if success else 1)
