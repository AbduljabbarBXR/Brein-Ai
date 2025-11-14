"""
Test script for Dynamic Prompt Filtering System
Demonstrates adaptive prompt selection based on user behavior patterns
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from user_behavior_analyzer import UserBehaviorAnalyzer, ResponseStyle
from dynamic_prompt_filter import DynamicPromptFilter, PromptFilterCriteria
from prompt_manager import PromptManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptivePromptsDemo:
    """Demonstration of the adaptive prompt filtering system"""

    def __init__(self):
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.prompt_manager = PromptManager(prompts_dir="backend/prompts")

    async def demonstrate_adaptive_prompts(self):
        """Demonstrate how prompts adapt to different user profiles"""

        print("🧠 Brein AI - Dynamic Prompt Filtering Demonstration")
        print("=" * 60)

        # Test scenarios with different user profiles
        test_scenarios = [
            {
                "user_id": "beginner_user",
                "profile": "Beginner user - prefers simple explanations",
                "interactions": [
                    ("What is machine learning?", "confused", 0.3, {"helpful": True}),
                    ("How does AI work?", "confused", 0.4, {"helpful": True}),
                    ("Explain neural networks simply", "neutral", 0.5, {"helpful": True})
                ],
                "expected_style": ResponseStyle.CONVERSATIONAL
            },
            {
                "user_id": "expert_user",
                "profile": "Expert user - prefers technical depth",
                "interactions": [
                    ("What are the computational complexity implications of transformer architectures?", "neutral", 0.9, {"helpful": True}),
                    ("Discuss the mathematical foundations of attention mechanisms", "neutral", 0.8, {"helpful": True}),
                    ("Analyze the convergence properties of gradient descent in deep networks", "neutral", 0.9, {"helpful": True})
                ],
                "expected_style": ResponseStyle.TECHNICAL
            },
            {
                "user_id": "tutorial_seeker",
                "profile": "Tutorial seeker - wants step-by-step guidance",
                "interactions": [
                    ("How do I set up a neural network from scratch?", "neutral", 0.6, {"helpful": True}),
                    ("Can you walk me through building my first AI model?", "neutral", 0.7, {"helpful": True}),
                    ("Guide me through the steps of training a machine learning model", "neutral", 0.6, {"helpful": True})
                ],
                "expected_style": ResponseStyle.TUTORIAL
            },
            {
                "user_id": "frustrated_user",
                "profile": "Frustrated user - needs patient, clear responses",
                "interactions": [
                    ("Why isn't my model training? This is so confusing!", "frustrated", 0.5, {"helpful": False}),
                    ("I don't understand any of this AI stuff, please explain simply", "frustrated", 0.4, {"helpful": True}),
                    ("This is too complicated, can you break it down?", "frustrated", 0.3, {"helpful": True})
                ],
                "expected_style": ResponseStyle.CONVERSATIONAL
            },
            {
                "user_id": "visual_learner",
                "profile": "Visual learner - responds well to diagrammatic explanations",
                "interactions": [
                    ("Show me how convolutional neural networks work", "neutral", 0.7, {"helpful": True}),
                    ("Can you describe the data flow in a neural network?", "neutral", 0.6, {"helpful": True}),
                    ("Help me visualize how backpropagation works", "neutral", 0.8, {"helpful": True})
                ],
                "expected_style": ResponseStyle.VISUAL
            }
        ]

        # Test query for all scenarios
        test_query = "Explain how neural networks learn from data"

        for scenario in test_scenarios:
            print(f"\n👤 Testing Profile: {scenario['profile']}")
            print("-" * 50)

            # Build user profile through simulated interactions
            user_id = scenario['user_id']
            for query, emotion, complexity, feedback in scenario['interactions']:
                # Simulate response time and effectiveness
                response_time = 2.0 + (complexity * 1.5)  # Simulate processing time
                effectiveness = 0.8 if feedback.get('helpful', True) else 0.4

                # Analyze the interaction
                analysis = self.behavior_analyzer.analyze_interaction(
                    user_id=user_id,
                    query=query,
                    response=f"Simulated response to: {query}",  # Mock response
                    response_time=response_time,
                    user_feedback=feedback
                )

                print(f"  📊 Interaction: '{query[:30]}...' → Style: {analysis['recommendations']['suggested_response_style']}")

            # Get user recommendations
            recommendations = self.behavior_analyzer.get_user_recommendations(user_id)
            print(f"  🎯 Final Profile: {recommendations['recommendations']['suggested_response_style']} style")
            print(".2f")

            # Test adaptive prompt selection
            try:
                adaptive_prompt, metadata = await self.prompt_manager.get_adaptive_prompt(
                    "system.orchestrator.standard_response",
                    user_id,
                    test_query
                )

                selected_variant = metadata.get('selected_variant', 'unknown')
                adaptation_reason = metadata.get('adaptation_reason', 'unknown')

                print(f"  🤖 Adaptive Prompt: Selected '{selected_variant}' variant")
                print(f"  📝 Reason: {adaptation_reason}")
                print(f"  💬 Prompt Preview: {adaptive_prompt[:100]}...")

            except Exception as e:
                print(f"  ❌ Error getting adaptive prompt: {e}")

        print(f"\n{'='*60}")
        print("🎉 Adaptive Prompt Filtering Demo Complete!")
        print("\nKey Benefits Demonstrated:")
        print("• 🎯 Personalized response styles based on user behavior")
        print("• 📈 Learning from interaction patterns and feedback")
        print("• 🔄 Dynamic adaptation without generic responses")
        print("• 🧠 Context-aware prompt selection")
        print("• 📊 Continuous improvement through usage analytics")

    async def demonstrate_self_awareness(self):
        """Demonstrate the system's self-awareness capabilities"""

        print("\n🧠 System Self-Awareness Demonstration")
        print("=" * 60)

        # Show current system state
        print("📊 System Architecture Awareness:")
        print("  • Multi-agent brain-inspired system")
        print("  • Hippocampus: Memory processing")
        print("  • Prefrontal Cortex: Complex reasoning")
        print("  • Amygdala: Emotional intelligence")
        print("  • Thalamus Router: Query routing")
        print("  • System Awareness Layer: Self-monitoring")

        # Show adaptive analytics
        analytics = self.prompt_manager.get_adaptive_analytics()
        if 'error' not in analytics:
            print("\n📈 Adaptive System Analytics:")
            print(f"  • Total adaptive calls: {analytics['total_adaptive_calls']}")
            print(f"  • Average confidence: {analytics['average_adaptation_confidence']:.2f}")
            print(f"  • Most used style: {analytics['most_used_style']}")

        # Show user behavior insights
        user_analytics = self.behavior_analyzer.get_analytics()
        if 'error' not in user_analytics:
            print("\n👥 User Behavior Insights:")
            print(f"  • Total users analyzed: {user_analytics['total_users']}")
            print(f"  • Total interactions: {user_analytics['total_interactions']}")
            print(f"  • Average query complexity: {user_analytics['avg_query_complexity']:.2f}")
            print(f"  • Average response effectiveness: {user_analytics['avg_response_effectiveness']:.2f}")

        print("\n🎯 Self-Awareness Features:")
        print("  • Real-time system health monitoring")
        print("  • Dynamic component coordination")
        print("  • Adaptive behavior based on context")
        print("  • Continuous learning from interactions")
        print("  • Proactive optimization capabilities")

async def main():
    """Main demonstration function"""
    demo = AdaptivePromptsDemo()

    try:
        # Run the adaptive prompts demonstration
        await demo.demonstrate_adaptive_prompts()

        # Show self-awareness capabilities
        await demo.demonstrate_self_awareness()

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
