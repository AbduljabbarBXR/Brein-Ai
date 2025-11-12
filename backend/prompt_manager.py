"""
Prompt Manager for Brein AI - Centralized prompt engineering control center.
Manages externalized prompts with SAL integration for dynamic optimization.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Centralized prompt management system with SAL integration.
    Handles prompt loading, versioning, validation, and optimization.
    """

    def __init__(self, prompts_dir: str = "prompts", sal=None):
        self.prompts_dir = prompts_dir
        self.sal = sal  # System Awareness Layer reference
        self.prompts = {}
        self.prompt_versions = {}
        self.prompt_metrics = {}

        # Load all prompts on initialization
        self._load_all_prompts()

        # SAL connection will be established later via set_sal()

    async def set_sal(self, sal):
        """Set SAL reference and establish connection"""
        self.sal = sal
        if self.sal:
            await self._connect_to_sal()

    async def _connect_to_sal(self):
        """Connect to System Awareness Layer for prompt optimization"""
        try:
            # Subscribe to relevant SAL events for prompt optimization
            await self.sal.event_bus.subscribe("prompt.performance_*", self.on_prompt_performance_update)
            await self.sal.event_bus.subscribe("sal.health_*", self.on_system_health_update)

            logger.info("PromptManager connected to SAL for optimization")
        except Exception as e:
            logger.warning(f"Failed to connect PromptManager to SAL: {e}")

    def _load_all_prompts(self):
        """Load all prompt files from the prompts directory"""
        try:
            # Load core prompts
            self._load_prompt_category("core")
            self._load_prompt_category("agents")
            self._load_prompt_category("system")
            self._load_prompt_category("config")

            logger.info("All prompt files loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            raise

    def _load_prompt_category(self, category: str):
        """Load all prompts from a specific category directory"""
        category_path = os.path.join(self.prompts_dir, category)

        if not os.path.exists(category_path):
            logger.warning(f"Prompt category directory not found: {category_path}")
            return

        for filename in os.listdir(category_path):
            if filename.endswith('.json'):
                prompt_name = filename[:-5]  # Remove .json extension
                # Skip config files as they're not prompts
                if category == "config":
                    logger.debug(f"Skipping config file: {category}.{prompt_name}")
                    continue
                self._load_prompt_file(category, prompt_name)

    def _load_prompt_file(self, category: str, prompt_name: str):
        """Load a specific prompt file"""
        file_path = os.path.join(self.prompts_dir, category, f"{prompt_name}.json")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)

            # Validate prompt structure
            self._validate_prompt_structure(prompt_data)

            # Store prompt with metadata
            self.prompts[f"{category}.{prompt_name}"] = {
                'data': prompt_data,
                'category': category,
                'name': prompt_name,
                'version': prompt_data.get('version', '1.0'),
                'last_modified': datetime.now().isoformat(),
                'file_path': file_path
            }

            # Track versions
            if 'version' in prompt_data:
                self.prompt_versions[f"{category}.{prompt_name}"] = prompt_data['version']

            logger.debug(f"Loaded prompt: {category}.{prompt_name} v{prompt_data.get('version', '1.0')}")

        except Exception as e:
            logger.error(f"Failed to load prompt {category}.{prompt_name}: {e}")
            raise

    def _validate_prompt_structure(self, prompt_data: Dict[str, Any]):
        """Validate that prompt data has required structure"""
        required_fields = ['version', 'description']
        for field in required_fields:
            if field not in prompt_data:
                raise ValueError(f"Prompt missing required field: {field}")

        # Validate that prompts have actual content
        if 'prompts' not in prompt_data and 'template' not in prompt_data:
            raise ValueError("Prompt must contain either 'prompts' or 'template' field")

    def get_prompt(self, prompt_key: str, variant: str = "default", **kwargs) -> str:
        """
        Get a formatted prompt by key with optional variable substitution.

        Args:
            prompt_key: Key in format "category.name" or "category.name.subkey"
            variant: Prompt variant to use
            **kwargs: Variables to substitute in the prompt

        Returns:
            Formatted prompt string
        """
        try:
            # Parse prompt key
            key_parts = prompt_key.split('.')
            if len(key_parts) < 2:
                raise ValueError(f"Invalid prompt key format: {prompt_key}")

            category = key_parts[0]
            name = key_parts[1]
            subkey = '.'.join(key_parts[2:]) if len(key_parts) > 2 else None

            # Get prompt data
            prompt_info = self.prompts.get(f"{category}.{name}")
            if not prompt_info:
                raise ValueError(f"Prompt not found: {category}.{name}")

            prompt_data = prompt_info['data']

            # Get the specific prompt
            prompt_text = self._extract_prompt_text(prompt_data, variant, subkey)

            # Apply variable substitution
            if kwargs:
                prompt_text = self._substitute_variables(prompt_text, kwargs)

            # Track usage for SAL optimization
            self._track_prompt_usage(prompt_key, variant)

            return prompt_text

        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_key}: {e}")
            raise

    def _extract_prompt_text(self, prompt_data: Dict[str, Any], variant: str, subkey: Optional[str]) -> str:
        """Extract the actual prompt text from prompt data"""
        # Check for direct prompts field
        if 'prompts' in prompt_data:
            prompts = prompt_data['prompts']

            # If subkey specified, look for nested structure
            if subkey:
                if subkey in prompts:
                    return prompts[subkey]
                else:
                    raise ValueError(f"Subkey '{subkey}' not found in prompts")

            # Look for variant
            if variant in prompts:
                return prompts[variant]
            elif 'default' in prompts:
                return prompts['default']
            else:
                # Return first available prompt
                return next(iter(prompts.values()))

        # Check for template field
        elif 'template' in prompt_data:
            return prompt_data['template']

        else:
            raise ValueError("No prompt content found")

    def _substitute_variables(self, prompt_text: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in prompt text"""
        try:
            return prompt_text.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in prompt substitution: {e}")
            # Return prompt with placeholders intact
            return prompt_text

    def _track_prompt_usage(self, prompt_key: str, variant: str):
        """Track prompt usage for SAL optimization"""
        if prompt_key not in self.prompt_metrics:
            self.prompt_metrics[prompt_key] = {
                'usage_count': 0,
                'variants': {},
                'last_used': None
            }

        self.prompt_metrics[prompt_key]['usage_count'] += 1
        self.prompt_metrics[prompt_key]['last_used'] = datetime.now().isoformat()

        if variant not in self.prompt_metrics[prompt_key]['variants']:
            self.prompt_metrics[prompt_key]['variants'][variant] = 0
        self.prompt_metrics[prompt_key]['variants'][variant] += 1

    def on_prompt_performance_update(self, event: str, data: Dict[str, Any]):
        """Handle prompt performance updates from SAL"""
        try:
            prompt_key = data.get('prompt_key')
            performance_score = data.get('performance_score', 0.5)

            if prompt_key and prompt_key in self.prompt_metrics:
                self.prompt_metrics[prompt_key]['performance_score'] = performance_score

                # Trigger optimization if performance is low
                if performance_score < 0.6:
                    self._optimize_prompt(prompt_key)

            logger.debug(f"Updated performance for prompt {prompt_key}: {performance_score}")

        except Exception as e:
            logger.error(f"Failed to handle prompt performance update: {e}")

    def on_system_health_update(self, event: str, data: Dict[str, Any]):
        """Handle system health updates that may affect prompt selection"""
        try:
            health_status = data.get('overall_health', 'healthy')

            # Adjust prompt complexity based on system health
            if health_status == 'degraded':
                self.system_health_modifier = 'simplified'
            else:
                self.system_health_modifier = 'normal'

            logger.debug(f"System health updated: {health_status}")

        except Exception as e:
            logger.error(f"Failed to handle system health update: {e}")

    def _optimize_prompt(self, prompt_key: str):
        """Optimize a prompt based on performance data"""
        # This would implement A/B testing and prompt optimization logic
        # For now, just log the optimization trigger
        logger.info(f"Triggering optimization for prompt: {prompt_key}")

    def get_prompt_metrics(self) -> Dict[str, Any]:
        """Get usage metrics for all prompts"""
        return self.prompt_metrics.copy()

    def reload_prompts(self):
        """Reload all prompts from disk"""
        self.prompts.clear()
        self.prompt_versions.clear()
        self._load_all_prompts()
        logger.info("All prompts reloaded")

    def list_available_prompts(self) -> List[str]:
        """List all available prompt keys"""
        return list(self.prompts.keys())
