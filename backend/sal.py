"""
System Awareness Layer (SAL) for Brein AI
Central nervous system enabling inter-brain communication and coordination
"""

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

import psutil

logger = logging.getLogger(__name__)


class SystemAwarenessLayer:
    """
    Central nervous system for Brein AI - coordinates all brain activities
    and enables inter-agent communication and self-awareness
    """

    def __init__(self):
        self.brain_state = BrainStateManager()
        self.event_bus = AsyncEventBus()
        self.message_router = MessageRouter()
        self.system_monitor = SystemMonitor()
        self.coordination_engine = CoordinationEngine()

        self.is_initialized = False
        self.brain_components = [
            'hippocampus',
            'prefrontal_cortex',
            'amygdala',
            'thalamus_router'
        ]

        # Setup error recovery patterns synchronously
        self._setup_error_recovery_patterns_sync()

    def _setup_error_recovery_patterns_sync(self):
        """Set up error recovery patterns synchronously for __init__"""
        # This is a placeholder - the actual async setup happens in initialize()
        pass

    async def initialize(self) -> bool:
        """
        Initialize SAL and establish brain communication channels

        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized:
            logger.info("SAL already initialized")
            return True

        try:
            logger.info("Initializing System Awareness Layer...")

            # Register brain components
            await self._register_brain_components()

            # Start system monitoring
            asyncio.create_task(self.system_monitor.start_monitoring())

            # Establish communication channels
            await self._establish_communication_channels()

            # Initialize coordination engine
            await self.coordination_engine.initialize()

            self.is_initialized = True
            logger.info("System Awareness Layer initialized successfully")

            # Broadcast initialization complete
            await self.event_bus.publish("sal.initialized", {
                'timestamp': datetime.now().isoformat(),
                'components_registered': len(self.brain_components)
            })

            return True

        except Exception as e:
            logger.error(f"SAL initialization failed: {e}")
            return False

    async def _register_brain_components(self):
        """Register all brain components with SAL"""
        for component in self.brain_components:
            await self.brain_state.register_component(component)
            logger.info(f"Registered brain component: {component}")

    async def _establish_communication_channels(self):
        """Establish communication channels between brain components"""
        # Set up standard communication patterns
        communication_patterns = [
            # Memory-Emotion coordination
            ("hippocampus", "amygdala", "emotional_memory"),
            ("amygdala", "hippocampus", "memory_prioritization"),

            # Reasoning coordination
            ("thalamus_router", "prefrontal_cortex", "complexity_assessment"),
            ("prefrontal_cortex", "thalamus_router", "reasoning_feedback"),

            # Cross-brain coordination
            ("prefrontal_cortex", "all", "reasoning_progress"),
            ("amygdala", "all", "emotional_context"),
            ("hippocampus", "all", "memory_update"),
        ]

        for sender, recipient, message_type in communication_patterns:
            await self.message_router.register_channel(sender, recipient, message_type)
            logger.debug(f"Established communication channel: {sender} → {recipient} ({message_type})")

    async def broadcast_message(self, sender: str, recipient: str, message_type: str, payload: dict):
        """
        Send messages between brain components

        Args:
            sender: Sending component name
            recipient: Receiving component name ("all" for broadcast)
            message_type: Type of message
            payload: Message data
        """
        message = {
            'id': str(uuid.uuid4()),
            'sender': sender,
            'recipient': recipient,
            'type': message_type,
            'payload': payload,
            'timestamp': datetime.now().isoformat()
        }

        # Update brain state for sender
        await self.brain_state.update_component_state(sender, {
            'last_activity': message['timestamp'],
            'active_message': message_type
        })

        # Route through message router
        await self.message_router.route_message(message)

        # Publish to event bus
        event_name = f"{recipient}.{message_type}" if recipient != "all" else f"broadcast.{message_type}"
        await self.event_bus.publish(event_name, message)

        logger.debug(f"Message broadcast: {sender} → {recipient} ({message_type})")

    async def query_brain_state(self, component: str = None) -> dict:
        """
        Query current brain state

        Args:
            component: Specific component to query, or None for full state

        Returns:
            dict: Brain state information
        """
        if component:
            return await self.brain_state.get_component_state(component)
        else:
            full_state = await self.brain_state.get_full_brain_state()
            # Add system metrics
            full_state['system'] = await self.system_monitor.get_current_metrics()
            return full_state

    async def coordinate_brain_activity(self, activity_type: str, context: dict, priority: str = "normal") -> dict:
        """
        Coordinate complex brain activities involving multiple components

        Args:
            activity_type: Type of activity to coordinate
            context: Activity context data
            priority: Coordination priority (low, normal, high, critical)

        Returns:
            dict: Coordination results
        """
        return await self.coordination_engine.coordinate(activity_type, context, priority)

    async def get_system_health(self) -> dict:
        """
        Get overall system health status

        Returns:
            dict: System health metrics
        """
        brain_health = await self.brain_state.get_health_status()
        system_health = await self.system_monitor.get_health_status()

        return {
            'overall_health': 'healthy' if brain_health['healthy'] and system_health['healthy'] else 'degraded',
            'brain_components': brain_health,
            'system_resources': system_health,
            'communication_status': await self.message_router.get_status(),
            'timestamp': datetime.now().isoformat()
        }

    async def shutdown(self):
        """Gracefully shutdown SAL"""
        logger.info("Shutting down System Awareness Layer...")

        # Stop system monitoring
        await self.system_monitor.stop_monitoring()

        # Save final state
        await self.brain_state.save_state()

        # Broadcast shutdown
        await self.event_bus.publish("sal.shutdown", {
            'timestamp': datetime.now().isoformat()
        })

        self.is_initialized = False
        logger.info("System Awareness Layer shutdown complete")


class BrainStateManager:
    """
    Maintains real-time state of all brain components
    """

    def __init__(self, state_file: str = "memory/brain_state.json"):
        self.state_file = state_file
        self.component_states = {}
        self.system_resources = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'active_models': [],
            'network_status': 'online'
        }
        self._lock = asyncio.Lock()

        # Ensure state directory exists
        os.makedirs(os.path.dirname(state_file), exist_ok=True)

    async def register_component(self, component_name: str):
        """Register a new brain component"""
        async with self._lock:
            if component_name not in self.component_states:
                self.component_states[component_name] = {
                    'status': 'registered',
                    'last_activity': None,
                    'active_tasks': [],
                    'performance_metrics': {},
                    'health_status': 'unknown',
                    'registered_at': datetime.now().isoformat()
                }
                logger.info(f"Brain component registered: {component_name}")

    async def update_component_state(self, component: str, state_updates: dict):
        """Update component state and notify dependents"""
        async with self._lock:
            if component in self.component_states:
                self.component_states[component].update(state_updates)
                self.component_states[component]['last_updated'] = datetime.now().isoformat()

                # Save state periodically
                if len(self.component_states) % 10 == 0:  # Every 10 updates
                    await self.save_state()

    async def get_component_state(self, component: str) -> dict:
        """Get state of specific component"""
        async with self._lock:
            return self.component_states.get(component, {}).copy()

    async def get_full_brain_state(self) -> dict:
        """Get complete brain state"""
        async with self._lock:
            return {
                'components': self.component_states.copy(),
                'system_resources': self.system_resources.copy(),
                'last_updated': datetime.now().isoformat()
            }

    async def get_health_status(self) -> dict:
        """Get brain health status"""
        async with self._lock:
            total_components = len(self.component_states)
            healthy_components = sum(
                1 for state in self.component_states.values()
                if state.get('health_status') == 'healthy'
            )

            return {
                'healthy': healthy_components == total_components,
                'total_components': total_components,
                'healthy_components': healthy_components,
                'component_health': {
                    name: state.get('health_status', 'unknown')
                    for name, state in self.component_states.items()
                }
            }

    async def save_state(self):
        """Save brain state to disk"""
        try:
            state_data = await self.get_full_brain_state()
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save brain state: {e}")

    async def load_state(self):
        """Load brain state from disk"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                    self.component_states = state_data.get('components', {})
                    self.system_resources = state_data.get('system_resources', {})
                logger.info("Brain state loaded from disk")
        except Exception as e:
            logger.error(f"Failed to load brain state: {e}")


class AsyncEventBus:
    """
    Event-driven communication system for brain components
    """

    def __init__(self, max_history: int = 1000):
        self.subscribers = defaultdict(list)
        self.event_history = deque(maxlen=max_history)
        self._lock = asyncio.Lock()

    async def subscribe(self, event_pattern: str, callback: Callable):
        """
        Subscribe to events matching a pattern

        Args:
            event_pattern: Event pattern (e.g., 'hippocampus.memory_*')
            callback: Async callback function
        """
        async with self._lock:
            self.subscribers[event_pattern].append(callback)
            logger.debug(f"Subscribed to event pattern: {event_pattern}")

    async def unsubscribe(self, event_pattern: str, callback: Callable):
        """Unsubscribe from events"""
        async with self._lock:
            if callback in self.subscribers[event_pattern]:
                self.subscribers[event_pattern].remove(callback)

    async def publish(self, event: str, data: dict):
        """
        Publish event to all matching subscribers

        Args:
            event: Event name
            data: Event data
        """
        # Add to history
        self.event_history.append({
            'event': event,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })

        # Find matching subscribers
        matching_callbacks = []
        async with self._lock:
            for pattern, callbacks in self.subscribers.items():
                if self._matches_pattern(event, pattern):
                    matching_callbacks.extend(callbacks)

        # Call subscribers asynchronously
        if matching_callbacks:
            tasks = [callback(event, data) for callback in matching_callbacks]
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.debug(f"Event published: {event} ({len(matching_callbacks)} subscribers)")

    def _matches_pattern(self, event: str, pattern: str) -> bool:
        """
        Check if event matches pattern (supports wildcards)

        Args:
            event: Event name
            pattern: Pattern with wildcards (*)

        Returns:
            bool: True if matches
        """
        if pattern == "*":
            return True

        if "*" in pattern:
            # Simple wildcard matching
            pattern_parts = pattern.split("*")
            if len(pattern_parts) == 2:
                prefix, suffix = pattern_parts
                return event.startswith(prefix) and event.endswith(suffix)

        return event == pattern

    async def get_event_history(self, limit: int = 100) -> List[dict]:
        """Get recent event history"""
        return list(self.event_history)[-limit:]


class MessageRouter:
    """
    Routes messages between brain components with intelligent routing logic
    """

    def __init__(self):
        self.channels = defaultdict(list)
        self.routing_rules = defaultdict(list)
        self.message_history = deque(maxlen=1000)
        self.message_stats = {
            'total_routed': 0,
            'by_type': defaultdict(int),
            'by_sender': defaultdict(int),
            'by_recipient': defaultdict(int),
            'errors': 0,
            'retries': 0,
            'timeouts': 0
        }
        self.delivery_confirmations = {}
        self.failed_messages = deque(maxlen=100)

    async def register_channel(self, sender: str, recipient: str, message_type: str):
        """Register a communication channel"""
        channel_key = f"{sender}:{recipient}"
        if message_type not in self.channels[channel_key]:
            self.channels[channel_key].append(message_type)
            logger.debug(f"Registered channel: {channel_key} ({message_type})")

    async def add_routing_rule(self, condition: callable, action: callable):
        """
        Add intelligent routing rules

        Args:
            condition: Function that returns True if rule applies
            action: Function that modifies routing behavior
        """
        self.routing_rules['custom'].append({'condition': condition, 'action': action})

    async def route_message(self, message: dict, retry_count: int = 0) -> bool:
        """
        Route a message to its destination with intelligent routing

        Args:
            message: Message dictionary
            retry_count: Number of retry attempts

        Returns:
            bool: True if successfully routed
        """
        try:
            sender = message['sender']
            recipient = message['recipient']
            message_type = message['type']
            message_id = message.get('id', str(uuid.uuid4()))

            # Store message in history
            self.message_history.append({
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'retry_count': retry_count
            })

            # Update statistics
            self.message_stats['total_routed'] += 1
            self.message_stats['by_type'][message_type] += 1
            self.message_stats['by_sender'][sender] += 1

            # Apply intelligent routing rules
            routing_decision = await self._apply_routing_rules(message)

            # Handle broadcast messages
            if recipient == "all" or routing_decision.get('broadcast', False):
                logger.debug(f"Broadcast message routed: {message_type}")
                return True

            # Handle intelligent recipient selection
            if routing_decision.get('override_recipient'):
                recipient = routing_decision['override_recipient']
                message['recipient'] = recipient

            # Validate channel exists
            channel_key = f"{sender}:{recipient}"
            if message_type not in self.channels[channel_key]:
                # Try to find alternative route
                alternative_recipient = await self._find_alternative_route(message)
                if alternative_recipient:
                    message['recipient'] = alternative_recipient
                    channel_key = f"{sender}:{alternative_recipient}"
                    logger.info(f"Rerouted message to alternative recipient: {alternative_recipient}")
                else:
                    logger.warning(f"No route found for {channel_key} ({message_type})")
                    await self._handle_routing_failure(message, retry_count)
                    return False

            # Set delivery timeout
            timeout_task = asyncio.create_task(self._set_delivery_timeout(message_id))

            # Track delivery confirmation
            self.delivery_confirmations[message_id] = {
                'message': message,
                'timeout_task': timeout_task,
                'delivered': False
            }

            logger.debug(f"Message routed: {sender} → {recipient} ({message_type})")
            return True

        except Exception as e:
            logger.error(f"Message routing error: {e}")
            self.message_stats['errors'] += 1
            await self._handle_routing_failure(message, retry_count)
            return False

    async def _apply_routing_rules(self, message: dict) -> dict:
        """
        Apply intelligent routing rules to message

        Returns:
            dict: Routing decisions
        """
        routing_decision = {}

        # Rule 1: High-priority emotional messages get broadcast
        if (message['type'] == 'emotional_context' and
            message['payload'].get('urgency_level') == 'high'):
            routing_decision['broadcast'] = True

        # Rule 2: Complex reasoning requests get routed to prefrontal cortex
        if (message['type'] == 'complexity_assessment' and
            message['payload'].get('query_complexity', 0) > 0.7):
            routing_decision['override_recipient'] = 'prefrontal_cortex'

        # Rule 3: Memory requests with emotional context get priority routing
        if (message['type'] == 'memory_boost' and
            message['payload'].get('confidence', 0) > 0.8):
            routing_decision['priority'] = 'high'

        # Apply custom routing rules
        for rule in self.routing_rules.get('custom', []):
            try:
                if await rule['condition'](message):
                    custom_decision = await rule['action'](message)
                    routing_decision.update(custom_decision)
            except Exception as e:
                logger.error(f"Custom routing rule error: {e}")

        return routing_decision

    async def _find_alternative_route(self, message: dict) -> Optional[str]:
        """
        Find alternative routing path when direct route fails

        Returns:
            Optional[str]: Alternative recipient or None
        """
        message_type = message['type']
        sender = message['sender']

        # Define fallback routing logic - these are the brain components that can handle each message type
        fallback_routes = {
            'memory_boost': ['hippocampus', 'prefrontal_cortex'],
            'emotional_context': ['amygdala', 'prefrontal_cortex'],
            'reasoning_progress': ['prefrontal_cortex', 'thalamus_router'],
            'resource_request': ['prefrontal_cortex', 'thalamus_router'],
            'complexity_assessment': ['prefrontal_cortex', 'thalamus_router']
        }

        alternatives = fallback_routes.get(message_type, [])

        # Return the first valid alternative (excluding the original sender to avoid loops)
        for alt_recipient in alternatives:
            if alt_recipient != sender:  # Don't route back to sender
                return alt_recipient

        return None

    async def _handle_routing_failure(self, message: dict, retry_count: int):
        """Handle message routing failures with retry logic"""
        message_id = message.get('id', str(uuid.uuid4()))

        if retry_count < 3:  # Max 3 retries
            delay = 2 ** retry_count  # Exponential backoff
            logger.info(f"Retrying message delivery in {delay}s (attempt {retry_count + 1})")

            self.message_stats['retries'] += 1
            self.failed_messages.append({
                'message': message,
                'failure_time': datetime.now().isoformat(),
                'retry_count': retry_count
            })

            # Schedule retry
            asyncio.get_event_loop().call_later(
                delay,
                lambda: asyncio.create_task(self.route_message(message, retry_count + 1))
            )
        else:
            logger.error(f"Message delivery failed after {retry_count} retries: {message}")
            self.message_stats['timeouts'] += 1

    async def _set_delivery_timeout(self, message_id: str):
        """Set delivery timeout for message tracking"""
        try:
            await asyncio.sleep(30)  # 30 second timeout

            if message_id in self.delivery_confirmations:
                confirmation = self.delivery_confirmations[message_id]
                if not confirmation['delivered']:
                    logger.warning(f"Message delivery timeout: {message_id}")
                    self.message_stats['timeouts'] += 1

                # Cleanup
                del self.delivery_confirmations[message_id]

        except asyncio.CancelledError:
            # Delivery confirmed, cancel timeout
            pass

    async def confirm_delivery(self, message_id: str):
        """Confirm successful message delivery"""
        if message_id in self.delivery_confirmations:
            self.delivery_confirmations[message_id]['delivered'] = True
            self.delivery_confirmations[message_id]['timeout_task'].cancel()

    async def get_status(self) -> dict:
        """Get comprehensive routing status"""
        return {
            'channels_registered': len(self.channels),
            'messages_routed': self.message_stats['total_routed'],
            'routing_errors': self.message_stats['errors'],
            'retries': self.message_stats['retries'],
            'timeouts': self.message_stats['timeouts'],
            'active_channels': dict(self.channels),
            'pending_deliveries': len(self.delivery_confirmations),
            'failed_messages_count': len(self.failed_messages),
            'message_type_distribution': dict(self.message_stats['by_type'])
        }

    async def get_routing_analytics(self) -> dict:
        """Get detailed routing analytics"""
        total_messages = self.message_stats['total_routed']
        if total_messages == 0:
            return {'error': 'No messages routed yet'}

        return {
            'total_messages': total_messages,
            'success_rate': (total_messages - self.message_stats['errors']) / total_messages,
            'retry_rate': self.message_stats['retries'] / total_messages,
            'timeout_rate': self.message_stats['timeouts'] / total_messages,
            'most_active_sender': max(self.message_stats['by_sender'], key=self.message_stats['by_sender'].get),
            'most_active_message_type': max(self.message_stats['by_type'], key=self.message_stats['by_type'].get),
            'channel_utilization': {
                channel: len(types) for channel, types in self.channels.items()
            }
        }


class SystemMonitor:
    """
    Monitors system resources and health
    """

    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.current_metrics = {}
        self.monitoring_task = None

    async def start_monitoring(self):
        """Start system monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        # Collect initial metrics synchronously
        self.current_metrics = await self._collect_metrics()
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")

    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self.current_metrics = await self._collect_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_metrics(self) -> dict:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Network status (simplified)
            network_status = "online"  # Could be enhanced with actual connectivity checks

            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'network_status': network_status,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0,
                'network_status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def get_current_metrics(self) -> dict:
        """Get current system metrics"""
        return self.current_metrics.copy()

    async def get_health_status(self) -> dict:
        """Get system health status"""
        metrics = await self.get_current_metrics()

        # Define health thresholds
        cpu_healthy = metrics.get('cpu_usage', 0) < 90
        memory_healthy = metrics.get('memory_usage', 0) < 90
        disk_healthy = metrics.get('disk_usage', 0) < 95

        return {
            'healthy': cpu_healthy and memory_healthy and disk_healthy,
            'cpu_healthy': cpu_healthy,
            'memory_healthy': memory_healthy,
            'disk_healthy': disk_healthy,
            'metrics': metrics
        }


class CoordinationEngine:
    """
    Advanced coordination engine for complex multi-agent brain activities
    with error recovery and adaptive coordination patterns
    """

    def __init__(self):
        self.active_coordinations = {}
        self.coordination_history = deque(maxlen=500)
        self.error_recovery_patterns = {}
        self.performance_metrics = defaultdict(int)

        # Enhanced coordination patterns
        self.coordination_patterns = {
            'complex_reasoning': self._coordinate_complex_reasoning,
            'emotional_processing': self._coordinate_emotional_processing,
            'memory_consolidation': self._coordinate_memory_consolidation,
            'learning_adaptation': self._coordinate_learning_adaptation,
            'crisis_response': self._coordinate_crisis_response,
            'creative_problem_solving': self._coordinate_creative_problem_solving,
            'multi_modal_processing': self._coordinate_multi_modal_processing
        }

        # Initialize error recovery patterns
        self._setup_error_recovery_patterns_sync()

    def _setup_error_recovery_patterns_sync(self):
        """Set up error recovery patterns synchronously for __init__"""
        # This is a placeholder - the actual async setup happens in initialize()
        pass

    async def initialize(self):
        """Initialize coordination engine with recovery patterns"""
        logger.info("Coordination engine initialized with advanced patterns")

        # Set up intelligent coordination rules
        await self._setup_coordination_rules()

    async def _setup_coordination_rules(self):
        """Set up intelligent coordination rules"""
        # Rule: High emotional intensity triggers amygdala-led coordination
        # Rule: Complex queries (>0.8 complexity) require full brain coordination
        # Rule: Memory-intensive tasks prioritize hippocampus availability
        # Rule: Time-critical responses use parallel processing
        pass

    async def _setup_error_recovery_patterns(self):
        """Set up error recovery patterns for different failure scenarios"""
        self.error_recovery_patterns = {
            'agent_unavailable': self._recover_agent_unavailable,
            'communication_failure': self._recover_communication_failure,
            'resource_exhaustion': self._recover_resource_exhaustion,
            'timeout_exceeded': self._recover_timeout_exceeded,
            'coordination_conflict': self._recover_coordination_conflict
        }

    async def coordinate(self, activity_type: str, context: dict, priority: str = "normal") -> dict:
        """
        Coordinate a complex brain activity with error recovery

        Args:
            activity_type: Type of activity to coordinate
            context: Activity context with requirements and constraints
            priority: Coordination priority (low, normal, high, critical)

        Returns:
            dict: Coordination results with status and metadata
        """
        if activity_type not in self.coordination_patterns:
            return {'error': f'Unknown coordination type: {activity_type}'}

        coordination_id = str(uuid.uuid4())
        start_time = datetime.now()

        coordination_record = {
            'id': coordination_id,
            'type': activity_type,
            'context': context,
            'priority': priority,
            'started_at': start_time.isoformat(),
            'status': 'running',
            'attempts': 0,
            'agents_involved': [],
            'performance_metrics': {}
        }

        self.active_coordinations[coordination_id] = coordination_record

        try:
            # Pre-coordination assessment
            assessment = await self._assess_coordination_requirements(activity_type, context)
            coordination_record.update(assessment)

            # Execute coordination with error recovery
            result = await self._execute_coordination_with_recovery(
                activity_type, context, coordination_record
            )

            # Post-coordination analysis
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            coordination_record.update({
                'status': 'completed',
                'completed_at': end_time.isoformat(),
                'duration': duration,
                'result': result,
                'performance_metrics': {
                    'duration': duration,
                    'agents_used': len(result.get('agents_involved', [])),
                    'success_rate': 1.0 if result.get('status') == 'success' else 0.0
                }
            })

            # Update performance metrics
            self.performance_metrics[activity_type] += 1
            self.coordination_history.append(coordination_record)

            return result

        except Exception as e:
            coordination_record.update({
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            })

            logger.error(f"Coordination failed: {activity_type} - {e}")

            # Attempt error recovery
            recovery_result = await self._attempt_error_recovery(activity_type, context, str(e))
            if recovery_result:
                coordination_record['recovery_attempted'] = True
                coordination_record['recovery_result'] = recovery_result
                return recovery_result

            return {'error': str(e), 'coordination_id': coordination_id}

    async def _assess_coordination_requirements(self, activity_type: str, context: dict) -> dict:
        """
        Assess resource and agent requirements for coordination

        Returns:
            dict: Assessment results
        """
        assessment = {
            'required_agents': [],
            'estimated_duration': 'unknown',
            'resource_requirements': {},
            'risk_level': 'low'
        }

        # Activity-specific assessments
        if activity_type == 'complex_reasoning':
            assessment.update({
                'required_agents': ['prefrontal_cortex', 'hippocampus'],
                'estimated_duration': 'moderate',
                'resource_requirements': {'cpu': 0.6, 'memory': 0.4},
                'risk_level': 'medium' if context.get('complexity', 0) > 0.8 else 'low'
            })

        elif activity_type == 'emotional_processing':
            assessment.update({
                'required_agents': ['amygdala', 'prefrontal_cortex'],
                'estimated_duration': 'short',
                'resource_requirements': {'cpu': 0.3, 'memory': 0.2},
                'risk_level': 'high' if context.get('urgency') == 'critical' else 'low'
            })

        elif activity_type == 'crisis_response':
            assessment.update({
                'required_agents': ['all'],
                'estimated_duration': 'immediate',
                'resource_requirements': {'cpu': 0.8, 'memory': 0.6},
                'risk_level': 'critical'
            })

        return assessment

    async def _execute_coordination_with_recovery(self, activity_type: str, context: dict,
                                                coordination_record: dict) -> dict:
        """
        Execute coordination with automatic error recovery

        Returns:
            dict: Coordination result
        """
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            coordination_record['attempts'] = attempt + 1

            try:
                result = await self.coordination_patterns[activity_type](context)

                # Validate result
                if self._validate_coordination_result(result):
                    return result
                else:
                    raise Exception("Invalid coordination result")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Coordination attempt {attempt + 1} failed: {e}")

                # Try recovery if not the last attempt
                if attempt < max_attempts - 1:
                    recovery_success = await self._attempt_mid_coordination_recovery(
                        activity_type, context, str(e)
                    )
                    if not recovery_success:
                        await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff

        # All attempts failed
        raise Exception(f"Coordination failed after {max_attempts} attempts. Last error: {last_error}")

    async def _attempt_error_recovery(self, activity_type: str, context: dict, error: str) -> Optional[dict]:
        """
        Attempt error recovery based on error type

        Returns:
            Optional[dict]: Recovery result or None if recovery failed
        """
        error_type = self._classify_error(error)

        if error_type in self.error_recovery_patterns:
            try:
                recovery_result = await self.error_recovery_patterns[error_type](
                    activity_type, context, error
                )
                if recovery_result:
                    logger.info(f"Error recovery successful for {error_type}")
                    return recovery_result
            except Exception as e:
                logger.error(f"Error recovery failed: {e}")

        return None

    async def _attempt_mid_coordination_recovery(self, activity_type: str, context: dict, error: str) -> bool:
        """
        Attempt recovery during coordination execution

        Returns:
            bool: True if recovery successful
        """
        # Implement recovery strategies like:
        # - Switch to backup agents
        # - Reduce coordination complexity
        # - Request additional resources
        # - Use cached results

        error_type = self._classify_error(error)

        if error_type == 'agent_unavailable':
            # Try alternative agent routing
            return await self._try_alternative_agent_routing(activity_type, context)

        elif error_type == 'resource_exhaustion':
            # Request resource reallocation
            return await self._request_resource_reallocation(context)

        return False

    def _classify_error(self, error: str) -> str:
        """Classify error type for appropriate recovery"""
        error_lower = error.lower()

        if 'unavailable' in error_lower or 'not found' in error_lower:
            return 'agent_unavailable'
        elif 'timeout' in error_lower:
            return 'timeout_exceeded'
        elif 'memory' in error_lower or 'cpu' in error_lower:
            return 'resource_exhaustion'
        elif 'communication' in error_lower or 'connection' in error_lower:
            return 'communication_failure'
        elif 'conflict' in error_lower:
            return 'coordination_conflict'
        else:
            return 'unknown_error'

    def _validate_coordination_result(self, result: dict) -> bool:
        """Validate coordination result structure"""
        required_fields = ['coordination_type', 'agents_involved', 'status']
        return all(field in result for field in required_fields)

    # Error Recovery Patterns
    async def _recover_agent_unavailable(self, activity_type: str, context: dict, error: str) -> Optional[dict]:
        """Recover from agent unavailability"""
        # Implement fallback coordination with available agents
        fallback_patterns = {
            'complex_reasoning': ['prefrontal_cortex'],  # Can work with just PFC
            'emotional_processing': ['prefrontal_cortex'],  # PFC can handle basic emotional processing
            'memory_consolidation': ['hippocampus']  # Direct memory operations
        }

        fallback_agents = fallback_patterns.get(activity_type, [])
        if fallback_agents:
            return {
                'coordination_type': f"{activity_type}_fallback",
                'agents_involved': fallback_agents,
                'status': 'recovered',
                'recovery_type': 'agent_fallback'
            }

        return None

    async def _recover_communication_failure(self, activity_type: str, context: dict, error: str) -> Optional[dict]:
        """Recover from communication failures"""
        # Retry with simplified communication
        return {
            'coordination_type': f"{activity_type}_simplified",
            'agents_involved': ['prefrontal_cortex'],  # Use most reliable agent
            'status': 'recovered',
            'recovery_type': 'simplified_communication'
        }

    async def _recover_resource_exhaustion(self, activity_type: str, context: dict, error: str) -> Optional[dict]:
        """Recover from resource exhaustion"""
        # Reduce coordination complexity
        simplified_context = context.copy()
        simplified_context['complexity'] = min(context.get('complexity', 0.5), 0.3)

        return {
            'coordination_type': f"{activity_type}_lightweight",
            'agents_involved': ['prefrontal_cortex'],
            'status': 'recovered',
            'recovery_type': 'resource_optimized',
            'simplified_context': simplified_context
        }

    async def _recover_timeout_exceeded(self, activity_type: str, context: dict, error: str) -> Optional[dict]:
        """Recover from timeouts"""
        # Use cached or partial results
        return {
            'coordination_type': f"{activity_type}_cached",
            'agents_involved': [],
            'status': 'recovered',
            'recovery_type': 'cached_result'
        }

    async def _recover_coordination_conflict(self, activity_type: str, context: dict, error: str) -> Optional[dict]:
        """Recover from coordination conflicts"""
        # Implement conflict resolution
        return {
            'coordination_type': f"{activity_type}_resolved",
            'agents_involved': ['prefrontal_cortex'],  # Use executive function for resolution
            'status': 'recovered',
            'recovery_type': 'conflict_resolved'
        }

    # Helper recovery methods
    async def _try_alternative_agent_routing(self, activity_type: str, context: dict) -> bool:
        """Try routing to alternative agents"""
        # Implementation would check agent availability and reroute
        return False  # Placeholder

    async def _request_resource_reallocation(self, context: dict) -> bool:
        """Request resource reallocation"""
        # Implementation would communicate with system monitor
        return False  # Placeholder

    # Enhanced Coordination Patterns
    async def _coordinate_complex_reasoning(self, context: dict) -> dict:
        """Coordinate complex reasoning with adaptive agent selection"""
        complexity = context.get('complexity', 0.5)
        emotional_context = context.get('emotional_context', False)

        agents = ['prefrontal_cortex']  # Always include PFC

        if complexity > 0.7:
            agents.extend(['hippocampus', 'amygdala'])
        elif emotional_context:
            agents.append('amygdala')
        else:
            agents.append('hippocampus')

        return {
            'coordination_type': 'complex_reasoning',
            'agents_involved': agents,
            'status': 'success',
            'coordination_strategy': 'adaptive',
            'complexity_handled': complexity,
            'emotional_integration': emotional_context
        }

    async def _coordinate_emotional_processing(self, context: dict) -> dict:
        """Coordinate emotional processing with intensity-based routing"""
        intensity = context.get('intensity', 0.5)
        urgency = context.get('urgency', 'normal')

        agents = ['amygdala']  # Primary emotional processor

        if intensity > 0.7 or urgency == 'high':
            agents.append('prefrontal_cortex')  # Involve executive function for high-intensity

        return {
            'coordination_type': 'emotional_processing',
            'agents_involved': agents,
            'status': 'success',
            'emotional_intensity': intensity,
            'response_urgency': urgency
        }

    async def _coordinate_memory_consolidation(self, context: dict) -> dict:
        """Coordinate memory consolidation with priority handling"""
        consolidation_type = context.get('type', 'standard')
        priority = context.get('priority', 'normal')

        agents = ['hippocampus']

        if consolidation_type == 'emotional':
            agents.append('amygdala')
        if priority == 'high':
            agents.append('prefrontal_cortex')

        return {
            'coordination_type': 'memory_consolidation',
            'agents_involved': agents,
            'status': 'success',
            'consolidation_type': consolidation_type,
            'processing_priority': priority
        }

    async def _coordinate_learning_adaptation(self, context: dict) -> dict:
        """Coordinate system learning and adaptation"""
        adaptation_focus = context.get('focus', 'general')

        return {
            'coordination_type': 'learning_adaptation',
            'agents_involved': ['all'],  # All agents participate in learning
            'status': 'success',
            'adaptation_focus': adaptation_focus,
            'learning_objective': context.get('objective', 'general_improvement')
        }

    async def _coordinate_crisis_response(self, context: dict) -> dict:
        """Coordinate crisis response with maximum resource allocation"""
        crisis_type = context.get('crisis_type', 'unknown')

        return {
            'coordination_type': 'crisis_response',
            'agents_involved': ['all'],
            'status': 'success',
            'crisis_type': crisis_type,
            'response_priority': 'critical',
            'resource_allocation': 'maximum'
        }

    async def _coordinate_creative_problem_solving(self, context: dict) -> dict:
        """Coordinate creative problem solving with divergent thinking"""
        problem_domain = context.get('domain', 'general')

        return {
            'coordination_type': 'creative_problem_solving',
            'agents_involved': ['prefrontal_cortex', 'hippocampus', 'amygdala'],
            'status': 'success',
            'problem_domain': problem_domain,
            'thinking_mode': 'divergent',
            'creativity_stimulation': True
        }

    async def _coordinate_multi_modal_processing(self, context: dict) -> dict:
        """Coordinate processing of multiple input modalities"""
        modalities = context.get('modalities', ['text'])

        agents = ['prefrontal_cortex']  # Integration hub

        if 'emotional' in modalities:
            agents.append('amygdala')
        if 'memory' in modalities:
            agents.append('hippocampus')

        return {
            'coordination_type': 'multi_modal_processing',
            'agents_involved': agents,
            'status': 'success',
            'modalities_processed': modalities,
            'integration_strategy': 'parallel_processing'
        }

    async def get_active_coordinations(self) -> dict:
        """Get status of active coordinations"""
        return self.active_coordinations.copy()

    async def get_coordination_analytics(self) -> dict:
        """Get coordination performance analytics"""
        total_coordinations = len(self.coordination_history)

        if total_coordinations == 0:
            return {'error': 'No coordination history available'}

        successful = sum(1 for c in self.coordination_history if c['status'] == 'completed')
        avg_duration = sum(c.get('duration', 0) for c in self.coordination_history) / total_coordinations

        return {
            'total_coordinations': total_coordinations,
            'success_rate': successful / total_coordinations,
            'average_duration': avg_duration,
            'most_common_type': max(self.performance_metrics, key=self.performance_metrics.get),
            'coordination_types': dict(self.performance_metrics)
        }
