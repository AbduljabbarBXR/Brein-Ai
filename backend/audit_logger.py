import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

logger = logging.getLogger(__name__)

class AuditLogger:
    """
    Comprehensive audit logging system for Brein AI safety and provenance tracking.
    Logs all operations, decisions, and data flows for compliance and debugging.
    """

    def __init__(self, log_dir: str = "logs/audit/"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create subdirectories for different log types
        self.operation_logs = os.path.join(log_dir, "operations/")
        self.security_logs = os.path.join(log_dir, "security/")
        self.data_flow_logs = os.path.join(log_dir, "data_flow/")

        for subdir in [self.operation_logs, self.security_logs, self.data_flow_logs]:
            os.makedirs(subdir, exist_ok=True)

    def log_operation(self, operation_type: str, user_id: str = "system",
                     session_id: Optional[str] = None, details: Dict[str, Any] = None,
                     success: bool = True, error_message: str = None) -> str:
        """
        Log a system operation.

        Args:
            operation_type: Type of operation (query, ingest, sync, etc.)
            user_id: User or system identifier
            session_id: Session identifier if applicable
            details: Additional operation details
            success: Whether operation succeeded
            error_message: Error message if failed

        Returns:
            Log entry ID
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "user_id": user_id,
            "session_id": session_id,
            "success": success,
            "error_message": error_message,
            "details": details or {},
            "log_type": "operation"
        }

        log_id = self._generate_log_id(log_entry)
        log_entry["log_id"] = log_id

        filename = f"operation_{operation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_id[:8]}.json"
        filepath = os.path.join(self.operation_logs, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        logger.info(f"Operation logged: {operation_type} by {user_id} - {'SUCCESS' if success else 'FAILED'}")
        return log_id

    def log_security_event(self, event_type: str, severity: str, source: str,
                          details: Dict[str, Any] = None, user_id: str = "system") -> str:
        """
        Log a security-related event.

        Args:
            event_type: Type of security event (access_denied, suspicious_content, etc.)
            severity: Severity level (low, medium, high, critical)
            source: Source of the event (web_fetch, api_call, etc.)
            details: Event details
            user_id: Associated user ID

        Returns:
            Log entry ID
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "source": source,
            "user_id": user_id,
            "details": details or {},
            "log_type": "security"
        }

        log_id = self._generate_log_id(log_entry)
        log_entry["log_id"] = log_id

        filename = f"security_{severity}_{event_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_id[:8]}.json"
        filepath = os.path.join(self.security_logs, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        logger.warning(f"Security event logged: {event_type} ({severity}) from {source}")
        return log_id

    def log_data_flow(self, data_type: str, source: str, destination: str,
                     operation: str, data_hash: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Log data movement and transformation.

        Args:
            data_type: Type of data (memory_node, embedding, web_content, etc.)
            source: Data source
            destination: Data destination
            operation: Operation performed (ingest, retrieve, sync, etc.)
            data_hash: Hash of the data for integrity checking
            metadata: Additional metadata

        Returns:
            Log entry ID
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "data_type": data_type,
            "source": source,
            "destination": destination,
            "operation": operation,
            "data_hash": data_hash,
            "metadata": metadata or {},
            "log_type": "data_flow"
        }

        log_id = self._generate_log_id(log_entry)
        log_entry["log_id"] = log_id

        filename = f"data_flow_{data_type}_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_id[:8]}.json"
        filepath = os.path.join(self.data_flow_logs, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        logger.debug(f"Data flow logged: {data_type} {operation} from {source} to {destination}")
        return log_id

    def log_web_fetch(self, url: str, success: bool, safety_warnings: List[str] = None,
                     quarantine_path: str = None, user_id: str = "system") -> str:
        """
        Log web content fetching operations.

        Args:
            url: Fetched URL
            success: Whether fetch succeeded
            safety_warnings: Any safety warnings detected
            quarantine_path: Path to quarantined content if applicable
            user_id: User who initiated the fetch

        Returns:
            Log entry ID
        """
        details = {
            "url": url,
            "safety_warnings": safety_warnings or [],
            "quarantine_path": quarantine_path,
            "domain": self._extract_domain(url)
        }

        if quarantine_path:
            operation_type = "web_fetch_quarantined"
        elif success:
            operation_type = "web_fetch_success"
        else:
            operation_type = "web_fetch_failed"

        return self.log_operation(
            operation_type=operation_type,
            user_id=user_id,
            details=details,
            success=success
        )

    def log_content_review(self, quarantine_path: str, approved: bool,
                          reviewer_id: str, notes: str = "") -> str:
        """
        Log content review decisions.

        Args:
            quarantine_path: Path to reviewed content
            approved: Whether content was approved
            reviewer_id: ID of the reviewer
            notes: Review notes

        Returns:
            Log entry ID
        """
        details = {
            "quarantine_path": quarantine_path,
            "approved": approved,
            "reviewer_id": reviewer_id,
            "notes": notes,
            "review_timestamp": datetime.now().isoformat()
        }

        operation_type = "content_approved" if approved else "content_rejected"

        return self.log_operation(
            operation_type=operation_type,
            user_id=reviewer_id,
            details=details,
            success=True
        )

    def log_memory_operation(self, operation: str, node_ids: List[str],
                           user_id: str = "system", session_id: str = None) -> str:
        """
        Log memory-related operations.

        Args:
            operation: Type of memory operation (add, retrieve, reinforce, etc.)
            node_ids: Affected memory node IDs
            user_id: User performing the operation
            session_id: Session context

        Returns:
            Log entry ID
        """
        details = {
            "node_ids": node_ids,
            "node_count": len(node_ids),
            "session_id": session_id
        }

        return self.log_operation(
            operation_type=f"memory_{operation}",
            user_id=user_id,
            session_id=session_id,
            details=details,
            success=True
        )

    def log_sync_operation(self, device_id: str, operation: str,
                          changes_count: int = 0, user_id: str = "system") -> str:
        """
        Log device synchronization operations.

        Args:
            device_id: Target device ID
            operation: Sync operation type (register, delta_sync, bundle_create, etc.)
            changes_count: Number of changes synced
            user_id: User initiating sync

        Returns:
            Log entry ID
        """
        details = {
            "device_id": device_id,
            "changes_count": changes_count
        }

        return self.log_operation(
            operation_type=f"sync_{operation}",
            user_id=user_id,
            details=details,
            success=True
        )

    def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get audit summary for the specified number of days.

        Args:
            days: Number of days to include in summary

        Returns:
            Summary statistics
        """
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)

        summary = {
            "period_days": days,
            "cutoff_date": cutoff_date.isoformat(),
            "operations": {},
            "security_events": {},
            "data_flow": {},
            "total_logs": 0
        }

        # Count logs in each category
        for log_dir, category in [
            (self.operation_logs, "operations"),
            (self.security_logs, "security_events"),
            (self.data_flow_logs, "data_flow")
        ]:
            if os.path.exists(log_dir):
                for filename in os.listdir(log_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(log_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                log_entry = json.load(f)

                            log_date = datetime.fromisoformat(log_entry["timestamp"])
                            if log_date >= cutoff_date:
                                summary["total_logs"] += 1

                                if category == "operations":
                                    op_type = log_entry.get("operation_type", "unknown")
                                    summary["operations"][op_type] = summary["operations"].get(op_type, 0) + 1
                                elif category == "security_events":
                                    event_type = log_entry.get("event_type", "unknown")
                                    severity = log_entry.get("severity", "unknown")
                                    key = f"{event_type}_{severity}"
                                    summary["security_events"][key] = summary["security_events"].get(key, 0) + 1
                                elif category == "data_flow":
                                    data_type = log_entry.get("data_type", "unknown")
                                    summary["data_flow"][data_type] = summary["data_flow"].get(data_type, 0) + 1

                        except Exception as e:
                            logger.error(f"Error reading log file {filename}: {e}")

        return summary

    def _generate_log_id(self, log_entry: Dict[str, Any]) -> str:
        """Generate a unique log entry ID."""
        content = json.dumps(log_entry, sort_keys=True, default=str)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"

    def search_logs(self, query: Dict[str, Any], log_type: str = "all",
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search audit logs based on criteria.

        Args:
            query: Search criteria dictionary
            log_type: Type of logs to search (operation, security, data_flow, all)
            limit: Maximum number of results

        Returns:
            List of matching log entries
        """
        results = []
        search_dirs = []

        if log_type in ["all", "operation"]:
            search_dirs.append(self.operation_logs)
        if log_type in ["all", "security"]:
            search_dirs.append(self.security_logs)
        if log_type in ["all", "data_flow"]:
            search_dirs.append(self.data_flow_logs)

        for log_dir in search_dirs:
            if os.path.exists(log_dir):
                for filename in os.listdir(log_dir):
                    if filename.endswith('.json') and len(results) < limit:
                        filepath = os.path.join(log_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                log_entry = json.load(f)

                            # Check if log entry matches query criteria
                            if self._matches_query(log_entry, query):
                                results.append(log_entry)

                        except Exception as e:
                            logger.error(f"Error reading log file {filename}: {e}")

        return results

    def _matches_query(self, log_entry: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if log entry matches search query."""
        for key, value in query.items():
            if key not in log_entry:
                return False

            log_value = log_entry[key]
            if isinstance(value, dict):
                # Nested query
                if not isinstance(log_value, dict) or not self._matches_query(log_value, value):
                    return False
            elif isinstance(value, list):
                # List contains check
                if not isinstance(log_value, list) or not all(v in log_value for v in value):
                    return False
            else:
                # Exact match
                if log_value != value:
                    return False

        return True