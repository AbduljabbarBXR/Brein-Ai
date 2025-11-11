import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SyncManager:
    """
    Manages synchronization between cloud and mobile devices with delta updates.
    Handles offline capabilities and data consistency.
    """

    def __init__(self, db_path: str = "memory/brein_memory.db", sync_dir: str = "sync/"):
        self.db_path = db_path
        self.sync_dir = sync_dir
        os.makedirs(sync_dir, exist_ok=True)

        # Initialize sync metadata
        self._init_sync_tables()

    def _init_sync_tables(self):
        """Initialize sync-related database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sync metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_metadata (
                device_id TEXT,
                last_sync_timestamp DATETIME,
                last_sync_version INTEGER,
                pending_changes INTEGER DEFAULT 0,
                PRIMARY KEY (device_id)
            )
        ''')

        # Change log for delta sync
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS change_log (
                change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT,
                change_type TEXT,  -- INSERT, UPDATE, DELETE
                change_data TEXT,  -- JSON data of the change
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                version INTEGER,
                synced INTEGER DEFAULT 0  -- 0=pending, 1=synced
            )
        ''')

        # Device registry
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                device_name TEXT,
                device_type TEXT,  -- mobile, desktop, server
                last_seen DATETIME,
                capabilities TEXT,  -- JSON array of device capabilities
                registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def register_device(self, device_id: str, device_name: str,
                       device_type: str = "mobile", capabilities: List[str] = None) -> bool:
        """
        Register a new device for synchronization.

        Args:
            device_id: Unique device identifier
            device_name: Human-readable device name
            device_type: Type of device (mobile, desktop, server)
            capabilities: List of device capabilities

        Returns:
            True if registration successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO devices (device_id, device_name, device_type, capabilities, last_seen)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                device_id,
                device_name,
                device_type,
                json.dumps(capabilities or []),
                datetime.now().isoformat()
            ))

            # Initialize sync metadata
            cursor.execute('''
                INSERT OR IGNORE INTO sync_metadata (device_id, last_sync_timestamp, last_sync_version, pending_changes)
                VALUES (?, ?, ?, ?)
            ''', (device_id, datetime.now().isoformat(), 0, 0))

            conn.commit()
            conn.close()

            logger.info(f"Registered device: {device_name} ({device_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to register device {device_id}: {e}")
            return False

    def create_sync_delta(self, device_id: str, since_version: int = 0) -> Dict[str, Any]:
        """
        Create a delta sync package for a device.

        Args:
            device_id: Target device ID
            since_version: Version to sync from

        Returns:
            Dictionary containing sync delta data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get changes since the specified version
            cursor.execute('''
                SELECT change_id, node_id, change_type, change_data, version
                FROM change_log
                WHERE version > ? AND synced = 0
                ORDER BY version ASC
            ''', (since_version,))

            changes = []
            max_version = since_version

            for row in cursor.fetchall():
                change_id, node_id, change_type, change_data, version = row
                changes.append({
                    "change_id": change_id,
                    "node_id": node_id,
                    "change_type": change_type,
                    "change_data": json.loads(change_data) if change_data else None,
                    "version": version
                })
                max_version = max(max_version, version)

            # Get current memory stats
            cursor.execute("SELECT COUNT(*) FROM nodes")
            total_nodes = cursor.fetchone()[0]

            conn.close()

            delta = {
                "device_id": device_id,
                "since_version": since_version,
                "to_version": max_version,
                "changes": changes,
                "total_nodes": total_nodes,
                "timestamp": datetime.now().isoformat(),
                "sync_type": "delta"
            }

            return delta

        except Exception as e:
            logger.error(f"Failed to create sync delta for {device_id}: {e}")
            return {}

    def apply_sync_delta(self, delta_data: Dict[str, Any]) -> bool:
        """
        Apply a sync delta received from another device.

        Args:
            delta_data: Delta sync data to apply

        Returns:
            True if application successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            changes_applied = 0

            for change in delta_data.get("changes", []):
                change_type = change["change_type"]
                node_id = change["node_id"]
                change_data = change["change_data"]

                if change_type == "INSERT":
                    # Insert new memory node
                    cursor.execute('''
                        INSERT OR REPLACE INTO nodes
                        (node_id, content, type, embedding, metadata, memory_type)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        node_id,
                        change_data.get("content"),
                        change_data.get("type", "conversational"),
                        change_data.get("embedding"),
                        json.dumps(change_data.get("metadata")),
                        change_data.get("memory_type", "working")
                    ))

                elif change_type == "UPDATE":
                    # Update existing node
                    cursor.execute('''
                        UPDATE nodes SET
                            content = ?,
                            type = ?,
                            embedding = ?,
                            metadata = ?,
                            memory_type = ?,
                            updated_at = ?
                        WHERE node_id = ?
                    ''', (
                        change_data.get("content"),
                        change_data.get("type"),
                        change_data.get("embedding"),
                        json.dumps(change_data.get("metadata")),
                        change_data.get("memory_type"),
                        datetime.now().isoformat(),
                        node_id
                    ))

                elif change_type == "DELETE":
                    # Mark node as deleted (soft delete)
                    cursor.execute('''
                        UPDATE nodes SET type = 'deleted', updated_at = ?
                        WHERE node_id = ?
                    ''', (datetime.now().isoformat(), node_id))

                changes_applied += 1

            # Update sync metadata
            device_id = delta_data.get("device_id")
            if device_id:
                cursor.execute('''
                    UPDATE sync_metadata SET
                        last_sync_timestamp = ?,
                        last_sync_version = ?,
                        pending_changes = 0
                    WHERE device_id = ?
                ''', (
                    datetime.now().isoformat(),
                    delta_data.get("to_version", 0),
                    device_id
                ))

            conn.commit()
            conn.close()

            logger.info(f"Applied {changes_applied} changes from delta sync")
            return True

        except Exception as e:
            logger.error(f"Failed to apply sync delta: {e}")
            return False

    def log_change(self, node_id: str, change_type: str, change_data: Dict[str, Any]) -> int:
        """
        Log a change for synchronization.

        Args:
            node_id: Node that changed
            change_type: Type of change (INSERT, UPDATE, DELETE)
            change_data: Data associated with the change

        Returns:
            Version number of the change
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get next version number
            cursor.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM change_log")
            version = cursor.fetchone()[0]

            # Insert change log
            cursor.execute('''
                INSERT INTO change_log (node_id, change_type, change_data, version)
                VALUES (?, ?, ?, ?)
            ''', (
                node_id,
                change_type,
                json.dumps(change_data),
                version
            ))

            change_id = cursor.lastrowid

            # Update pending changes count for all devices
            cursor.execute('''
                UPDATE sync_metadata SET pending_changes = pending_changes + 1
            ''')

            conn.commit()
            conn.close()

            return version

        except Exception as e:
            logger.error(f"Failed to log change: {e}")
            return -1

    def get_sync_status(self, device_id: str) -> Dict[str, Any]:
        """
        Get synchronization status for a device.

        Args:
            device_id: Device ID to check

        Returns:
            Dictionary with sync status information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT last_sync_timestamp, last_sync_version, pending_changes
                FROM sync_metadata WHERE device_id = ?
            ''', (device_id,))

            row = cursor.fetchone()

            if row:
                last_sync, last_version, pending = row
                return {
                    "device_id": device_id,
                    "last_sync": last_sync,
                    "last_version": last_version,
                    "pending_changes": pending,
                    "status": "registered"
                }
            else:
                return {
                    "device_id": device_id,
                    "status": "not_registered"
                }

        except Exception as e:
            logger.error(f"Failed to get sync status for {device_id}: {e}")
            return {"device_id": device_id, "status": "error"}

    def create_offline_bundle(self, device_id: str, max_nodes: int = 1000) -> str:
        """
        Create an offline bundle for a device with essential data.

        Args:
            device_id: Target device ID
            max_nodes: Maximum number of memory nodes to include

        Returns:
            Path to created bundle file
        """
        try:
            bundle_data = {
                "device_id": device_id,
                "created_at": datetime.now().isoformat(),
                "version": 1,
                "memory_nodes": [],
                "neural_mesh": {},
                "sync_metadata": self.get_sync_status(device_id)
            }

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get recent memory nodes (prioritize working and long-term memory)
            cursor.execute('''
                SELECT node_id, content, type, metadata, memory_type
                FROM nodes
                WHERE memory_type IN ('working', 'long_term') AND type != 'deleted'
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (max_nodes,))

            for row in cursor.fetchall():
                node_id, content, node_type, metadata_str, memory_type = row
                bundle_data["memory_nodes"].append({
                    "node_id": node_id,
                    "content": content,
                    "type": node_type,
                    "metadata": json.loads(metadata_str) if metadata_str else {},
                    "memory_type": memory_type
                })

            conn.close()

            # Save bundle to file
            bundle_filename = f"offline_bundle_{device_id}_{int(datetime.now().timestamp())}.json"
            bundle_path = os.path.join(self.sync_dir, bundle_filename)

            with open(bundle_path, "w") as f:
                json.dump(bundle_data, f, indent=2)

            logger.info(f"Created offline bundle: {bundle_path}")
            return bundle_path

        except Exception as e:
            logger.error(f"Failed to create offline bundle for {device_id}: {e}")
            return None

    def get_registered_devices(self) -> List[Dict[str, Any]]:
        """Get list of all registered devices."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT d.device_id, d.device_name, d.device_type, d.last_seen, d.capabilities,
                       sm.last_sync_version, sm.pending_changes
                FROM devices d
                LEFT JOIN sync_metadata sm ON d.device_id = sm.device_id
                ORDER BY d.last_seen DESC
            ''')

            devices = []
            for row in cursor.fetchall():
                device_id, name, device_type, last_seen, capabilities, last_version, pending = row
                devices.append({
                    "device_id": device_id,
                    "device_name": name,
                    "device_type": device_type,
                    "last_seen": last_seen,
                    "capabilities": json.loads(capabilities) if capabilities else [],
                    "last_sync_version": last_version or 0,
                    "pending_changes": pending or 0
                })

            conn.close()
            return devices

        except Exception as e:
            logger.error(f"Failed to get registered devices: {e}")
            return []