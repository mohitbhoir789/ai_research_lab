#backend/app/utils/memory_manager.py

"""
Memory Manager Module
Handles persistent storage and retrieval of session data and conversation history.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages persistent memory across sessions including conversation history and summaries."""

    def __init__(self, sessions_dir: str = "sessions"):
        """Initialize the memory manager with a sessions directory."""
        self.sessions_dir = sessions_dir
        self._ensure_sessions_dir()

    def _ensure_sessions_dir(self) -> None:
        """Create sessions directory if it doesn't exist."""
        if not os.path.exists(self.sessions_dir):
            os.makedirs(self.sessions_dir)
            logger.info(f"Created sessions directory: {self.sessions_dir}")

    def create_session(self) -> str:
        """
        Create a new session with a unique ID.
        
        Returns:
            Session ID string
        """
        import uuid
        session_id = str(uuid.uuid4())
        
        # Initialize empty session data
        session_data = {
            "created_at": self.get_timestamp(),
            "last_updated": self.get_timestamp(),
            "interactions": 0,
            "global_context": {
                "conversation_history": [],
                "uploaded_files": [],
                "summaries": []
            }
        }
        
        self.save_session(session_id, session_data)
        logger.info(f"Created new session: {session_id}")
        
        return session_id

    def save_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Save session data to persistent storage.
        
        Args:
            session_id: Session identifier
            data: Session data to save
            
        Returns:
            Boolean indicating success
        """
        try:
            # Update metadata
            data["last_updated"] = self.get_timestamp()
            if "interactions" in data:
                data["interactions"] += 1
            
            # Save to file
            file_path = os.path.join(self.sessions_dir, f"{session_id}.json")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {str(e)}")
            return False

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from persistent storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dict or None if not found
        """
        try:
            file_path = os.path.join(self.sessions_dir, f"{session_id}.json")
            if not os.path.exists(file_path):
                return None
                
            with open(file_path, "r") as f:
                data = json.load(f)
            
            logger.debug(f"Loaded session {session_id}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Boolean indicating success
        """
        try:
            file_path = os.path.join(self.sessions_dir, f"{session_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted session {session_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions with metadata.
        
        Returns:
            List of session info dictionaries
        """
        try:
            sessions = []
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith(".json"):
                    session_id = filename[:-5]  # Remove .json
                    data = self.load_session(session_id)
                    if data:
                        sessions.append({
                            "session_id": session_id,
                            "created_at": data.get("created_at"),
                            "last_updated": data.get("last_updated"),
                            "interactions": data.get("interactions", 0)
                        })
            return sessions
            
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []

    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()

    def add_summary(self, session_id: str, summary: str) -> bool:
        """
        Add a session summary.
        
        Args:
            session_id: Session identifier
            summary: Summary text
            
        Returns:
            Boolean indicating success
        """
        try:
            data = self.load_session(session_id)
            if not data:
                return False
            
            if "summaries" not in data["global_context"]:
                data["global_context"]["summaries"] = []
            
            data["global_context"]["summaries"].append({
                "timestamp": self.get_timestamp(),
                "content": summary
            })
            
            return self.save_session(session_id, data)
            
        except Exception as e:
            logger.error(f"Error adding summary for session {session_id}: {str(e)}")
            return False