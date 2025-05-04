#backend/app/utils/memory_manager.py


import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

class MemoryManager:
    def __init__(self, storage_dir: str = "memory_store"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.sessions: Dict[str, Dict[str, Any]] = {}  # in-memory cache

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "global_context": {
                "conversation_history": []
            },
            "created_at": self.get_timestamp(),
            "last_updated": self.get_timestamp()
        }
        self.save_session(session_id, self.sessions[session_id])
        return session_id

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                session_data = json.load(f)
                self.sessions[session_id] = session_data
                return session_data
        return None

    def save_session(self, session_id: str, session_data: Dict[str, Any]):
        path = os.path.join(self.storage_dir, f"{session_id}.json")
        session_data["last_updated"] = self.get_timestamp()
        with open(path, "w") as f:
            json.dump(session_data, f, indent=2)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
        path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        for file in os.listdir(self.storage_dir):
            if file.endswith(".json"):
                with open(os.path.join(self.storage_dir, file), "r") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": file.replace(".json", ""),
                        "created_at": data.get("created_at"),
                        "last_updated": data.get("last_updated"),
                        "message_count": len(data.get("global_context", {}).get("conversation_history", [])),
                        "title": "Untitled"
                    })
        return sessions

    def get_timestamp(self) -> str:
        return datetime.now().isoformat()