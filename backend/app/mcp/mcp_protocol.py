"""
MCP Protocol Module
Defines the communication protocol for interacting with the MCP server.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class IntentType(str, Enum):
    """Types of user intent for routing"""
    DIRECT_QUERY = "direct_query"
    SUMMARY = "summary"
    RESEARCH = "research"


class MessageRole(str, Enum):
    """Roles in conversation messages"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A message in the conversation"""
    role: MessageRole
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentInfo(BaseModel):
    """Information about an agent"""
    id: str
    name: str
    description: str
    is_active: bool


class TraceEntry(BaseModel):
    """An entry in the execution trace"""
    stage: str
    agent: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FileInfo(BaseModel):
    """Information about an uploaded file"""
    path: str
    type: str
    name: str
    size: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SessionInfo(BaseModel):
    """Information about a session"""
    id: str
    created_at: str
    last_updated: str
    message_count: int
    title: str


class MCPRequest(BaseModel):
    """Base request to the MCP server"""
    session_id: Optional[str] = None
    

class ProcessRequest(MCPRequest):
    """Request to process a user message"""
    message: str
    uploaded_files: Optional[List[FileInfo]] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class StartSessionRequest(MCPRequest):
    """Request to start a new session or resume an existing one"""
    pass


class ListSessionsRequest(BaseModel):
    """Request to list all available sessions"""
    pass


class DeleteSessionRequest(MCPRequest):
    """Request to delete a session"""
    pass


class GetAgentsRequest(BaseModel):
    """Request to get information about all agents"""
    pass


class SetActiveAgentRequest(BaseModel):
    """Request to set the active agent"""
    agent_id: str


class SessionSummaryRequest(MCPRequest):
    """Request to generate a summary of a session"""
    pass


class UpdateModelRequest(BaseModel):
    """Request to update model settings"""
    model: str
    provider: str


class MCPResponse(BaseModel):
    """Base response from the MCP server"""
    success: bool
    error: Optional[str] = None


class ProcessResponse(MCPResponse):
    """Response from processing a user message"""
    trace: List[Dict[str, Any]]
    final_output: str
    session_id: str
    conversation: Optional[List[Message]] = None


class StartSessionResponse(MCPResponse):
    """Response from starting a session"""
    session_id: str


class ListSessionsResponse(MCPResponse):
    """Response from listing sessions"""
    sessions: List[SessionInfo]


class DeleteSessionResponse(MCPResponse):
    """Response from deleting a session"""
    deleted: bool


class GetAgentsResponse(MCPResponse):
    """Response from getting agent information"""
    agents: List[AgentInfo]


class SetActiveAgentResponse(MCPResponse):
    """Response from setting the active agent"""
    set: bool
    agent_name: Optional[str] = None


class SessionSummaryResponse(MCPResponse):
    """Response from generating a session summary"""
    summary: str


class UpdateModelResponse(MCPResponse):
    """Response from updating model settings"""
    model: str
    provider: str


# Helper functions for serialization/deserialization

def serialize_request(request: BaseModel) -> str:
    """Serialize a request object to JSON string"""
    return request.json()


def deserialize_request(json_str: str, request_type: type) -> BaseModel:
    """Deserialize a JSON string to a request object"""
    data = json.loads(json_str)
    return request_type(**data)


def serialize_response(response: BaseModel) -> str:
    """Serialize a response object to JSON string"""
    return response.json()


def deserialize_response(json_str: str, response_type: type) -> BaseModel:
    """Deserialize a JSON string to a response object"""
    data = json.loads(json_str)
    return response_type(**data)


# Protocol handler for the frontend

class MCPProtocolHandler:
    """
    Handles protocol communication between frontend and MCP server
    """
    
    def __init__(self, server_url: str = "http://localhost:8000/api"):
        """
        Initialize the protocol handler
        
        Args:
            server_url: URL of the MCP server API
        """
        self.server_url = server_url
        self.logger = logging.getLogger(__name__)
        self.user_query_history = []  # Store only user queries, not responses
        self.max_query_history_length = 10  # Keep only last 10 user queries
        self.max_query_length = 1000  # Maximum character length for stored queries - reduced from previous value

    def add_user_query(self, query: str) -> None:
        """
        Add a user query to the history with character limit
        
        Args:
            query: The user's query text
        """
        # Trim the query if it exceeds the maximum length
        if len(query) > self.max_query_length:
            trimmed_query = query[:self.max_query_length] + "..."
            self.logger.info(f"Query trimmed from {len(query)} to {self.max_query_length} characters")
        else:
            trimmed_query = query
            
        # Add to history
        self.user_query_history.append({
            "query": trimmed_query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history length
        if len(self.user_query_history) > self.max_query_history_length:
            self.user_query_history = self.user_query_history[-self.max_query_history_length:]
            
    def get_recent_queries(self, limit: int = 3) -> List[str]:
        """
        Get the most recent user queries
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of recent query strings
        """
        recent = self.user_query_history[-limit:] if len(self.user_query_history) > 0 else []
        return [item["query"] for item in recent]
        
    def get_total_context_size(self) -> int:
        """
        Calculate the total size of stored context in characters
        
        Returns:
            Total character count
        """
        return sum(len(item["query"]) for item in self.user_query_history)
        
    def enforce_query_limit(self, query: str) -> str:
        """
        Enforce the character limit on a query
        
        Args:
            query: The original query
            
        Returns:
            The query, potentially truncated to respect character limit
        """
        if len(query) > self.max_query_length:
            truncated = query[:self.max_query_length] + "..."
            self.logger.warning(f"Query truncated from {len(query)} to {self.max_query_length} characters")
            return truncated
        return query