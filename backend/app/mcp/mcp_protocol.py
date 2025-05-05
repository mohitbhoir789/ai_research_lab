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