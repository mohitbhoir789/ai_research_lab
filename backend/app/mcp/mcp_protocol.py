"""
MCP Protocol Module
Defines the communication protocol for interacting with the MCP server.
"""

import json
import aiohttp
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
    EXPERIMENTAL = "experimental"


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
    
    async def _make_request(self, endpoint: str, request_obj: BaseModel) -> Dict[str, Any]:
        """
        Make an HTTP request to the MCP server
        
        Args:
            endpoint: API endpoint
            request_obj: Request object to serialize
            
        Returns:
            Dict containing the response data
        """
        url = f"{self.server_url}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        data = serialize_request(request_obj)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error from server: {error_text}")
                        return {"success": False, "error": f"Server error: {response.status}"}
        except Exception as e:
            self.logger.error(f"Error connecting to server: {str(e)}")
            return {"success": False, "error": f"Connection error: {str(e)}"}
    
    async def process_message(self, message: str, session_id: Optional[str] = None, 
                        uploaded_files: Optional[List[FileInfo]] = None,
                        model: Optional[str] = None, provider: Optional[str] = None) -> ProcessResponse:
        """
        Process a user message
        
        Args:
            message: The user's message
            session_id: Optional session ID
            uploaded_files: Optional list of uploaded files
            model: Optional model to use
            provider: Optional provider to use
            
        Returns:
            ProcessResponse with the results
        """
        request = ProcessRequest(
            message=message,
            session_id=session_id,
            uploaded_files=uploaded_files,
            model=model,
            provider=provider
        )
        
        response_data = await self._make_request("process", request)
        return ProcessResponse(**response_data)
    
    async def start_session(self, session_id: Optional[str] = None) -> StartSessionResponse:
        """
        Start a new session or resume an existing one
        
        Args:
            session_id: Optional existing session ID to resume
            
        Returns:
            StartSessionResponse with the session ID
        """
        request = StartSessionRequest(session_id=session_id)
        response_data = await self._make_request("start_session", request)
        return StartSessionResponse(**response_data)
    
    async def list_sessions(self) -> ListSessionsResponse:
        """
        List all available sessions
        
        Returns:
            ListSessionsResponse with session information
        """
        request = ListSessionsRequest()
        response_data = await self._make_request("list_sessions", request)
        return ListSessionsResponse(**response_data)
    
    async def delete_session(self, session_id: str) -> DeleteSessionResponse:
        """
        Delete a session and its associated data
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            DeleteSessionResponse indicating success
        """
        request = DeleteSessionRequest(session_id=session_id)
        response_data = await self._make_request("delete_session", request)
        return DeleteSessionResponse(**response_data)
    
    async def get_agents(self) -> GetAgentsResponse:
        """
        Get information about all available agents
        
        Returns:
            GetAgentsResponse with agent information
        """
        request = GetAgentsRequest()
        response_data = await self._make_request("get_agents", request)
        return GetAgentsResponse(**response_data)
    
    async def set_active_agent(self, agent_id: str) -> SetActiveAgentResponse:
        """
        Set an agent as the active agent
        
        Args:
            agent_id: ID of the agent to set as active
            
        Returns:
            SetActiveAgentResponse indicating success
        """
        request = SetActiveAgentRequest(agent_id=agent_id)
        response_data = await self._make_request("set_active_agent", request)
        return SetActiveAgentResponse(**response_data)
    
    async def generate_session_summary(self, session_id: str) -> SessionSummaryResponse:
        """
        Generate a summary of a session
        
        Args:
            session_id: ID of the session to summarize
            
        Returns:
            SessionSummaryResponse with the summary
        """
        request = SessionSummaryRequest(session_id=session_id)
        response_data = await self._make_request("generate_session_summary", request)
        return SessionSummaryResponse(**response_data)
    
    async def update_model_settings(self, model: str, provider: str) -> UpdateModelResponse:
        """
        Update model settings for all agents
        
        Args:
            model: The LLM model to use
            provider: The LLM provider (groq, openai, gemini)
            
        Returns:
            UpdateModelResponse indicating updated settings
        """
        request = UpdateModelRequest(model=model, provider=provider)
        response_data = await self._make_request("update_model_settings", request)
        return UpdateModelResponse(**response_data)
    
    async def upload_file(self, file_path: str, file_type: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a file for processing
        
        Args:
            file_path: Path to the file
            file_type: Type of the file (pdf, csv, txt, etc.)
            session_id: Optional session ID
            
        Returns:
            Dict containing upload and processing results
        """
        # In a real implementation, this would use aiohttp's file upload capabilities
        # For now, we'll create a custom request
        url = f"{self.server_url}/upload_file"
        
        file_info = {
            "path": file_path,
            "type": file_type,
            "session_id": session_id
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # In a real implementation, use FormData for file uploads
                async with session.post(url, json=file_info) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error uploading file: {error_text}")
                        return {"success": False, "error": f"Upload error: {response.status}"}
        except Exception as e:
            self.logger.error(f"Error connecting to server: {str(e)}")
            return {"success": False, "error": f"Connection error: {str(e)}"}


# Example usage of the protocol handler
async def example_usage():
    handler = MCPProtocolHandler()
    
    # Start a new session
    session_response = await handler.start_session()
    session_id = session_response.session_id
    print(f"Started session: {session_id}")
    
    # Process a message
    process_response = await handler.process_message(
        message="What are the latest advances in quantum computing?",
        session_id=session_id
    )
    print(f"Response: {process_response.final_output}")
    
    # List all sessions
    sessions_response = await handler.list_sessions()
    print(f"Available sessions: {len(sessions_response.sessions)}")
    
    # Get agent information
    agents_response = await handler.get_agents()
    print(f"Available agents: {len(agents_response.agents)}")
    
    # Generate a session summary
    summary_response = await handler.generate_session_summary(session_id)
    print(f"Session summary: {summary_response.summary}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())