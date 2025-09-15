
"""
Session management for ADK-based interview system
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import asyncio
import time

@dataclass
class SessionState:
    """Tracks the state of a client session with ADK agents"""
    # Original session state
    is_receiving_response: bool = False
    interrupted: bool = False
    current_tool_execution: Optional[asyncio.Task] = None
    current_audio_stream: Optional[Any] = None
    genai_session: Optional[Any] = None
    received_model_response: bool = False
    
    # ADK-specific state
    shared_state: Dict[str, Any] = field(default_factory=dict)
    conversational_agent: Optional[Any] = None
    state_agent: Optional[Any] = None
    state_agent_processor: Optional[asyncio.Task] = None
    session_start_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize shared state after object creation"""
        if not self.shared_state:
            self.shared_state = {
                "session_id": "",
                "start_time": self.session_start_time,
                "last_update": self.session_start_time,
                "conversation_active": False
            }
    
    @property
    def session_id(self) -> str:
        """Get the session ID from shared state"""
        return self.shared_state.get("session_id", "")

# Global session storage
active_sessions: Dict[str, SessionState] = {}

def create_session(session_id: str) -> SessionState:
    """Create and store a new session with ADK agents"""
    session = SessionState()
    session.shared_state["session_id"] = session_id
    
    # Import here to avoid circular imports
    from core.conversational_agent import ConversationalAgent
    from core.state_agent import StateAgent
    
    # Initialize agents
    session.conversational_agent = ConversationalAgent(session_id, session.shared_state)
    
    # Initialize StateAgent with extractor configuration
    # You can change this to "real" when you want to use your actual extractor
    extractor_type = "dummy"  # Change to "real" for production
    extractor_config = {
        # Add your extractor configuration here when using "real"
        # "api_endpoint": "https://your-api.com/analyze",
        # "api_key": "your-api-key"
    }
    session.state_agent = StateAgent(session_id, session.shared_state, extractor_type, extractor_config)
    
    # Start the state agent processor (will be started when session is used)
    # The processor will be started in the websocket handler when the session is active
    
    active_sessions[session_id] = session
    return session

def get_session(session_id: str) -> Optional[SessionState]:
    """Get an existing session"""
    return active_sessions.get(session_id)

def remove_session(session_id: str) -> None:
    """Remove a session and cleanup agents"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        
        # Cancel state agent processor if running
        if session.state_agent_processor and not session.state_agent_processor.done():
            session.state_agent_processor.cancel()
        
        del active_sessions[session_id] 