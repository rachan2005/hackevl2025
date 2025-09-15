
"""
Conversational Agent for ADK-based interview system
Handles natural speech-to-speech conversation with timestamping capabilities
"""

import logging
import asyncio
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from google.genai import types

logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """State for tracking conversation flow and timestamps"""
    current_question: Optional[str] = None
    current_answer: Optional[str] = None
    question_timestamp: Optional[float] = None
    answer_timestamp: Optional[float] = None
    conversation_history: list = field(default_factory=list)
    is_question_complete: bool = False
    is_answer_complete: bool = False

class ConversationalAgent:
    """Handles natural conversation flow with timestamping and state signaling"""
    
    def __init__(self, session_id: str, shared_state: Dict[str, Any]):
        self.session_id = session_id
        self.shared_state = shared_state
        self.conversation_state = ConversationState()
        self.state_agent_signal_queue = asyncio.Queue()
        
    async def process_user_input(self, input_data: str, input_type: str = "text") -> Dict[str, Any]:
        """
        Process user input and determine if it's a question or answer
        Returns response data and triggers appropriate tool calls
        """
        current_time = time.time()
        
        if input_type == "text":
            # Determine if this is a question or answer based on context
            if self._is_question(input_data):
                await self._handle_question(input_data, current_time)
            else:
                await self._handle_answer(input_data, current_time)
        
        return {
            "processed": True,
            "input_type": input_type,
            "timestamp": current_time,
            "conversation_state": self.conversation_state
        }
    
    def _is_question(self, text: str) -> bool:
        """Determine if the input text is a question"""
        # Simple heuristic - can be enhanced with more sophisticated NLP
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you', 'could you', 'would you']
        text_lower = text.lower().strip()
        
        # Check for question mark
        if text_lower.endswith('?'):
            return True
            
        # Check for question words at the beginning
        for indicator in question_indicators:
            if text_lower.startswith(indicator):
                return True
                
        # If we have a current question and this doesn't seem like a question, it's likely an answer
        if self.conversation_state.current_question and not any(indicator in text_lower for indicator in question_indicators):
            return False
            
        return False
    
    async def _handle_question(self, question: str, timestamp: float):
        """Handle a new question with timestamping"""
        logger.info(f"Processing question: {question}")
        
        # Store the question and timestamp
        self.conversation_state.current_question = question
        self.conversation_state.question_timestamp = timestamp
        self.conversation_state.is_question_complete = True
        self.conversation_state.is_answer_complete = False
        
        # Update shared state
        self.shared_state["current_question"] = question
        self.shared_state["question_timestamp"] = timestamp
        self.shared_state["last_update"] = timestamp
        
        # Add to conversation history
        self.conversation_state.conversation_history.append({
            "type": "question",
            "content": question,
            "timestamp": timestamp
        })
        
        # Signal state agent
        await self._signal_state_agent("question_timestamped", {
            "question": question,
            "timestamp": timestamp
        })
        
        logger.info(f"Question timestamped and state agent signaled: {timestamp}")
    
    async def _handle_answer(self, answer: str, timestamp: float):
        """Handle an answer with timestamping"""
        logger.info(f"Processing answer: {answer}")
        
        # Store the answer and timestamp
        self.conversation_state.current_answer = answer
        self.conversation_state.answer_timestamp = timestamp
        self.conversation_state.is_answer_complete = True
        
        # Update shared state
        self.shared_state["current_answer"] = answer
        self.shared_state["answer_timestamp"] = timestamp
        self.shared_state["last_update"] = timestamp
        
        # Add to conversation history
        self.conversation_state.conversation_history.append({
            "type": "answer",
            "content": answer,
            "timestamp": timestamp
        })
        
        # Signal state agent
        await self._signal_state_agent("answer_timestamped", {
            "answer": answer,
            "timestamp": timestamp
        })
        
        logger.info(f"Answer timestamped and state agent signaled: {timestamp}")
    
    async def _handle_qa_pair(self, question: str, answer: str, timestamp: float):
        """Handle a complete question-answer pair with single timestamp"""
        logger.info(f"Processing Q&A pair: Q: {question[:50]}... A: {answer[:50]}...")
        
        # Store both question and answer with the same timestamp
        self.conversation_state.current_question = question
        self.conversation_state.current_answer = answer
        self.conversation_state.question_timestamp = timestamp
        self.conversation_state.answer_timestamp = timestamp
        self.conversation_state.is_question_complete = True
        self.conversation_state.is_answer_complete = True
        
        # Update shared state
        self.shared_state["current_question"] = question
        self.shared_state["current_answer"] = answer
        self.shared_state["question_timestamp"] = timestamp
        self.shared_state["answer_timestamp"] = timestamp
        self.shared_state["last_update"] = timestamp
        
        # Add to conversation history as a pair
        self.conversation_state.conversation_history.append({
            "type": "qa_pair",
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        })
        
        # Signal state agent with the complete Q&A pair
        await self._signal_state_agent("qa_pair_timestamped", {
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        })
        
        logger.info(f"Q&A pair timestamped and state agent signaled: {timestamp}")
    
    async def _signal_state_agent(self, signal_type: str, data: Dict[str, Any]):
        """Signal the state agent to process new information"""
        signal = {
            "type": signal_type,
            "data": data,
            "timestamp": time.time(),
            "session_id": self.session_id
        }
        
        await self.state_agent_signal_queue.put(signal)
        logger.info(f"📤 Signaled state agent with: {signal_type}")
        logger.info(f"📊 Signal data: {data}")
        logger.info(f"📋 Queue size: {self.state_agent_signal_queue.qsize()}")
    
    async def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context for AI response generation"""
        return {
            "current_question": self.conversation_state.current_question,
            "current_answer": self.conversation_state.current_answer,
            "question_timestamp": self.conversation_state.question_timestamp,
            "answer_timestamp": self.conversation_state.answer_timestamp,
            "conversation_history": self.conversation_state.conversation_history,
            "is_question_complete": self.conversation_state.is_question_complete,
            "is_answer_complete": self.conversation_state.is_answer_complete
        }
    
    def get_shared_state(self) -> Dict[str, Any]:
        """Get the current shared state"""
        return self.shared_state.copy()
