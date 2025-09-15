
"""
Tool execution and handling for ADK-based interview system
"""

import logging
import aiohttp
import time
import asyncio
from typing import Dict, Any, Optional
from config.config import CLOUD_FUNCTIONS
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

async def execute_tool(tool_name: str, params: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """Execute a tool based on name and parameters"""
    
    # Handle ADK-specific tools
    if tool_name in ["timestamp_qa_pair", "signal_state_agent", "detect_user_response"]:
        return await execute_adk_tool(tool_name, params, session_id)
    
    # Handle external cloud function tools
    try:
        if tool_name not in CLOUD_FUNCTIONS:
            logger.error(f"Tool not found: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

        base_url = CLOUD_FUNCTIONS[tool_name]
        # Convert params to URL query parameters
        query_string = urlencode(params)
        function_url = f"{base_url}?{query_string}" if params else base_url
        
        logger.debug(f"Calling cloud function for {tool_name}")
        logger.debug(f"URL with params: {function_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(function_url) as response:
                response_text = await response.text()
                logger.debug(f"Response status: {response.status}")
                logger.debug(f"Response headers: {dict(response.headers)}")
                logger.debug(f"Response body: {response_text}")
                
                if response.status != 200:
                    logger.error(f"Cloud function error: {response_text}")
                    return {"error": f"Cloud function returned status {response.status}"}
                
                try:
                    return await response.json()
                except Exception as e:
                    logger.error(f"Failed to parse JSON response: {response_text}")
                    return {"error": f"Invalid JSON response from cloud function: {str(e)}"}

    except aiohttp.ClientError as e:
        logger.error(f"Network error calling cloud function for {tool_name}: {str(e)}")
        return {"error": f"Failed to call cloud function: {str(e)}"}
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return {"error": f"Tool execution failed: {str(e)}"}

async def execute_adk_tool(tool_name: str, params: Dict[str, Any], session_id: Optional[str]) -> Dict[str, Any]:
    """Execute ADK-specific tools for timestamping and signaling"""
    
    if not session_id:
        return {"error": "Session ID required for ADK tools"}
    
    # Import here to avoid circular imports
    from core.session import get_session
    
    session = get_session(session_id)
    if not session or not session.conversational_agent:
        return {"error": "Session or conversational agent not found"}
    
    try:
        if tool_name == "timestamp_qa_pair":
            question = params.get("question", "")
            answer = params.get("answer", "")
            timestamp = time.time()
            
            # Process both question and answer together through conversational agent
            await session.conversational_agent._handle_qa_pair(question, answer, timestamp)
            
            return {
                "success": True,
                "tool": "timestamp_qa_pair",
                "question": question,
                "answer": answer,
                "timestamp": timestamp,
                "message": f"Q&A pair timestamped at {timestamp}"
            }
        
        elif tool_name == "signal_state_agent":
            signal_type = params.get("signal_type", "")
            signal_data = params.get("signal_data", {})
            
            # Create signal for state agent
            signal = {
                "type": signal_type,
                "data": signal_data,
                "timestamp": time.time(),
                "session_id": session_id
            }
            
            # Add signal to state agent queue
            await session.conversational_agent.state_agent_signal_queue.put(signal)
            
            return {
                "success": True,
                "tool": "signal_state_agent",
                "signal_type": signal_type,
                "message": f"State agent signaled with {signal_type}"
            }
        
        elif tool_name == "detect_user_response":
            # Tool to help detect and timestamp user responses
            user_response = params.get("user_response", "")
            response_type = params.get("response_type", "answer")  # "answer" or "question"
            
            if response_type == "answer":
                await session.conversational_agent._handle_answer(user_response, time.time())
            else:
                await session.conversational_agent._handle_question(user_response, time.time())
            
            return {
                "success": True,
                "tool": "detect_user_response",
                "response_type": response_type,
                "message": f"User {response_type} timestamped: {user_response[:50]}..."
            }
        
        else:
            return {"error": f"Unknown ADK tool: {tool_name}"}
    
    except Exception as e:
        logger.error(f"Error executing ADK tool {tool_name}: {str(e)}")
        return {"error": f"ADK tool execution failed: {str(e)}"}

async def start_state_agent_processor(session_id: str):
    """Start the state agent processor for a session"""
    from core.session import get_session
    
    session = get_session(session_id)
    if not session or not session.state_agent or not session.conversational_agent:
        logger.error(f"Cannot start state agent processor for session {session_id}")
        return
    
    async def process_state_agent_signals():
        """Process signals from conversational agent to state agent"""
        try:
            logger.info(f"🔄 State agent processor started for session {session_id}")
            while True:
                # Wait for signals from conversational agent
                logger.info(f"⏳ Waiting for signals in session {session_id}...")
                signal = await session.conversational_agent.state_agent_signal_queue.get()
                logger.info(f"📨 Received signal in session {session_id}: {signal.get('type', 'unknown')}")
                
                # Process the signal in state agent
                await session.state_agent.process_signal(signal)
                
                # Mark task as done
                session.conversational_agent.state_agent_signal_queue.task_done()
                logger.info(f"✅ Signal processed for session {session_id}")
                
        except asyncio.CancelledError:
            logger.info(f"🛑 State agent processor cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"❌ Error in state agent processor for session {session_id}: {str(e)}")
    
    # Start the processor task
    session.state_agent_processor = asyncio.create_task(process_state_agent_signals())
    logger.info(f"Started state agent processor for session {session_id}") 