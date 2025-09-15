
"""
WebSocket message handling for Gemini Multimodal Live Proxy Server
"""

import logging
import json
import asyncio
import base64
import traceback
import time
from typing import Any, Optional
from google.genai import types

from core.tool_handler import execute_tool, start_state_agent_processor
from core.session import create_session, remove_session, SessionState
from core.gemini_client import create_gemini_session

logger = logging.getLogger(__name__)

async def get_conversation_context_for_ai(session: SessionState) -> str:
    """Get conversation context and shared state for AI to use as memory"""
    try:
        context_parts = []
        
        # Get conversation context from conversational agent
        if session.conversational_agent:
            conv_context = await session.conversational_agent.get_conversation_context()
            
            # Add conversation history
            if conv_context.get("conversation_history"):
                context_parts.append("CONVERSATION HISTORY:")
                for item in conv_context["conversation_history"][-5:]:  # Last 5 items
                    context_parts.append(f"- {item['type'].upper()}: {item['content']} (at {item['timestamp']})")
            
            # Add current state
            if conv_context.get("current_question"):
                context_parts.append(f"CURRENT QUESTION: {conv_context['current_question']}")
            if conv_context.get("current_answer"):
                context_parts.append(f"CURRENT ANSWER: {conv_context['current_answer']}")
        
        # Get behavioral insights from state agent
        if session.state_agent:
            insights = session.state_agent.get_insights_summary()
            if insights["enriched_qa_pairs"] > 0:
                context_parts.append("ENRICHED Q&A PAIRS WITH BEHAVIORAL INSIGHTS:")
                context_parts.append(f"- Total Q&A pairs: {insights['enriched_qa_pairs']}")
                context_parts.append(f"- Enriched with insights: {insights['enriched_with_insights']}")
                context_parts.append(f"- Average confidence: {insights['average_confidence']:.2f}")
                
                # Add enriched Q&A pairs if available
                if session.shared_state.get("enriched_qa_pairs"):
                    enriched_pairs = session.shared_state["enriched_qa_pairs"]
                    context_parts.append(f"- Recent Q&A pairs with behavioral analysis:")
                    for pair in enriched_pairs[-3:]:  # Last 3 pairs
                        context_parts.append(f"  * Q: {pair['question']}")
                        context_parts.append(f"    A: {pair['answer']}")
                        if pair['is_enriched']:
                            context_parts.append(f"    🧠 Behavioral Insights: {pair['behavioral_insights']}")
                            context_parts.append(f"    📊 Confidence: {pair['confidence']:.2f}")
                        else:
                            context_parts.append(f"    ⏳ Behavioral analysis in progress...")
                        context_parts.append("")
                
                # Add latest enriched Q&A pair if available
                if session.shared_state.get("last_qa_pair"):
                    last_pair = session.shared_state["last_qa_pair"]
                    context_parts.append(f"- Latest Q&A pair:")
                    context_parts.append(f"  Q: {last_pair.question}")
                    context_parts.append(f"  A: {last_pair.answer}")
                    if last_pair.is_enriched:
                        context_parts.append(f"  🧠 Behavioral Insights: {last_pair.behavioral_insights}")
                    else:
                        context_parts.append(f"  ⏳ Behavioral analysis in progress...")
        
        # Add shared state information
        if session.shared_state:
            context_parts.append("SHARED STATE:")
            for key, value in session.shared_state.items():
                if key not in ["session_id", "start_time", "last_update"]:
                    context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts) if context_parts else "No context available"
    
    except Exception as e:
        logger.error(f"Error getting conversation context: {e}")
        return "Error retrieving context"

async def send_error_message(websocket: Any, error_data: dict) -> None:
    """Send formatted error message to client."""
    try:
        await websocket.send(json.dumps({
            "type": "error",
            "data": error_data
        }))
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")

async def cleanup_session(session: Optional[SessionState], session_id: str) -> None:
    """Clean up session resources."""
    try:
        if session:
            # Cancel any running tasks
            if session.current_tool_execution:
                session.current_tool_execution.cancel()
                try:
                    await session.current_tool_execution
                except asyncio.CancelledError:
                    pass
            
            # Close Gemini session
            if session.genai_session:
                try:
                    await session.genai_session.close()
                except Exception as e:
                    logger.error(f"Error closing Gemini session: {e}")
            
            # Remove session from active sessions
            remove_session(session_id)
            logger.info(f"Session {session_id} cleaned up and ended")
    except Exception as cleanup_error:
        logger.error(f"Error during session cleanup: {cleanup_error}")

async def handle_messages(websocket: Any, session: SessionState) -> None:
    """Handles bidirectional message flow between client and Gemini."""
    client_task = None
    gemini_task = None
    
    try:
        async with asyncio.TaskGroup() as tg:
            # Task 1: Handle incoming messages from client
            client_task = tg.create_task(handle_client_messages(websocket, session))
            # Task 2: Handle responses from Gemini
            gemini_task = tg.create_task(handle_gemini_responses(websocket, session))
    except* Exception as eg:
        handled = False
        for exc in eg.exceptions:
            if "Quota exceeded" in str(exc):
                logger.info("Quota exceeded error occurred")
                try:
                    # Send error message for UI handling
                    await send_error_message(websocket, {
                        "message": "Quota exceeded.",
                        "action": "Please wait a moment and try again in a few minutes.",
                        "error_type": "quota_exceeded"
                    })
                    # Send text message to show in chat
                    await websocket.send(json.dumps({
                        "type": "text",
                        "data": "⚠️ Quota exceeded. Please wait a moment and try again in a few minutes."
                    }))
                    handled = True
                    break
                except Exception as send_err:
                    logger.error(f"Failed to send quota error message: {send_err}")
            elif "connection closed" in str(exc).lower():
                logger.info("WebSocket connection closed")
                handled = True
                break
        
        if not handled:
            # For other errors, log and re-raise
            # Check if it's a connection-related error
            error_str = str(eg).lower()
            if any(phrase in error_str for phrase in [
                "connection closed", "websocket", "no close frame", 
                "network name is no longer available", "connection reset"
            ]):
                logger.info(f"WebSocket connection closed: {eg}")
            else:
                logger.error(f"Error in message handling: {eg}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
    finally:
        # Cancel tasks if they're still running
        if client_task and not client_task.done():
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass
        
        if gemini_task and not gemini_task.done():
            gemini_task.cancel()
            try:
                await gemini_task
            except asyncio.CancelledError:
                pass

async def handle_client_messages(websocket: Any, session: SessionState) -> None:
    """Handle incoming messages from the client."""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if "type" in data:
                    msg_type = data["type"]
                    if msg_type == "audio":
                        logger.debug("Client -> Gemini: Sending audio data...")
                    elif msg_type == "image":
                        logger.debug("Client -> Gemini: Sending image data...")
                    else:
                        # Replace audio data with placeholder in debug output
                        debug_data = data.copy()
                        if "data" in debug_data and debug_data["type"] == "audio":
                            debug_data["data"] = "<audio data>"
                        logger.debug(f"Client -> Gemini: {json.dumps(debug_data, indent=2)}")
                
                # Handle different types of input
                if "type" in data:
                    if data["type"] == "audio":
                        logger.debug("Sending audio to Gemini...")
                        await session.genai_session.send(input={
                            "data": data.get("data"),
                            "mime_type": "audio/pcm"
                        }, end_of_turn=True)
                        logger.debug("Audio sent to Gemini")
                    elif data["type"] == "image":
                        logger.info("Sending image to Gemini...")
                        await session.genai_session.send(input={
                            "data": data.get("data"),
                            "mime_type": "image/jpeg"
                        })
                        logger.info("Image sent to Gemini")
                    elif data["type"] == "text":
                        logger.info("Sending text to Gemini...")
                        
                        # Get current shared state and conversation context
                        context_info = await get_conversation_context_for_ai(session)
                        
                        # Send text with context
                        user_text = data.get('data')
                        message_with_context = f"[CONTEXT: {context_info}]\n\nUser: {user_text}"
                        await session.genai_session.send(input=message_with_context, end_of_turn=True)
                        logger.info("Text with context sent to Gemini")
                    elif data["type"] == "end":
                        logger.info("Received end signal")
                    else:
                        logger.warning(f"Unsupported message type: {data.get('type')}")
            except Exception as e:
                # Check if it's a connection-related error
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in [
                    "connection closed", "websocket", "no close frame", 
                    "network name is no longer available", "connection reset"
                ]):
                    logger.info(f"Client connection closed: {e}")
                    break  # Exit the message loop gracefully
                else:
                    logger.error(f"Error handling client message: {e}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    except Exception as e:
        error_str = str(e).lower()
        if not any(phrase in error_str for phrase in [
            "connection closed", "websocket", "no close frame", 
            "network name is no longer available", "connection reset"
        ]):
            logger.error(f"WebSocket connection error: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        else:
            logger.info(f"WebSocket connection closed: {e}")
        raise  # Re-raise to let the parent handle cleanup

async def handle_gemini_responses(websocket: Any, session: SessionState) -> None:
    """Handle responses from Gemini."""
    tool_queue = asyncio.Queue()  # Queue for tool responses
    
    # Start a background task to process tool calls
    tool_processor = asyncio.create_task(process_tool_queue(tool_queue, websocket, session))
    
    try:
        while True:
            async for response in session.genai_session.receive():
                try:
                    # Replace audio data with placeholder in debug output
                    debug_response = str(response)
                    if 'data=' in debug_response and 'mime_type=\'audio/pcm' in debug_response:
                        debug_response = debug_response.split('data=')[0] + 'data=<audio data>' + debug_response.split('mime_type=')[1]
                    logger.debug(f"Received response from Gemini: {debug_response}")
                    
                    # If there's a tool call, add it to the queue and continue
                    if response.tool_call:
                        await tool_queue.put(response.tool_call)
                        continue  # Continue processing other responses while tool executes
                    
                    # Process server content (including audio) immediately
                    await process_server_content(websocket, session, response.server_content)
                    
                except Exception as e:
                    # Check if it's a connection-related error
                    error_str = str(e).lower()
                    if any(phrase in error_str for phrase in [
                        "connection closed", "websocket", "no close frame", 
                        "network name is no longer available", "connection reset"
                    ]):
                        logger.info(f"Gemini connection closed: {e}")
                        break  # Exit the response loop gracefully
                    else:
                        logger.error(f"Error handling Gemini response: {e}")
                        logger.error(f"Full traceback:\n{traceback.format_exc()}")
    finally:
        # Cancel and clean up tool processor
        if tool_processor and not tool_processor.done():
            tool_processor.cancel()
            try:
                await tool_processor
            except asyncio.CancelledError:
                pass
        
        # Clear any remaining items in the queue
        while not tool_queue.empty():
            try:
                tool_queue.get_nowait()
                tool_queue.task_done()
            except asyncio.QueueEmpty:
                break

async def process_tool_queue(queue: asyncio.Queue, websocket: Any, session: SessionState):
    """Process tool calls from the queue."""
    while True:
        tool_call = await queue.get()
        try:
            function_responses = []
            for function_call in tool_call.function_calls:
                # Store the tool execution in session state
                session.current_tool_execution = asyncio.current_task()
                
                # Send function call to client (for UI feedback)
                await websocket.send(json.dumps({
                    "type": "function_call",
                    "data": {
                        "name": function_call.name,
                        "args": function_call.args
                    }
                }))
                
                tool_result = await execute_tool(function_call.name, function_call.args, session.session_id)
                
                # Send function response to client
                await websocket.send(json.dumps({
                    "type": "function_response",
                    "data": tool_result
                }))
                
                function_responses.append(
                    types.FunctionResponse(
                        name=function_call.name,
                        id=function_call.id,
                        response=tool_result
                    )
                )
                
                session.current_tool_execution = None
            
            if function_responses:
                tool_response = types.LiveClientToolResponse(
                    function_responses=function_responses
                )
                await session.genai_session.send(input=tool_response)
        except Exception as e:
            # Check if it's a connection-related error
            error_str = str(e).lower()
            if any(phrase in error_str for phrase in [
                "connection closed", "websocket", "no close frame", 
                "network name is no longer available", "connection reset"
            ]):
                logger.info(f"Tool processing connection closed: {e}")
            else:
                logger.error(f"Error processing tool call: {e}")
        finally:
            queue.task_done()

async def process_server_content(websocket: Any, session: SessionState, server_content: Any):
    """Process server content including audio and text."""
    # Check if server_content is None
    if server_content is None:
        logger.debug("Server content is None, skipping processing")
        return
    
    # Check for interruption first
    if hasattr(server_content, 'interrupted') and server_content.interrupted:
        logger.info("Interruption detected from Gemini")
        await websocket.send(json.dumps({
            "type": "interrupted",
            "data": {
                "message": "Response interrupted by user input"
            }
        }))
        session.is_receiving_response = False
        return

    if hasattr(server_content, 'model_turn') and server_content.model_turn:
        session.received_model_response = True
        session.is_receiving_response = True
        for part in server_content.model_turn.parts:
            if part.inline_data:
                audio_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                await websocket.send(json.dumps({
                    "type": "audio",
                    "data": audio_base64
                }))
            elif part.text:
                await websocket.send(json.dumps({
                    "type": "text",
                    "data": part.text
                }))
    
    if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
        await websocket.send(json.dumps({
            "type": "turn_complete"
        }))
        session.received_model_response = False
        session.is_receiving_response = False

async def handle_client(websocket: Any) -> None:
    """Handles a new client connection."""
    session_id = str(id(websocket))
    session = create_session(session_id)
    
    # Set websocket reference in StateAgent for client updates
    if session.state_agent:
        session.state_agent.websocket = websocket
    
    # Start the state agent processor
    from core.tool_handler import start_state_agent_processor
    await start_state_agent_processor(session_id)
    
    try:
        # Create and initialize Gemini session
        async with await create_gemini_session() as gemini_session:
            session.genai_session = gemini_session
            
            # Start the state agent processor
            await start_state_agent_processor(session_id)
            
            # Send ready message to client
            await websocket.send(json.dumps({"ready": True}))
            logger.info(f"New ADK session started: {session_id}")
            
            try:
                # Start message handling
                await handle_messages(websocket, session)
            except Exception as e:
                if "code = 1006" in str(e) or "connection closed abnormally" in str(e).lower():
                    logger.info(f"Browser disconnected or refreshed for session {session_id}")
                    await send_error_message(websocket, {
                        "message": "Connection closed unexpectedly",
                        "action": "Reconnecting...",
                        "error_type": "connection_closed"
                    })
                else:
                    raise
            
    except asyncio.TimeoutError:
        logger.info(f"Session {session_id} timed out - this is normal for long idle periods")
        await send_error_message(websocket, {
            "message": "Session timed out due to inactivity.",
            "action": "You can start a new conversation.",
            "error_type": "timeout"
        })
    except Exception as e:
        error_str = str(e).lower()
        if any(phrase in error_str for phrase in [
            "connection closed", "websocket", "no close frame", 
            "network name is no longer available", "connection reset"
        ]):
            logger.info(f"WebSocket connection closed for session {session_id}: {e}")
            # No need to send error message as connection is already closed
        else:
            logger.error(f"Error in handle_client: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            await send_error_message(websocket, {
                "message": "An unexpected error occurred.",
                "action": "Please try again.",
                "error_type": "general"
            })
    finally:
        # Always ensure cleanup happens
        await cleanup_session(session, session_id) 