
"""
HTTP API server for external systems to interact with the ADK interview system
"""

import logging
import json
import asyncio
from aiohttp import web, web_request
from aiohttp.web_response import Response
from core.session import get_session

logger = logging.getLogger(__name__)

async def add_behavioral_feature(session_id: str, data: dict) -> dict:
    """Add behavioral feature to a session"""
    try:
        session = get_session(session_id)
        if not session:
            return {
                "success": False,
                "error": f"Session {session_id} not found"
            }
        
        # Add behavioral feature to the session's shared state
        if "behavioral_features" not in session.shared_state:
            session.shared_state["behavioral_features"] = []
        
        feature = {
            "timestamp": data.get("timestamp", asyncio.get_event_loop().time()),
            "feature_type": data.get("feature_type", "unknown"),
            "value": data.get("value"),
            "confidence": data.get("confidence", 0.0),
            "description": data.get("description", "")
        }
        
        session.shared_state["behavioral_features"].append(feature)
        
        return {
            "success": True,
            "message": "Behavioral feature added successfully",
            "feature": feature
        }
    
    except Exception as e:
        logger.error(f"Error adding behavioral feature: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to add behavioral feature: {str(e)}"
        }

async def get_session_insights(session_id: str) -> dict:
    """Get insights for a session"""
    try:
        session = get_session(session_id)
        if not session:
            return {
                "success": False,
                "error": f"Session {session_id} not found"
            }
        
        # Get insights from the state agent if available
        insights = {}
        if hasattr(session, 'state_agent') and session.state_agent:
            insights = session.state_agent.get_insights_summary()
        
        return {
            "success": True,
            "session_id": session_id,
            "insights": insights,
            "behavioral_features": session.shared_state.get("behavioral_features", []),
            "enriched_qa_pairs": session.shared_state.get("enriched_qa_pairs", [])
        }
    
    except Exception as e:
        logger.error(f"Error getting session insights: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get session insights: {str(e)}"
        }

async def get_conversation_context(session_id: str) -> dict:
    """Get conversation context for a session"""
    try:
        session = get_session(session_id)
        if not session:
            return {
                "success": False,
                "error": f"Session {session_id} not found"
            }
        
        # Get conversation context from the websocket handler
        from core.websocket_handler import get_conversation_context_for_ai
        context = get_conversation_context_for_ai(session)
        
        return {
            "success": True,
            "session_id": session_id,
            "context": context,
            "conversation_state": session.shared_state.get("conversation_state", {}),
            "enriched_qa_pairs": session.shared_state.get("enriched_qa_pairs", [])
        }
    
    except Exception as e:
        logger.error(f"Error getting conversation context: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get conversation context: {str(e)}"
        }

async def add_behavioral_feature_handler(request: web_request.Request) -> Response:
    """HTTP handler for adding behavioral features"""
    try:
        data = await request.json()
        session_id = request.match_info.get('session_id')
        
        if not session_id:
            return web.json_response({
                "success": False,
                "error": "Session ID is required"
            }, status=400)
        
        result = await add_behavioral_feature(session_id, data)
        
        if result["success"]:
            return web.json_response(result)
        else:
            return web.json_response(result, status=400)
    
    except json.JSONDecodeError:
        return web.json_response({
            "success": False,
            "error": "Invalid JSON in request body"
        }, status=400)
    except Exception as e:
        logger.error(f"Error in add_behavioral_feature_handler: {str(e)}")
        return web.json_response({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }, status=500)

async def get_insights_handler(request: web_request.Request) -> Response:
    """HTTP handler for getting session insights"""
    try:
        session_id = request.match_info.get('session_id')
        
        if not session_id:
            return web.json_response({
                "success": False,
                "error": "Session ID is required"
            }, status=400)
        
        result = await get_session_insights(session_id)
        
        if result["success"]:
            return web.json_response(result)
        else:
            return web.json_response(result, status=404)
    
    except Exception as e:
        logger.error(f"Error in get_insights_handler: {str(e)}")
        return web.json_response({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }, status=500)

async def get_context_handler(request: web_request.Request) -> Response:
    """HTTP handler for getting conversation context"""
    try:
        session_id = request.match_info.get('session_id')
        
        if not session_id:
            return web.json_response({
                "success": False,
                "error": "Session ID is required"
            }, status=400)
        
        result = await get_conversation_context(session_id)
        
        if result["success"]:
            return web.json_response(result)
        else:
            return web.json_response(result, status=404)
    
    except Exception as e:
        logger.error(f"Error in get_context_handler: {str(e)}")
        return web.json_response({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }, status=500)

async def health_check_handler(request: web_request.Request) -> Response:
    """Health check endpoint"""
    return web.json_response({
        "status": "healthy",
        "service": "ADK Interview System API",
        "timestamp": asyncio.get_event_loop().time()
    })

def create_api_app() -> web.Application:
    """Create the HTTP API application"""
    app = web.Application()
    
    # Add CORS middleware
    async def cors_handler(request, handler):
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    app.middlewares.append(cors_handler)
    
    # Add routes
    app.router.add_post('/api/sessions/{session_id}/behavioral-features', add_behavioral_feature_handler)
    app.router.add_get('/api/sessions/{session_id}/insights', get_insights_handler)
    app.router.add_get('/api/sessions/{session_id}/context', get_context_handler)
    app.router.add_get('/health', health_check_handler)
    
    return app

async def start_api_server(port: int = 8082):
    """Start the HTTP API server"""
    app = create_api_app()
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"HTTP API server started on port {port}")
    logger.info("Available endpoints:")
    logger.info("  POST /api/sessions/{session_id}/behavioral-features")
    logger.info("  GET  /api/sessions/{session_id}/insights")
    logger.info("  GET  /api/sessions/{session_id}/context")
    logger.info("  GET  /health")
    
    return runner
