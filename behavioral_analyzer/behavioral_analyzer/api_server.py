"""
API Server for Behavioral Analyzer
Exposes real-time behavioral data via HTTP API for integration with ADK voice agent
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from aiohttp import web, web_request
from aiohttp.web_response import Response
import threading
from dataclasses import asdict

logger = logging.getLogger(__name__)

class BehavioralAPIServer:
    """HTTP API server for exposing behavioral analysis data"""
    
    def __init__(self, port: int = 8083):
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        self.current_data = {}
        self.session_start_time = time.time()
        self.data_lock = threading.Lock()
        
    def update_data(self, data: Dict[str, Any]):
        """Update current behavioral data (thread-safe)"""
        with self.data_lock:
            self.current_data = {
                **data,
                'timestamp': time.time(),
                'session_duration': time.time() - self.session_start_time
            }
    
    async def get_current_data(self) -> Dict[str, Any]:
        """Get current behavioral data (thread-safe)"""
        with self.data_lock:
            return self.current_data.copy()
    
    async def health_check_handler(self, request: web_request.Request) -> Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "service": "Behavioral Analyzer API",
            "timestamp": time.time(),
            "session_duration": time.time() - self.session_start_time
        })
    
    async def get_behavioral_data_handler(self, request: web_request.Request) -> Response:
        """Get current behavioral analysis data"""
        try:
            data = await self.get_current_data()
            return web.json_response({
                "success": True,
                "data": data,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"Error getting behavioral data: {str(e)}")
            return web.json_response({
                "success": False,
                "error": f"Failed to get behavioral data: {str(e)}"
            }, status=500)
    
    async def get_emotion_handler(self, request: web_request.Request) -> Response:
        """Get current emotion data"""
        try:
            data = await self.get_current_data()
            emotion_data = {
                "emotion": data.get("emotion", "unknown"),
                "confidence": data.get("emotion_confidence", 0.0),
                "timestamp": data.get("timestamp", time.time())
            }
            return web.json_response({
                "success": True,
                "data": emotion_data
            })
        except Exception as e:
            logger.error(f"Error getting emotion data: {str(e)}")
            return web.json_response({
                "success": False,
                "error": f"Failed to get emotion data: {str(e)}"
            }, status=500)
    
    async def get_attention_handler(self, request: web_request.Request) -> Response:
        """Get current attention data"""
        try:
            data = await self.get_current_data()
            attention_data = {
                "attention_level": data.get("attention", "unknown"),
                "attention_score": data.get("attention_score", 0.0),
                "face_orientation": data.get("face_orientation", "unknown"),
                "timestamp": data.get("timestamp", time.time())
            }
            return web.json_response({
                "success": True,
                "data": attention_data
            })
        except Exception as e:
            logger.error(f"Error getting attention data: {str(e)}")
            return web.json_response({
                "success": False,
                "error": f"Failed to get attention data: {str(e)}"
            }, status=500)
    
    async def get_fatigue_handler(self, request: web_request.Request) -> Response:
        """Get current fatigue data"""
        try:
            data = await self.get_current_data()
            fatigue_data = {
                "fatigue_level": data.get("fatigue", "normal"),
                "blink_rate": data.get("blink_rate", 0.0),
                "eye_openness": data.get("eye_openness", 0.0),
                "timestamp": data.get("timestamp", time.time())
            }
            return web.json_response({
                "success": True,
                "data": fatigue_data
            })
        except Exception as e:
            logger.error(f"Error getting fatigue data: {str(e)}")
            return web.json_response({
                "success": False,
                "error": f"Failed to get fatigue data: {str(e)}"
            }, status=500)
    
    async def get_sentiment_handler(self, request: web_request.Request) -> Response:
        """Get current sentiment data"""
        try:
            data = await self.get_current_data()
            sentiment_data = {
                "sentiment": data.get("sentiment", "neutral"),
                "sentiment_score": data.get("sentiment_score", 0.0),
                "transcription": data.get("transcription", ""),
                "timestamp": data.get("timestamp", time.time())
            }
            return web.json_response({
                "success": True,
                "data": sentiment_data
            })
        except Exception as e:
            logger.error(f"Error getting sentiment data: {str(e)}")
            return web.json_response({
                "success": False,
                "error": f"Failed to get sentiment data: {str(e)}"
            }, status=500)
    
    async def get_person_tracking_handler(self, request: web_request.Request) -> Response:
        """Get current person tracking data"""
        try:
            data = await self.get_current_data()
            person_data = {
                "person_count": data.get("person_count", 0),
                "main_person_id": data.get("main_person_id"),
                "main_person_confidence": data.get("main_person_confidence", 0.0),
                "timestamp": data.get("timestamp", time.time())
            }
            return web.json_response({
                "success": True,
                "data": person_data
            })
        except Exception as e:
            logger.error(f"Error getting person tracking data: {str(e)}")
            return web.json_response({
                "success": False,
                "error": f"Failed to get person tracking data: {str(e)}"
            }, status=500)
    
    def create_app(self) -> web.Application:
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
        app.router.add_get('/health', self.health_check_handler)
        app.router.add_get('/api/behavioral-data', self.get_behavioral_data_handler)
        app.router.add_get('/api/emotion', self.get_emotion_handler)
        app.router.add_get('/api/attention', self.get_attention_handler)
        app.router.add_get('/api/fatigue', self.get_fatigue_handler)
        app.router.add_get('/api/sentiment', self.get_sentiment_handler)
        app.router.add_get('/api/person-tracking', self.get_person_tracking_handler)
        
        return app
    
    async def start(self):
        """Start the API server"""
        self.app = self.create_app()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()
        
        logger.info(f"Behavioral Analyzer API server started on port {self.port}")
        logger.info("Available endpoints:")
        logger.info("  GET /health")
        logger.info("  GET /api/behavioral-data")
        logger.info("  GET /api/emotion")
        logger.info("  GET /api/attention")
        logger.info("  GET /api/fatigue")
        logger.info("  GET /api/sentiment")
        logger.info("  GET /api/person-tracking")
    
    async def stop(self):
        """Stop the API server"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Behavioral Analyzer API server stopped")

# Global API server instance
api_server = None

def get_api_server(port: int = 8083) -> BehavioralAPIServer:
    """Get or create the global API server instance"""
    global api_server
    if api_server is None:
        api_server = BehavioralAPIServer(port)
    return api_server

async def start_api_server(port: int = 8083):
    """Start the API server (for use in main analyzer)"""
    server = get_api_server(port)
    await server.start()
    return server

async def stop_api_server():
    """Stop the API server"""
    global api_server
    if api_server:
        await api_server.stop()
        api_server = None
