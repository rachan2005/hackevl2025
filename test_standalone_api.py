#!/usr/bin/env python3
"""
Standalone test of the API server without importing the full behavioral analyzer
"""

import asyncio
import aiohttp
import time
import threading
from typing import Dict, Any, Optional
from aiohttp import web, web_request
from aiohttp.web_response import Response

class SimpleAPIServer:
    """Simple API server for testing"""
    
    def __init__(self, port: int = 8084):
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
            "service": "Test Behavioral Analyzer API",
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
            return web.json_response({
                "success": False,
                "error": f"Failed to get behavioral data: {str(e)}"
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
        
        return app
    
    async def start(self):
        """Start the API server"""
        self.app = self.create_app()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await self.site.start()
        
        print(f"✅ Test API server started on port {self.port}")
        print("Available endpoints:")
        print("  GET /health")
        print("  GET /api/behavioral-data")
    
    async def stop(self):
        """Stop the API server"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        print("✅ Test API server stopped")

async def test_api_server():
    """Test the API server"""
    print("🧪 Testing Standalone API Server...")
    
    try:
        # Create API server
        server = SimpleAPIServer(port=8084)
        
        # Start server
        await server.start()
        
        # Test updating data
        test_data = {
            "emotion": "happy",
            "attention": "focused",
            "fatigue": "normal",
            "transcription": "Hello world",
            "sentiment": "positive"
        }
        server.update_data(test_data)
        print("✅ Test data updated")
        
        # Get current data
        current_data = await server.get_current_data()
        print(f"✅ Current data: {current_data}")
        
        # Test HTTP endpoints
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get("http://localhost:8084/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check: {data}")
                else:
                    print(f"❌ Health check failed: {response.status}")
            
            # Test behavioral data endpoint
            async with session.get("http://localhost:8084/api/behavioral-data") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Behavioral data: {data}")
                else:
                    print(f"❌ Behavioral data failed: {response.status}")
        
        # Stop server
        await server.stop()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing API server: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_api_server())
    if success:
        print("🎉 Standalone API server test passed!")
    else:
        print("💥 Standalone API server test failed!")
        exit(1)
