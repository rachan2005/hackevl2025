#!/usr/bin/env python3
"""
Simple test of the API server functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the behavioral analyzer to the path
sys.path.append(str(Path(__file__).parent / "behavioral_analyzer"))

from behavioral_analyzer.api_server import BehavioralAPIServer

async def test_api_server():
    """Test the API server"""
    print("🧪 Testing API Server...")
    
    try:
        # Create API server
        server = BehavioralAPIServer(port=8084)
        
        # Start server
        await server.start()
        print("✅ API server started on port 8084")
        
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
        
        # Stop server
        await server.stop()
        print("✅ API server stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing API server: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_api_server())
    if success:
        print("🎉 API server test passed!")
    else:
        print("💥 API server test failed!")
        sys.exit(1)
