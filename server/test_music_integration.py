#!/usr/bin/env python3
"""
Test the music agent integration with the conversational system
"""

import asyncio
import aiohttp
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_music_tools():
    """Test the music recommendation tools via the API."""
    
    print("🎵 Testing Music Agent Integration")
    print("=" * 50)
    
    # Test music recommendations tool
    print("1. Testing get_music_recommendations tool...")
    
    tool_params = {
        "user_input": "I'm feeling happy, play some upbeat music",
        "behavioral_data": {
            "emotion": "happy",
            "attention": "focused", 
            "fatigue": "low",
            "sentiment": 0.7
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test the tool execution endpoint
            async with session.post(
                "http://localhost:8082/api/execute-tool",
                json={
                    "tool_name": "get_music_recommendations",
                    "params": tool_params
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Music recommendations tool working!")
                    print(f"📊 Intent: {result.get('intent_type', 'unknown')}")
                    print(f"😊 Mood: {result.get('mood', 'unknown')}")
                    print(f"📈 Confidence: {result.get('confidence', 0.0):.2f}")
                    print(f"🎵 Music count: {result.get('music_count', 0)}")
                    print(f"📺 Video count: {result.get('video_count', 0)}")
                    
                    # Show a snippet of recommendations
                    recommendations = result.get('recommendations', '')
                    if recommendations:
                        lines = recommendations.split('\n')[:10]  # First 10 lines
                        print("\n📝 Recommendations preview:")
                        for line in lines:
                            if line.strip():
                                print(f"   {line}")
                else:
                    print(f"❌ Tool execution failed: {response.status}")
                    error_text = await response.text()
                    print(f"Error: {error_text}")
    except Exception as e:
        print(f"❌ Error testing music recommendations: {e}")
    
    print()
    
    # Test trending content tool
    print("2. Testing get_trending_content tool...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8082/api/execute-tool",
                json={
                    "tool_name": "get_trending_content",
                    "params": {"content_type": "both"}
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Trending content tool working!")
                    print(f"📊 Content type: {result.get('content_type', 'unknown')}")
                    print(f"🎵 Music count: {result.get('music_count', 0)}")
                    print(f"📺 Video count: {result.get('video_count', 0)}")
                else:
                    print(f"❌ Tool execution failed: {response.status}")
                    error_text = await response.text()
                    print(f"Error: {error_text}")
    except Exception as e:
        print(f"❌ Error testing trending content: {e}")
    
    print()
    print("🎉 Music agent integration testing completed!")

async def test_behavioral_analyzer_integration():
    """Test if behavioral analyzer is providing data for music recommendations."""
    
    print("🧠 Testing Behavioral Analyzer Integration")
    print("=" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get current behavioral data
            async with session.get("http://localhost:8083/api/behavioral-data") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success'):
                        behavioral_data = data.get('data', {})
                        print("✅ Behavioral analyzer is running!")
                        print(f"😊 Emotion: {behavioral_data.get('emotion', 'unknown')}")
                        print(f"👁️ Attention: {behavioral_data.get('attention', 'unknown')}")
                        print(f"😴 Fatigue: {behavioral_data.get('fatigue', 'unknown')}")
                        print(f"💬 Recent speech: {behavioral_data.get('transcription', 'none')}")
                        print(f"📊 Sentiment: {behavioral_data.get('sentiment', 0.0)}")
                        
                        # Test music recommendations with real behavioral data
                        print("\n🎵 Testing music recommendations with real behavioral data...")
                        
                        tool_params = {
                            "user_input": "recommend something based on how I'm feeling",
                            "behavioral_data": behavioral_data
                        }
                        
                        async with session.post(
                            "http://localhost:8082/api/execute-tool",
                            json={
                                "tool_name": "get_music_recommendations",
                                "params": tool_params
                            }
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                print("✅ Music recommendations with behavioral data working!")
                                print(f"📊 Intent: {result.get('intent_type', 'unknown')}")
                                print(f"😊 Mood: {result.get('mood', 'unknown')}")
                                print(f"📈 Confidence: {result.get('confidence', 0.0):.2f}")
                            else:
                                print(f"❌ Music recommendations failed: {response.status}")
                    else:
                        print("❌ Behavioral analyzer not providing data")
                else:
                    print(f"❌ Behavioral analyzer not responding: {response.status}")
    except Exception as e:
        print(f"❌ Error testing behavioral integration: {e}")

if __name__ == "__main__":
    async def main():
        await test_music_tools()
        print()
        await test_behavioral_analyzer_integration()
    
    asyncio.run(main())
