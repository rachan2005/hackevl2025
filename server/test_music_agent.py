#!/usr/bin/env python3
"""
Test script for the Music Recommendation Agent
"""

import asyncio
import os
import sys
import logging
from core.music_agent import MusicRecommendationAgent
from config.music_agent_config import music_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_music_agent():
    """Test the music recommendation agent with sample requests."""
    
    print("🎵 Testing Music Recommendation Agent")
    print("=" * 50)
    
    # Check configuration
    if not music_config.is_configured():
        missing = music_config.get_missing_configs()
        print(f"❌ Music agent not configured. Missing: {', '.join(missing)}")
        return False
    
    print("✅ Configuration check passed")
    print(f"📺 YouTube API Key: {music_config.youtube_api_key[:20]}...")
    print(f"🎵 Spotify Client ID: {music_config.spotify_client_id[:20]}...")
    print()
    
    # Initialize music agent
    try:
        music_agent = MusicRecommendationAgent(
            youtube_api_key=music_config.youtube_api_key,
            spotify_client_id=music_config.spotify_client_id,
            spotify_client_secret=music_config.spotify_client_secret
        )
        print("✅ Music agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize music agent: {e}")
        return False
    
    # Test cases
    test_cases = [
        {
            "input": "I'm feeling happy, play some upbeat music",
            "description": "Happy mood music request"
        },
        {
            "input": "I need music for working out",
            "description": "Activity-based music request"
        },
        {
            "input": "Show me trending videos",
            "description": "Trending content request"
        },
        {
            "input": "Play something by The Beatles",
            "description": "Artist-specific request"
        }
    ]
    
    # Sample behavioral data
    sample_behavioral_data = {
        "emotion": "happy",
        "attention": "focused",
        "fatigue": "low",
        "sentiment": 0.7
    }
    
    print("🧪 Running test cases...")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['description']}")
        print(f"Input: '{test_case['input']}'")
        print("-" * 40)
        
        try:
            # Get recommendations
            recommendations = await music_agent.get_recommendations(
                test_case['input'], 
                sample_behavioral_data
            )
            
            # Display results
            print(f"🎯 Intent: {recommendations.intent_analysis.intent_type.value}")
            print(f"😊 Mood: {recommendations.intent_analysis.mood.value}")
            print(f"📊 Confidence: {recommendations.intent_analysis.confidence:.2f}")
            print(f"🎵 Music recommendations: {len(recommendations.music_recommendations)}")
            print(f"📺 Video recommendations: {len(recommendations.video_recommendations)}")
            print(f"📝 Insights: {recommendations.behavioral_insights}")
            
            # Show top recommendations
            if recommendations.combined_recommendations:
                print("\n🏆 Top 3 Recommendations:")
                for j, rec in enumerate(recommendations.combined_recommendations[:3], 1):
                    source_emoji = "🎵" if rec.source == "spotify" else "📺"
                    print(f"  {j}. {source_emoji} {rec.title}")
                    print(f"     {rec.reason}")
                    print(f"     🔗 {rec.url}")
            
            print("✅ Test passed")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            logger.error(f"Test {i} failed: {e}", exc_info=True)
        
        print()
    
    # Test trending content
    print("🔥 Testing trending content...")
    try:
        trending = await music_agent.get_trending_content("both")
        print(f"📈 Trending music: {len(trending.music_recommendations)}")
        print(f"📈 Trending videos: {len(trending.video_recommendations)}")
        print("✅ Trending content test passed")
    except Exception as e:
        print(f"❌ Trending content test failed: {e}")
    
    print()
    print("🎉 Music agent testing completed!")
    return True

async def test_api_connectivity():
    """Test API connectivity separately."""
    print("🔌 Testing API connectivity...")
    
    # Test YouTube API
    try:
        from core.music_agent.youtube_client import YouTubeClient
        async with YouTubeClient(music_config.youtube_api_key) as youtube:
            videos = await youtube.search_videos("test", max_results=1)
            print(f"✅ YouTube API: Found {len(videos)} videos")
    except Exception as e:
        print(f"❌ YouTube API failed: {e}")
    
    # Test Spotify API
    try:
        from core.music_agent.spotify_client import SpotifyClient
        async with SpotifyClient(music_config.spotify_client_id, music_config.spotify_client_secret) as spotify:
            tracks = await spotify.search_tracks("test", limit=1)
            print(f"✅ Spotify API: Found {len(tracks)} tracks")
    except Exception as e:
        print(f"❌ Spotify API failed: {e}")

if __name__ == "__main__":
    async def main():
        print("🚀 Starting Music Agent Tests")
        print()
        
        # Test API connectivity first
        await test_api_connectivity()
        print()
        
        # Test full music agent
        await test_music_agent()
    
    asyncio.run(main())
