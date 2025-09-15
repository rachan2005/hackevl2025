"""
Music and Video Recommendation Agent

This module provides intelligent music and video recommendations based on:
- User intent detection
- Behavioral analysis (emotion, mood, activity)
- YouTube and Spotify API integration
- Personalized preferences
"""

from .music_recommendation_agent import MusicRecommendationAgent
from .youtube_client import YouTubeClient
from .spotify_client import SpotifyClient
from .intent_detector import IntentDetector
from .recommendation_engine import RecommendationEngine

__all__ = [
    'MusicRecommendationAgent',
    'YouTubeClient', 
    'SpotifyClient',
    'IntentDetector',
    'RecommendationEngine'
]
