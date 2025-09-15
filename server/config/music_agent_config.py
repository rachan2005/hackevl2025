"""
Configuration for Music Recommendation Agent

Contains API keys and settings for YouTube and Spotify integration.
"""

import os
from typing import Optional

class MusicAgentConfig:
    """Configuration for the music recommendation agent."""
    
    def __init__(self):
        # YouTube Data API v3
        self.youtube_api_key: Optional[str] = os.getenv('YOUTUBE_API_KEY')
        
        # Spotify Web API
        self.spotify_client_id: Optional[str] = os.getenv('SPOTIFY_CLIENT_ID')
        self.spotify_client_secret: Optional[str] = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        # Default settings
        self.max_recommendations: int = 10
        self.max_youtube_results: int = 15
        self.max_spotify_tracks: int = 20
        self.max_spotify_playlists: int = 10
        
        # Recommendation settings
        self.min_confidence_threshold: float = 0.3
        self.enable_trending_content: bool = True
        self.enable_behavioral_analysis: bool = True
        
        # API rate limiting
        self.youtube_rate_limit: int = 100  # requests per 100 seconds
        self.spotify_rate_limit: int = 1000  # requests per hour
        
    def is_configured(self) -> bool:
        """Check if all required API keys are configured."""
        return all([
            self.youtube_api_key,
            self.spotify_client_id,
            self.spotify_client_secret
        ])
    
    def get_missing_configs(self) -> list[str]:
        """Get list of missing configuration items."""
        missing = []
        
        if not self.youtube_api_key:
            missing.append("YOUTUBE_API_KEY")
        if not self.spotify_client_id:
            missing.append("SPOTIFY_CLIENT_ID")
        if not self.spotify_client_secret:
            missing.append("SPOTIFY_CLIENT_SECRET")
        
        return missing

# Global configuration instance
music_config = MusicAgentConfig()
