"""
Main Music Recommendation Agent

Orchestrates the entire music and video recommendation system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from .intent_detector import IntentDetector, MusicIntent
from .youtube_client import YouTubeClient, YouTubeVideo
from .spotify_client import SpotifyClient, SpotifyTrack, SpotifyPlaylist
from .recommendation_engine import RecommendationEngine, RecommendationSet

logger = logging.getLogger(__name__)

class MusicRecommendationAgent:
    """Main agent for music and video recommendations."""
    
    def __init__(self, youtube_api_key: str, spotify_client_id: str, spotify_client_secret: str):
        self.youtube_api_key = youtube_api_key
        self.spotify_client_id = spotify_client_id
        self.spotify_client_secret = spotify_client_secret
        
        # Initialize components
        self.intent_detector = IntentDetector()
        self.recommendation_engine = RecommendationEngine()
        
        # API clients (will be initialized when needed)
        self.youtube_client: Optional[YouTubeClient] = None
        self.spotify_client: Optional[SpotifyClient] = None

    async def get_recommendations(self, user_input: str, 
                                behavioral_data: Optional[Dict] = None) -> RecommendationSet:
        """
        Get music and video recommendations based on user input and behavioral data.
        
        Args:
            user_input: User's request for music/video
            behavioral_data: Current behavioral analysis data
            
        Returns:
            RecommendationSet with categorized recommendations
        """
        try:
            # Step 1: Detect user intent
            intent = self.intent_detector.detect_intent(user_input, behavioral_data)
            logger.info(f"🎯 Detected intent: {intent.intent_type.value}, mood: {intent.mood.value}, confidence: {intent.confidence:.2f}")
            
            # Step 2: Search for content based on intent
            youtube_videos = []
            spotify_tracks = []
            spotify_playlists = []
            
            # Search YouTube if needed
            if intent.intent_type in [intent.intent_type.VIDEO, intent.intent_type.BOTH]:
                youtube_videos = await self._search_youtube(intent)
            
            # Search Spotify if needed
            if intent.intent_type in [intent.intent_type.MUSIC, intent.intent_type.BOTH]:
                spotify_tracks, spotify_playlists = await self._search_spotify(intent)
            
            # Step 3: Generate intelligent recommendations
            recommendations = await self.recommendation_engine.generate_recommendations(
                intent=intent,
                behavioral_data=behavioral_data,
                youtube_videos=youtube_videos,
                spotify_tracks=spotify_tracks,
                spotify_playlists=spotify_playlists
            )
            
            logger.info(f"🎵 Generated {len(recommendations.music_recommendations)} music and {len(recommendations.video_recommendations)} video recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            # Return empty recommendations on error
            return RecommendationSet(
                music_recommendations=[],
                video_recommendations=[],
                combined_recommendations=[],
                intent_analysis=intent if 'intent' in locals() else None,
                behavioral_insights="Sorry, I encountered an error while getting recommendations."
            )

    async def _search_youtube(self, intent: MusicIntent) -> List[YouTubeVideo]:
        """Search YouTube for relevant videos."""
        try:
            async with YouTubeClient(self.youtube_api_key) as youtube:
                videos = []
                
                # Build search query based on intent
                query_parts = []
                
                if intent.artist:
                    query_parts.append(intent.artist)
                if intent.song:
                    query_parts.append(intent.song)
                if intent.genre:
                    query_parts.append(intent.genre)
                if intent.activity:
                    query_parts.append(intent.activity)
                
                # Add mood-based keywords
                if intent.mood.value != 'unknown':
                    mood_keywords = self.recommendation_engine.mood_mappings.get(intent.mood, {}).get('keywords', [])
                    if mood_keywords:
                        query_parts.append(mood_keywords[0])  # Use first keyword
                
                # Create search query
                if query_parts:
                    query = " ".join(query_parts)
                else:
                    # Fallback queries based on mood
                    if intent.mood.value == 'happy':
                        query = "happy music video"
                    elif intent.mood.value == 'sad':
                        query = "emotional music video"
                    elif intent.mood.value == 'energetic':
                        query = "energetic music video"
                    elif intent.mood.value == 'calm':
                        query = "calm music video"
                    else:
                        query = "music video"
                
                # Search for music videos
                videos = await youtube.search_music_videos(query, max_results=15)
                
                # If no music videos found, try general search
                if not videos:
                    videos = await youtube.search_videos(query, max_results=10)
                
                logger.info(f"🔍 Found {len(videos)} YouTube videos for query: '{query}'")
                return videos
                
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
            return []

    async def _search_spotify(self, intent: MusicIntent) -> tuple[List[SpotifyTrack], List[SpotifyPlaylist]]:
        """Search Spotify for relevant tracks and playlists."""
        try:
            async with SpotifyClient(self.spotify_client_id, self.spotify_client_secret) as spotify:
                tracks = []
                playlists = []
                
                # Build search queries
                queries = []
                
                # Artist-specific search
                if intent.artist:
                    queries.append(intent.artist)
                
                # Song-specific search
                if intent.song:
                    queries.append(intent.song)
                
                # Genre-specific search
                if intent.genre:
                    queries.append(intent.genre)
                
                # Mood-based search
                if intent.mood.value != 'unknown':
                    mood_config = self.recommendation_engine.mood_mappings.get(intent.mood, {})
                    if mood_config.get('genres'):
                        queries.append(mood_config['genres'][0])  # Use first genre
                
                # Activity-based search
                if intent.activity:
                    activity_config = self.recommendation_engine.activity_mappings.get(intent.activity, {})
                    if activity_config.get('keywords'):
                        queries.append(activity_config['keywords'][0])  # Use first keyword
                
                # If no specific queries, use mood-based fallback
                if not queries:
                    if intent.mood.value == 'happy':
                        queries = ['happy music', 'upbeat songs']
                    elif intent.mood.value == 'sad':
                        queries = ['sad songs', 'emotional music']
                    elif intent.mood.value == 'energetic':
                        queries = ['energetic music', 'high energy songs']
                    elif intent.mood.value == 'calm':
                        queries = ['calm music', 'peaceful songs']
                    else:
                        queries = ['popular music', 'trending songs']
                
                # Search for tracks
                for query in queries[:3]:  # Limit to 3 queries
                    track_results = await spotify.search_tracks(query, limit=10)
                    tracks.extend(track_results)
                
                # Search for playlists
                for query in queries[:2]:  # Limit to 2 queries for playlists
                    playlist_results = await spotify.search_playlists(query, limit=5)
                    playlists.extend(playlist_results)
                
                # Remove duplicates
                tracks = list({track.track_id: track for track in tracks}.values())
                playlists = list({playlist.playlist_id: playlist for playlist in playlists}.values())
                
                logger.info(f"🎵 Found {len(tracks)} Spotify tracks and {len(playlists)} playlists")
                return tracks, playlists
                
        except Exception as e:
            logger.error(f"Error searching Spotify: {e}")
            return [], []

    async def get_trending_content(self, content_type: str = "both") -> RecommendationSet:
        """
        Get trending music and videos.
        
        Args:
            content_type: "music", "video", or "both"
            
        Returns:
            RecommendationSet with trending content
        """
        try:
            youtube_videos = []
            spotify_tracks = []
            spotify_playlists = []
            
            # Get trending YouTube videos
            if content_type in ["video", "both"]:
                async with YouTubeClient(self.youtube_api_key) as youtube:
                    youtube_videos = await youtube.get_trending_videos(max_results=10)
            
            # Get trending Spotify content (using popular searches)
            if content_type in ["music", "both"]:
                async with SpotifyClient(self.spotify_client_id, self.spotify_client_secret) as spotify:
                    # Search for popular terms
                    popular_terms = ["trending", "popular", "hits", "top songs", "new music"]
                    for term in popular_terms:
                        track_results = await spotify.search_tracks(term, limit=5)
                        spotify_tracks.extend(track_results)
                        
                        playlist_results = await spotify.search_playlists(term, limit=3)
                        spotify_playlists.extend(playlist_results)
            
            # Create a generic intent for trending content
            from .intent_detector import IntentType, MoodType
            intent = MusicIntent(
                intent_type=IntentType.BOTH if content_type == "both" else 
                          IntentType.MUSIC if content_type == "music" else 
                          IntentType.VIDEO,
                mood=MoodType.UNKNOWN,
                confidence=0.8
            )
            
            # Generate recommendations
            recommendations = await self.recommendation_engine.generate_recommendations(
                intent=intent,
                behavioral_data=None,
                youtube_videos=youtube_videos,
                spotify_tracks=spotify_tracks,
                spotify_playlists=spotify_playlists
            )
            
            recommendations.behavioral_insights = "Here are the trending music and videos right now!"
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting trending content: {e}")
            return RecommendationSet(
                music_recommendations=[],
                video_recommendations=[],
                combined_recommendations=[],
                intent_analysis=None,
                behavioral_insights="Sorry, I couldn't get trending content right now."
            )

    def format_recommendations_for_ai(self, recommendations: RecommendationSet) -> str:
        """
        Format recommendations for inclusion in AI context.
        
        Args:
            recommendations: RecommendationSet to format
            
        Returns:
            Formatted string for AI context
        """
        if not recommendations.combined_recommendations:
            return "No recommendations available at the moment."
        
        formatted = []
        formatted.append("🎵 MUSIC & VIDEO RECOMMENDATIONS:")
        formatted.append(f"📊 Analysis: {recommendations.behavioral_insights}")
        formatted.append("")
        
        # Add top recommendations
        top_recs = recommendations.combined_recommendations[:5]
        for i, rec in enumerate(top_recs, 1):
            source_emoji = "🎵" if rec.source == "spotify" else "📺"
            formatted.append(f"{i}. {source_emoji} {rec.title}")
            formatted.append(f"   {rec.reason}")
            formatted.append(f"   🔗 {rec.url}")
            formatted.append("")
        
        # Add summary
        music_count = len(recommendations.music_recommendations)
        video_count = len(recommendations.video_recommendations)
        formatted.append(f"📈 Found {music_count} music and {video_count} video recommendations")
        
        return "\n".join(formatted)
