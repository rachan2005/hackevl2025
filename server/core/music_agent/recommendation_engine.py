"""
Recommendation Engine for Music and Video

Combines behavioral data, user intent, and API responses to provide intelligent recommendations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .intent_detector import MusicIntent, IntentType, MoodType
from .youtube_client import YouTubeVideo
from .spotify_client import SpotifyTrack, SpotifyPlaylist

logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    title: str
    description: str
    url: str
    source: str  # 'youtube' or 'spotify'
    confidence: float
    reason: str
    metadata: Dict[str, Any]

@dataclass
class RecommendationSet:
    music_recommendations: List[Recommendation]
    video_recommendations: List[Recommendation]
    combined_recommendations: List[Recommendation]
    intent_analysis: MusicIntent
    behavioral_insights: str

class RecommendationEngine:
    """Engine for generating intelligent music and video recommendations."""
    
    def __init__(self):
        # Mood-based recommendation mappings
        self.mood_mappings = {
            MoodType.HAPPY: {
                'genres': ['pop', 'dance', 'electronic', 'funk'],
                'keywords': ['upbeat', 'happy', 'energetic', 'positive'],
                'energy_level': 0.8,
                'valence': 0.9
            },
            MoodType.SAD: {
                'genres': ['blues', 'indie', 'alternative', 'folk'],
                'keywords': ['melancholic', 'emotional', 'deep', 'reflective'],
                'energy_level': 0.3,
                'valence': 0.2
            },
            MoodType.ENERGETIC: {
                'genres': ['rock', 'electronic', 'hip hop', 'metal'],
                'keywords': ['high energy', 'intense', 'powerful', 'aggressive'],
                'energy_level': 0.9,
                'valence': 0.7
            },
            MoodType.CALM: {
                'genres': ['ambient', 'classical', 'jazz', 'acoustic'],
                'keywords': ['peaceful', 'calm', 'serene', 'gentle'],
                'energy_level': 0.2,
                'valence': 0.6
            },
            MoodType.FOCUSED: {
                'genres': ['classical', 'ambient', 'instrumental', 'lo-fi'],
                'keywords': ['concentration', 'study', 'work', 'instrumental'],
                'energy_level': 0.4,
                'valence': 0.5
            },
            MoodType.RELAXED: {
                'genres': ['jazz', 'blues', 'acoustic', 'folk'],
                'keywords': ['chill', 'relaxed', 'easy', 'comfortable'],
                'energy_level': 0.3,
                'valence': 0.6
            },
            MoodType.EXCITED: {
                'genres': ['pop', 'dance', 'electronic', 'rock'],
                'keywords': ['exciting', 'thrilling', 'energetic', 'upbeat'],
                'energy_level': 0.8,
                'valence': 0.8
            },
            MoodType.MELANCHOLIC: {
                'genres': ['indie', 'alternative', 'folk', 'blues'],
                'keywords': ['nostalgic', 'reflective', 'contemplative', 'deep'],
                'energy_level': 0.4,
                'valence': 0.3
            }
        }
        
        # Activity-based recommendation mappings
        self.activity_mappings = {
            'workout': {
                'genres': ['electronic', 'hip hop', 'rock', 'pop'],
                'keywords': ['workout', 'fitness', 'gym', 'training'],
                'energy_level': 0.9,
                'valence': 0.7
            },
            'study': {
                'genres': ['classical', 'ambient', 'instrumental', 'lo-fi'],
                'keywords': ['study', 'concentration', 'focus', 'instrumental'],
                'energy_level': 0.3,
                'valence': 0.5
            },
            'work': {
                'genres': ['ambient', 'classical', 'jazz', 'instrumental'],
                'keywords': ['work', 'office', 'productivity', 'background'],
                'energy_level': 0.4,
                'valence': 0.6
            },
            'party': {
                'genres': ['dance', 'electronic', 'pop', 'hip hop'],
                'keywords': ['party', 'dance', 'celebration', 'club'],
                'energy_level': 0.9,
                'valence': 0.8
            },
            'relaxation': {
                'genres': ['ambient', 'classical', 'jazz', 'acoustic'],
                'keywords': ['relaxation', 'meditation', 'peaceful', 'calm'],
                'energy_level': 0.2,
                'valence': 0.7
            },
            'driving': {
                'genres': ['rock', 'pop', 'country', 'alternative'],
                'keywords': ['road trip', 'driving', 'travel', 'journey'],
                'energy_level': 0.6,
                'valence': 0.7
            }
        }

    async def generate_recommendations(self, intent: MusicIntent, 
                                     behavioral_data: Optional[Dict] = None,
                                     youtube_videos: List[YouTubeVideo] = None,
                                     spotify_tracks: List[SpotifyTrack] = None,
                                     spotify_playlists: List[SpotifyPlaylist] = None) -> RecommendationSet:
        """
        Generate intelligent recommendations based on intent and available data.
        
        Args:
            intent: Detected user intent
            behavioral_data: Current behavioral analysis data
            youtube_videos: Available YouTube videos
            spotify_tracks: Available Spotify tracks
            spotify_playlists: Available Spotify playlists
            
        Returns:
            RecommendationSet with categorized recommendations
        """
        # Generate behavioral insights
        behavioral_insights = self._generate_behavioral_insights(intent, behavioral_data)
        
        # Generate music recommendations
        music_recommendations = []
        if intent.intent_type in [IntentType.MUSIC, IntentType.BOTH]:
            music_recommendations = await self._generate_music_recommendations(
                intent, behavioral_data, spotify_tracks, spotify_playlists
            )
        
        # Generate video recommendations
        video_recommendations = []
        if intent.intent_type in [IntentType.VIDEO, IntentType.BOTH]:
            video_recommendations = await self._generate_video_recommendations(
                intent, behavioral_data, youtube_videos
            )
        
        # Generate combined recommendations
        combined_recommendations = self._combine_recommendations(
            music_recommendations, video_recommendations, intent
        )
        
        return RecommendationSet(
            music_recommendations=music_recommendations,
            video_recommendations=video_recommendations,
            combined_recommendations=combined_recommendations,
            intent_analysis=intent,
            behavioral_insights=behavioral_insights
        )

    async def _generate_music_recommendations(self, intent: MusicIntent,
                                            behavioral_data: Optional[Dict],
                                            spotify_tracks: List[SpotifyTrack],
                                            spotify_playlists: List[SpotifyPlaylist]) -> List[Recommendation]:
        """Generate music recommendations from Spotify data."""
        recommendations = []
        
        # Add track recommendations
        for track in spotify_tracks[:10]:  # Limit to top 10
            confidence = self._calculate_track_confidence(track, intent, behavioral_data)
            reason = self._generate_track_reason(track, intent, behavioral_data)
            
            recommendations.append(Recommendation(
                title=f"{track.name} - {', '.join(track.artists)}",
                description=f"From album: {track.album}",
                url=track.external_urls.get('spotify', ''),
                source='spotify',
                confidence=confidence,
                reason=reason,
                metadata={
                    'track_id': track.track_id,
                    'duration': track.duration_ms,
                    'popularity': track.popularity,
                    'preview_url': track.preview_url,
                    'explicit': track.explicit
                }
            ))
        
        # Add playlist recommendations
        for playlist in spotify_playlists[:5]:  # Limit to top 5
            confidence = self._calculate_playlist_confidence(playlist, intent, behavioral_data)
            reason = self._generate_playlist_reason(playlist, intent, behavioral_data)
            
            recommendations.append(Recommendation(
                title=playlist.name,
                description=playlist.description or f"Playlist by {playlist.owner}",
                url=playlist.external_urls.get('spotify', ''),
                source='spotify',
                confidence=confidence,
                reason=reason,
                metadata={
                    'playlist_id': playlist.playlist_id,
                    'tracks_count': playlist.tracks_count,
                    'owner': playlist.owner,
                    'images': playlist.images
                }
            ))
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations

    async def _generate_video_recommendations(self, intent: MusicIntent,
                                            behavioral_data: Optional[Dict],
                                            youtube_videos: List[YouTubeVideo]) -> List[Recommendation]:
        """Generate video recommendations from YouTube data."""
        recommendations = []
        
        for video in youtube_videos[:10]:  # Limit to top 10
            confidence = self._calculate_video_confidence(video, intent, behavioral_data)
            reason = self._generate_video_reason(video, intent, behavioral_data)
            
            recommendations.append(Recommendation(
                title=video.title,
                description=video.description[:200] + "..." if len(video.description) > 200 else video.description,
                url=video.video_url,
                source='youtube',
                confidence=confidence,
                reason=reason,
                metadata={
                    'video_id': video.video_id,
                    'channel': video.channel_title,
                    'duration': video.duration,
                    'view_count': video.view_count,
                    'like_count': video.like_count,
                    'published_at': video.published_at,
                    'thumbnail': video.thumbnail_url
                }
            ))
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations

    def _combine_recommendations(self, music_recs: List[Recommendation],
                               video_recs: List[Recommendation],
                               intent: MusicIntent) -> List[Recommendation]:
        """Combine and rank all recommendations."""
        combined = music_recs + video_recs
        
        # Adjust confidence based on intent type
        for rec in combined:
            if intent.intent_type == IntentType.MUSIC and rec.source == 'spotify':
                rec.confidence *= 1.2
            elif intent.intent_type == IntentType.VIDEO and rec.source == 'youtube':
                rec.confidence *= 1.2
            elif intent.intent_type == IntentType.BOTH:
                rec.confidence *= 1.1
        
        # Sort by adjusted confidence
        combined.sort(key=lambda x: x.confidence, reverse=True)
        return combined[:15]  # Return top 15 combined recommendations

    def _calculate_track_confidence(self, track: SpotifyTrack, intent: MusicIntent,
                                  behavioral_data: Optional[Dict]) -> float:
        """Calculate confidence score for a track recommendation."""
        confidence = 0.5  # Base confidence
        
        # Genre matching
        if intent.genre and intent.genre.lower() in track.name.lower():
            confidence += 0.2
        
        # Artist matching
        if intent.artist and any(intent.artist.lower() in artist.lower() for artist in track.artists):
            confidence += 0.3
        
        # Song matching
        if intent.song and intent.song.lower() in track.name.lower():
            confidence += 0.4
        
        # Mood matching
        if intent.mood != MoodType.UNKNOWN:
            mood_config = self.mood_mappings.get(intent.mood, {})
            if any(keyword in track.name.lower() for keyword in mood_config.get('keywords', [])):
                confidence += 0.2
        
        # Popularity boost
        confidence += (track.popularity / 100) * 0.1
        
        return min(confidence, 1.0)

    def _calculate_playlist_confidence(self, playlist: SpotifyPlaylist, intent: MusicIntent,
                                     behavioral_data: Optional[Dict]) -> float:
        """Calculate confidence score for a playlist recommendation."""
        confidence = 0.4  # Base confidence
        
        # Name matching
        if intent.genre and intent.genre.lower() in playlist.name.lower():
            confidence += 0.3
        
        if intent.mood != MoodType.UNKNOWN:
            mood_name = intent.mood.value
            if mood_name in playlist.name.lower():
                confidence += 0.3
        
        # Description matching
        if intent.activity and intent.activity.lower() in playlist.description.lower():
            confidence += 0.2
        
        return min(confidence, 1.0)

    def _calculate_video_confidence(self, video: YouTubeVideo, intent: MusicIntent,
                                  behavioral_data: Optional[Dict]) -> float:
        """Calculate confidence score for a video recommendation."""
        confidence = 0.5  # Base confidence
        
        # Title matching
        if intent.artist and intent.artist.lower() in video.title.lower():
            confidence += 0.3
        
        if intent.song and intent.song.lower() in video.title.lower():
            confidence += 0.4
        
        # Channel matching
        if intent.artist and intent.artist.lower() in video.channel_title.lower():
            confidence += 0.2
        
        # View count boost (popularity)
        if video.view_count > 1000000:  # 1M+ views
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _generate_track_reason(self, track: SpotifyTrack, intent: MusicIntent,
                             behavioral_data: Optional[Dict]) -> str:
        """Generate human-readable reason for track recommendation."""
        reasons = []
        
        if intent.artist and any(intent.artist.lower() in artist.lower() for artist in track.artists):
            reasons.append(f"matches your request for {intent.artist}")
        
        if intent.genre and intent.genre.lower() in track.name.lower():
            reasons.append(f"fits the {intent.genre} genre you're looking for")
        
        if intent.mood != MoodType.UNKNOWN:
            mood_config = self.mood_mappings.get(intent.mood, {})
            if any(keyword in track.name.lower() for keyword in mood_config.get('keywords', [])):
                reasons.append(f"matches your {intent.mood.value} mood")
        
        if track.popularity > 70:
            reasons.append("is highly popular and well-liked")
        
        return "This track " + ", ".join(reasons) if reasons else "This is a great track for you"

    def _generate_playlist_reason(self, playlist: SpotifyPlaylist, intent: MusicIntent,
                                behavioral_data: Optional[Dict]) -> str:
        """Generate human-readable reason for playlist recommendation."""
        reasons = []
        
        if intent.genre and intent.genre.lower() in playlist.name.lower():
            reasons.append(f"curated for {intent.genre} music lovers")
        
        if intent.mood != MoodType.UNKNOWN:
            mood_name = intent.mood.value
            if mood_name in playlist.name.lower():
                reasons.append(f"perfect for {mood_name} moments")
        
        if intent.activity and intent.activity.lower() in playlist.description.lower():
            reasons.append(f"ideal for {intent.activity}")
        
        if playlist.tracks_count > 20:
            reasons.append(f"features {playlist.tracks_count} carefully selected tracks")
        
        return "This playlist is " + ", ".join(reasons) if reasons else "This is a great playlist for you"

    def _generate_video_reason(self, video: YouTubeVideo, intent: MusicIntent,
                             behavioral_data: Optional[Dict]) -> str:
        """Generate human-readable reason for video recommendation."""
        reasons = []
        
        if intent.artist and intent.artist.lower() in video.title.lower():
            reasons.append(f"features {intent.artist}")
        
        if intent.song and intent.song.lower() in video.title.lower():
            reasons.append(f"contains the song you're looking for")
        
        if video.view_count > 1000000:
            reasons.append("has millions of views and is highly popular")
        
        if "official" in video.title.lower():
            reasons.append("is the official version")
        
        return "This video " + ", ".join(reasons) if reasons else "This is a great video for you"

    def _generate_behavioral_insights(self, intent: MusicIntent,
                                    behavioral_data: Optional[Dict]) -> str:
        """Generate insights about user's current state and preferences."""
        insights = []
        
        if behavioral_data:
            emotion = behavioral_data.get('emotion', 'unknown')
            sentiment = behavioral_data.get('sentiment', 0.0)
            attention = behavioral_data.get('attention', 'unknown')
            fatigue = behavioral_data.get('fatigue', 'unknown')
            
            if emotion != 'unknown':
                insights.append(f"Your current emotional state is {emotion}")
            
            if sentiment > 0.3:
                insights.append("you're feeling positive")
            elif sentiment < -0.3:
                insights.append("you might be feeling down")
            
            if attention == 'focused':
                insights.append("you're in a focused state")
            elif attention == 'low':
                insights.append("you might need something energizing")
            
            if fatigue == 'high':
                insights.append("you seem tired and might benefit from calming music")
        
        if intent.mood != MoodType.UNKNOWN:
            insights.append(f"you're looking for {intent.mood.value} content")
        
        if intent.activity:
            insights.append(f"you're planning to {intent.activity}")
        
        return "Based on your current state, " + ", ".join(insights) if insights else "Here are some great recommendations for you"
