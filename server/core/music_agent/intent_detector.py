"""
Intent Detection for Music and Video Recommendations

Detects user intent for music/video requests and extracts relevant parameters.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class IntentType(Enum):
    MUSIC = "music"
    VIDEO = "video"
    BOTH = "both"
    UNKNOWN = "unknown"

class MoodType(Enum):
    HAPPY = "happy"
    SAD = "sad"
    ENERGETIC = "energetic"
    CALM = "calm"
    FOCUSED = "focused"
    RELAXED = "relaxed"
    EXCITED = "excited"
    MELANCHOLIC = "melancholic"
    UNKNOWN = "unknown"

@dataclass
class MusicIntent:
    intent_type: IntentType
    mood: MoodType
    genre: Optional[str] = None
    artist: Optional[str] = None
    song: Optional[str] = None
    activity: Optional[str] = None
    duration: Optional[str] = None
    language: Optional[str] = None
    confidence: float = 0.0

class IntentDetector:
    """Detects user intent for music and video recommendations."""
    
    def __init__(self):
        self.music_keywords = [
            'music', 'song', 'play', 'listen', 'audio', 'track', 'album',
            'playlist', 'artist', 'band', 'singer', 'melody', 'beat',
            'rhythm', 'sound', 'tune', 'jam', 'concert'
        ]
        
        self.video_keywords = [
            'video', 'watch', 'movie', 'film', 'clip', 'youtube', 'stream',
            'documentary', 'trailer', 'show', 'series', 'episode', 'channel'
        ]
        
        self.mood_keywords = {
            MoodType.HAPPY: ['happy', 'joyful', 'cheerful', 'upbeat', 'positive', 'bright'],
            MoodType.SAD: ['sad', 'melancholic', 'depressed', 'down', 'blue', 'sorrowful'],
            MoodType.ENERGETIC: ['energetic', 'pumped', 'excited', 'high energy', 'intense', 'powerful'],
            MoodType.CALM: ['calm', 'peaceful', 'serene', 'tranquil', 'gentle', 'soft'],
            MoodType.FOCUSED: ['focused', 'concentrated', 'study', 'work', 'productive', 'mindful'],
            MoodType.RELAXED: ['relaxed', 'chill', 'laid back', 'easy', 'comfortable', 'cozy'],
            MoodType.EXCITED: ['excited', 'thrilled', 'pumped up', 'enthusiastic', 'animated'],
            MoodType.MELANCHOLIC: ['melancholic', 'nostalgic', 'reflective', 'contemplative', 'thoughtful']
        }
        
        self.genre_keywords = {
            'pop': ['pop', 'popular', 'mainstream'],
            'rock': ['rock', 'alternative', 'indie rock', 'classic rock'],
            'hip hop': ['hip hop', 'rap', 'hip-hop', 'urban'],
            'electronic': ['electronic', 'edm', 'techno', 'house', 'trance', 'dubstep'],
            'jazz': ['jazz', 'blues', 'soul', 'funk'],
            'classical': ['classical', 'orchestral', 'symphony', 'piano'],
            'country': ['country', 'folk', 'bluegrass', 'americana'],
            'r&b': ['r&b', 'rnb', 'rhythm and blues', 'soul'],
            'reggae': ['reggae', 'ska', 'dancehall'],
            'latin': ['latin', 'salsa', 'bossa nova', 'flamenco'],
            'world': ['world', 'ethnic', 'traditional', 'cultural']
        }
        
        self.activity_keywords = {
            'workout': ['workout', 'exercise', 'gym', 'running', 'fitness', 'training'],
            'study': ['study', 'learning', 'reading', 'research', 'homework'],
            'work': ['work', 'office', 'meeting', 'presentation', 'conference'],
            'party': ['party', 'celebration', 'dance', 'club', 'festival'],
            'relaxation': ['relaxation', 'meditation', 'yoga', 'spa', 'massage'],
            'driving': ['driving', 'road trip', 'commute', 'travel'],
            'cooking': ['cooking', 'kitchen', 'baking', 'recipe'],
            'sleep': ['sleep', 'bedtime', 'lullaby', 'night', 'rest']
        }

    def detect_intent(self, user_input: str, behavioral_data: Optional[Dict] = None) -> MusicIntent:
        """
        Detect music/video intent from user input and behavioral data.
        
        Args:
            user_input: User's text input
            behavioral_data: Current behavioral analysis data
            
        Returns:
            MusicIntent object with detected parameters
        """
        user_input_lower = user_input.lower()
        
        # Detect intent type
        intent_type = self._detect_intent_type(user_input_lower)
        
        # Detect mood from text and behavioral data
        mood = self._detect_mood(user_input_lower, behavioral_data)
        
        # Extract other parameters
        genre = self._extract_genre(user_input_lower)
        artist = self._extract_artist(user_input_lower)
        song = self._extract_song(user_input_lower)
        activity = self._extract_activity(user_input_lower)
        duration = self._extract_duration(user_input_lower)
        language = self._extract_language(user_input_lower)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            intent_type, mood, genre, artist, song, activity
        )
        
        return MusicIntent(
            intent_type=intent_type,
            mood=mood,
            genre=genre,
            artist=artist,
            song=song,
            activity=activity,
            duration=duration,
            language=language,
            confidence=confidence
        )

    def _detect_intent_type(self, text: str) -> IntentType:
        """Detect whether user wants music, video, or both."""
        music_score = sum(1 for keyword in self.music_keywords if keyword in text)
        video_score = sum(1 for keyword in self.video_keywords if keyword in text)
        
        if music_score > video_score and music_score > 0:
            return IntentType.MUSIC
        elif video_score > music_score and video_score > 0:
            return IntentType.VIDEO
        elif music_score > 0 and video_score > 0:
            return IntentType.BOTH
        else:
            return IntentType.UNKNOWN

    def _detect_mood(self, text: str, behavioral_data: Optional[Dict] = None) -> MoodType:
        """Detect mood from text and behavioral data."""
        # First check text for explicit mood indicators
        for mood, keywords in self.mood_keywords.items():
            if any(keyword in text for keyword in keywords):
                return mood
        
        # If no explicit mood in text, use behavioral data
        if behavioral_data:
            emotion = behavioral_data.get('emotion', '').lower()
            sentiment = behavioral_data.get('sentiment', 0.0)
            attention = behavioral_data.get('attention', '').lower()
            fatigue = behavioral_data.get('fatigue', '').lower()
            
            # Map behavioral data to mood
            if emotion == 'happy' or sentiment > 0.3:
                return MoodType.HAPPY
            elif emotion == 'sad' or sentiment < -0.3:
                return MoodType.SAD
            elif emotion == 'angry' or attention == 'high':
                return MoodType.ENERGETIC
            elif fatigue == 'high' or attention == 'low':
                return MoodType.RELAXED
            elif attention == 'focused':
                return MoodType.FOCUSED
            elif sentiment > 0.1:
                return MoodType.CALM
        
        return MoodType.UNKNOWN

    def _extract_genre(self, text: str) -> Optional[str]:
        """Extract music genre from text."""
        for genre, keywords in self.genre_keywords.items():
            if any(keyword in text for keyword in keywords):
                return genre
        return None

    def _extract_artist(self, text: str) -> Optional[str]:
        """Extract artist name from text."""
        # Look for patterns like "by [artist]", "from [artist]", "[artist] song"
        patterns = [
            r'by\s+([a-zA-Z\s]+)',
            r'from\s+([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+)\s+song',
            r'([a-zA-Z\s]+)\s+music'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                artist = match.group(1).strip()
                if len(artist) > 1 and len(artist) < 50:  # Reasonable artist name length
                    return artist
        return None

    def _extract_song(self, text: str) -> Optional[str]:
        """Extract song name from text."""
        # Look for quoted song names or "song called [name]"
        patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'song\s+called\s+([a-zA-Z\s]+)',
            r'track\s+called\s+([a-zA-Z\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                song = match.group(1).strip()
                if len(song) > 1 and len(song) < 100:  # Reasonable song name length
                    return song
        return None

    def _extract_activity(self, text: str) -> Optional[str]:
        """Extract activity context from text."""
        for activity, keywords in self.activity_keywords.items():
            if any(keyword in text for keyword in keywords):
                return activity
        return None

    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract duration preference from text."""
        patterns = [
            r'(\d+)\s*min(?:ute)?s?',
            r'(\d+)\s*hour(?:s)?',
            r'short',
            r'long',
            r'quick',
            r'extended'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    def _extract_language(self, text: str) -> Optional[str]:
        """Extract language preference from text."""
        languages = {
            'english': ['english', 'eng'],
            'spanish': ['spanish', 'español', 'es'],
            'french': ['french', 'français', 'fr'],
            'german': ['german', 'deutsch', 'de'],
            'italian': ['italian', 'italiano', 'it'],
            'portuguese': ['portuguese', 'português', 'pt'],
            'japanese': ['japanese', '日本語', 'ja'],
            'korean': ['korean', '한국어', 'ko'],
            'chinese': ['chinese', '中文', 'zh']
        }
        
        for lang, keywords in languages.items():
            if any(keyword in text for keyword in keywords):
                return lang
        return None

    def _calculate_confidence(self, intent_type: IntentType, mood: MoodType, 
                            genre: Optional[str], artist: Optional[str], 
                            song: Optional[str], activity: Optional[str]) -> float:
        """Calculate confidence score for the detected intent."""
        confidence = 0.0
        
        # Base confidence for intent type
        if intent_type != IntentType.UNKNOWN:
            confidence += 0.3
        
        # Add confidence for detected parameters
        if mood != MoodType.UNKNOWN:
            confidence += 0.2
        if genre:
            confidence += 0.15
        if artist:
            confidence += 0.15
        if song:
            confidence += 0.1
        if activity:
            confidence += 0.1
        
        return min(confidence, 1.0)
