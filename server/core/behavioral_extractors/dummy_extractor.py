
"""
Dummy Behavioral Extractor
Synthetic behavioral feature generation for testing and development
"""

import logging
import asyncio
import time
from typing import List, Dict, Any
from .base_extractor import BaseBehavioralExtractor, BehavioralFeature

logger = logging.getLogger(__name__)

class DummyBehavioralExtractor(BaseBehavioralExtractor):
    """Dummy extractor that generates synthetic behavioral features"""
    
    def __init__(self):
        super().__init__("DummyBehavioralExtractor")
        self.supported_features = [
            "facial_expression",
            "audio_pitch", 
            "sentiment",
            "attention",
            "stress",
            "transcription"
        ]
    
    async def initialize(self) -> bool:
        """Initialize the dummy extractor"""
        logger.info("🔧 Initializing dummy behavioral extractor...")
        # Simulate initialization delay
        await asyncio.sleep(0.1)
        self.is_initialized = True
        logger.info("✅ Dummy behavioral extractor initialized")
        return True
    
    async def extract_features_for_timestamp(self, timestamp: float, context: Dict[str, Any] = None) -> List[BehavioralFeature]:
        """
        Generate synthetic behavioral features for a specific timestamp
        This simulates what your real extractor will do
        """
        if not self.is_initialized:
            await self.initialize()
        
        logger.info(f"🔍 DUMMY: Extracting behavioral features for timestamp {timestamp}")
        
        features = []
        
        # Simulate facial expression analysis
        emotions = ["confident", "thoughtful", "nervous", "engaged", "uncertain", "focused", "relaxed"]
        selected_emotion = emotions[int(timestamp) % len(emotions)]
        features.append(BehavioralFeature(
            timestamp=timestamp,
            feature_type="facial_expression",
            value=selected_emotion,
            confidence=0.85,
            description=f"Facial expression analysis detected: {selected_emotion}"
        ))
        
        # Simulate audio pitch analysis
        pitch_variation = 0.3 + (timestamp % 0.4)  # Vary between 0.3-0.7
        features.append(BehavioralFeature(
            timestamp=timestamp,
            feature_type="audio_pitch",
            value=pitch_variation,
            confidence=0.78,
            description=f"Audio pitch variation: {pitch_variation:.2f} (normal range: 0.2-0.8)"
        ))
        
        # Simulate sentiment analysis
        sentiment_score = -0.5 + (timestamp % 1.0)  # Vary between -0.5 to 0.5
        sentiment_label = "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
        features.append(BehavioralFeature(
            timestamp=timestamp,
            feature_type="sentiment",
            value=sentiment_score,
            confidence=0.72,
            description=f"Sentiment analysis: {sentiment_label} ({sentiment_score:.2f})"
        ))
        
        # Simulate attention level
        attention_level = 0.6 + (timestamp % 0.3)  # Vary between 0.6-0.9
        features.append(BehavioralFeature(
            timestamp=timestamp,
            feature_type="attention",
            value=attention_level,
            confidence=0.81,
            description=f"Attention level: {attention_level:.2f} (high engagement)"
        ))
        
        # Simulate stress indicators
        stress_level = 0.2 + (timestamp % 0.4)  # Vary between 0.2-0.6
        features.append(BehavioralFeature(
            timestamp=timestamp,
            feature_type="stress",
            value=stress_level,
            confidence=0.69,
            description=f"Stress indicators: {stress_level:.2f} (low to moderate)"
        ))
        
        # Simulate transcription with word-level timing
        sample_words = ["I", "think", "that", "would", "be", "a", "great", "opportunity"]
        word_timestamps = [timestamp + (i * 0.5) for i in range(len(sample_words))]
        for i, word in enumerate(sample_words):
            features.append(BehavioralFeature(
                timestamp=word_timestamps[i],
                feature_type="transcription",
                value=word,
                confidence=0.95,
                description=f"Transcribed word: '{word}' at {word_timestamps[i]:.2f}s"
            ))
        
        logger.info(f"🔍 DUMMY: Generated {len(features)} behavioral features for timestamp {timestamp}")
        return features
    
    async def extract_features_for_time_range(self, start_timestamp: float, end_timestamp: float, context: Dict[str, Any] = None) -> List[BehavioralFeature]:
        """Extract features for a time range (dummy implementation)"""
        if not self.is_initialized:
            await self.initialize()
        
        logger.info(f"🔍 DUMMY: Extracting features for time range {start_timestamp} to {end_timestamp}")
        
        # Generate features at regular intervals within the range
        features = []
        interval = 1.0  # 1 second intervals
        current_time = start_timestamp
        
        while current_time <= end_timestamp:
            range_features = await self.extract_features_for_timestamp(current_time, context)
            features.extend(range_features)
            current_time += interval
        
        logger.info(f"🔍 DUMMY: Generated {len(features)} features for time range")
        return features
    
    def get_supported_feature_types(self) -> List[str]:
        """Get list of feature types this extractor supports"""
        return self.supported_features.copy()
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get information about this extractor"""
        return {
            "name": self.extractor_name,
            "type": "dummy",
            "supported_features": self.supported_features,
            "is_initialized": self.is_initialized,
            "description": "Synthetic behavioral feature generator for testing and development"
        }
