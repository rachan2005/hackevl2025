
"""
Real Behavioral Extractor
Connects to the actual behavioral analysis system via HTTP API
"""

import logging
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from .base_extractor import BaseBehavioralExtractor, BehavioralFeature

logger = logging.getLogger(__name__)

class RealBehavioralExtractor(BaseBehavioralExtractor):
    """Real extractor that connects to the behavioral analysis system via HTTP API"""
    
    def __init__(self, api_endpoint: str = "http://localhost:8083", api_key: str = None):
        super().__init__("RealBehavioralExtractor")
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.session = None
        self.last_data = {}
        self.last_update_time = 0
        self.update_interval = 0.5  # Update every 500ms
        self.supported_features = [
            "facial_expression",
            "audio_pitch",
            "sentiment", 
            "attention",
            "stress",
            "transcription",
            "eye_gaze",
            "body_language",
            "voice_emotion",
            "fatigue",
            "posture",
            "movement",
            "person_tracking"
        ]
    
    async def initialize(self) -> bool:
        """Initialize connection to the behavioral analysis system API"""
        logger.info("🔧 Initializing real behavioral extractor...")
        
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0),
                headers={'Content-Type': 'application/json'}
            )
            
            # Test connection to API
            health_url = f"{self.api_endpoint}/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"✅ Connected to behavioral analyzer API: {health_data.get('service', 'Unknown')}")
                else:
                    logger.warning(f"⚠️ API health check returned status {response.status}")
            
            self.is_initialized = True
            logger.info("✅ Real behavioral extractor initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize real behavioral extractor: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            return False
    
    async def extract_features_for_timestamp(self, timestamp: float, context: Dict[str, Any] = None) -> List[BehavioralFeature]:
        """
        Extract real behavioral features for a specific timestamp
        Connect to your external system here
        """
        if not self.is_initialized:
            await self.initialize()
        
        logger.info(f"🔍 REAL: Extracting behavioral features for timestamp {timestamp}")
        
        try:
            # Check if we need to update data (throttle API calls)
            current_time = time.time()
            if current_time - self.last_update_time < self.update_interval:
                # Use cached data
                data = self.last_data
            else:
                # Fetch fresh data from API
                data = await self._fetch_behavioral_data()
                if data:
                    self.last_data = data
                    self.last_update_time = current_time
            
            if not data:
                features = []
            else:
                # Extract features from the API data
                features = self._process_api_data(data, timestamp)
            
            logger.info(f"🔍 REAL: Extracted {len(features)} behavioral features for timestamp {timestamp}")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features for timestamp {timestamp}: {e}")
            return []
    
    async def extract_features_for_time_range(self, start_timestamp: float, end_timestamp: float, context: Dict[str, Any] = None) -> List[BehavioralFeature]:
        """Extract features for a time range"""
        if not self.is_initialized:
            await self.initialize()
        
        logger.info(f"🔍 REAL: Extracting features for time range {start_timestamp} to {end_timestamp}")
        
        try:
            # TODO: Implement your time range extraction logic
            # This might be more efficient than calling single timestamps
            
            features = []
            
            logger.info(f"🔍 REAL: Extracted {len(features)} features for time range")
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features for time range: {e}")
            return []
    
    def get_supported_feature_types(self) -> List[str]:
        """Get list of feature types this extractor supports"""
        return self.supported_features.copy()
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get information about this extractor"""
        return {
            "name": self.extractor_name,
            "type": "real",
            "supported_features": self.supported_features,
            "is_initialized": self.is_initialized,
            "api_endpoint": self.api_endpoint,
            "description": "Real behavioral feature extractor connected to external analysis system"
        }
    
    async def _fetch_behavioral_data(self) -> Optional[Dict[str, Any]]:
        """Fetch current behavioral data from the API"""
        if not self.session:
            return None
        
        try:
            url = f"{self.api_endpoint}/api/behavioral-data"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success'):
                        return data.get('data', {})
                else:
                    logger.warning(f"API request failed with status {response.status}")
        except Exception as e:
            logger.error(f"Error fetching behavioral data: {e}")
        
        return None
    
    def _process_api_data(self, data: Dict[str, Any], timestamp: float) -> List[BehavioralFeature]:
        """Process API data and convert to BehavioralFeature objects"""
        features = []
        
        try:
            # Emotion feature
            if 'emotion' in data and data['emotion'] != 'unknown':
                emotion_confidence = data.get('emotion_confidence', 0.8)
                if isinstance(emotion_confidence, str):
                    emotion_confidence = 0.8
                features.append(BehavioralFeature(
                    timestamp=timestamp,
                    feature_type="emotion",
                    value=data['emotion'],
                    confidence=float(emotion_confidence),
                    description=f"Facial emotion: {data['emotion']}"
                ))
            
            # Attention feature
            if 'attention' in data and data['attention'] != 'unknown':
                attention_score = data.get('attention_score', 0.7)
                if isinstance(attention_score, str):
                    attention_score = 0.7
                features.append(BehavioralFeature(
                    timestamp=timestamp,
                    feature_type="attention",
                    value=data['attention'],
                    confidence=float(attention_score),
                    description=f"Attention level: {data['attention']}"
                ))
            
            # Sentiment feature
            if 'sentiment' in data and data['sentiment'] != 'neutral':
                sentiment_score = data.get('sentiment_score', 0.0)
                if isinstance(sentiment_score, str):
                    sentiment_score = 0.0
                features.append(BehavioralFeature(
                    timestamp=timestamp,
                    feature_type="sentiment",
                    value=data['sentiment'],
                    confidence=abs(float(sentiment_score)),
                    description=f"Audio sentiment: {data['sentiment']}"
                ))
            
            # Transcription feature
            if 'transcription' in data and data['transcription'].strip():
                features.append(BehavioralFeature(
                    timestamp=timestamp,
                    feature_type="transcription",
                    value=data['transcription'],
                    confidence=0.9,
                    description=f"Speech transcription: {data['transcription'][:50]}..."
                ))
            
            # Fatigue feature
            if 'fatigue' in data and data['fatigue'] != 'normal':
                features.append(BehavioralFeature(
                    timestamp=timestamp,
                    feature_type="fatigue",
                    value=data['fatigue'],
                    confidence=0.8,
                    description=f"Fatigue level: {data['fatigue']}"
                ))
            
            # Person tracking feature
            if 'person_count' in data and data['person_count'] > 0:
                main_person_confidence = data.get('main_person_confidence', 0.8)
                if isinstance(main_person_confidence, str):
                    main_person_confidence = 0.8
                features.append(BehavioralFeature(
                    timestamp=timestamp,
                    feature_type="person_tracking",
                    value={
                        'person_count': data['person_count'],
                        'main_person_id': data.get('main_person_id'),
                        'main_person_confidence': float(main_person_confidence)
                    },
                    confidence=float(main_person_confidence),
                    description=f"Person tracking: {data['person_count']} people detected"
                ))
            
        except Exception as e:
            logger.error(f"Error processing API data: {e}")
        
        return features
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Real behavioral extractor cleaned up")
