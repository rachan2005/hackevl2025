
"""
Behavioral Extractor Factory
Factory for creating and managing behavioral extractors
"""

import logging
from typing import Dict, Any, Optional
from .base_extractor import BaseBehavioralExtractor
from .dummy_extractor import DummyBehavioralExtractor
from .real_extractor import RealBehavioralExtractor

logger = logging.getLogger(__name__)

class BehavioralExtractorFactory:
    """Factory for creating behavioral extractors"""
    
    @staticmethod
    def create_extractor(extractor_type: str = "dummy", config: Dict[str, Any] = None) -> BaseBehavioralExtractor:
        """
        Create a behavioral extractor based on type and configuration
        
        Args:
            extractor_type: Type of extractor ("dummy", "real", "custom")
            config: Configuration parameters for the extractor
            
        Returns:
            Initialized behavioral extractor
        """
        if config is None:
            config = {}
        
        logger.info(f"Creating behavioral extractor of type: {extractor_type}")
        
        if extractor_type == "dummy":
            extractor = DummyBehavioralExtractor()
            
        elif extractor_type == "real":
            api_endpoint = config.get("api_endpoint")
            api_key = config.get("api_key")
            extractor = RealBehavioralExtractor(api_endpoint=api_endpoint, api_key=api_key)
            
        elif extractor_type == "custom":
            # TODO: Add support for custom extractors
            logger.warning("Custom extractor type not implemented yet, falling back to dummy")
            extractor = DummyBehavioralExtractor()
            
        else:
            logger.warning(f"Unknown extractor type: {extractor_type}, falling back to dummy")
            extractor = DummyBehavioralExtractor()
        
        logger.info(f"Created {extractor.extractor_name}")
        return extractor
    
    @staticmethod
    def get_available_extractors() -> Dict[str, Dict[str, Any]]:
        """Get information about available extractors"""
        return {
            "dummy": {
                "name": "DummyBehavioralExtractor",
                "description": "Synthetic behavioral feature generator for testing",
                "supported_features": ["facial_expression", "audio_pitch", "sentiment", "attention", "stress", "transcription"],
                "config_required": False
            },
            "real": {
                "name": "RealBehavioralExtractor", 
                "description": "Real behavioral feature extractor for production use",
                "supported_features": ["facial_expression", "audio_pitch", "sentiment", "attention", "stress", "transcription", "eye_gaze", "body_language", "voice_emotion"],
                "config_required": True,
                "config_options": ["api_endpoint", "api_key"]
            }
        }
