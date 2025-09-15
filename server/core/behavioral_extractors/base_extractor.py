
"""
Base Behavioral Extractor
Abstract base class for all behavioral feature extractors
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BehavioralFeature:
    """Represents a behavioral feature extracted from external systems"""
    timestamp: float
    feature_type: str  # e.g., "emotion", "sentiment", "attention", "stress"
    value: Any
    confidence: float
    description: str

class BaseBehavioralExtractor(ABC):
    """Abstract base class for behavioral feature extractors"""
    
    def __init__(self, extractor_name: str):
        self.extractor_name = extractor_name
        self.is_initialized = False
        logger.info(f"Initialized {extractor_name} behavioral extractor")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the extractor (connect to external systems, load models, etc.)"""
        pass
    
    @abstractmethod
    async def extract_features_for_timestamp(self, timestamp: float, context: Dict[str, Any] = None) -> List[BehavioralFeature]:
        """
        Extract behavioral features for a specific timestamp
        
        Args:
            timestamp: The timestamp to extract features for
            context: Additional context (e.g., audio/video data, session info)
            
        Returns:
            List of BehavioralFeature objects
        """
        pass
    
    @abstractmethod
    async def extract_features_for_time_range(self, start_timestamp: float, end_timestamp: float, context: Dict[str, Any] = None) -> List[BehavioralFeature]:
        """
        Extract behavioral features for a time range
        
        Args:
            start_timestamp: Start of the time range
            end_timestamp: End of the time range
            context: Additional context
            
        Returns:
            List of BehavioralFeature objects
        """
        pass
    
    @abstractmethod
    def get_supported_feature_types(self) -> List[str]:
        """Get list of feature types this extractor supports"""
        pass
    
    @abstractmethod
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get information about this extractor"""
        pass
    
    async def cleanup(self):
        """Cleanup resources when extractor is no longer needed"""
        logger.info(f"Cleaning up {self.extractor_name} extractor")
        self.is_initialized = False
