"""
Behavioral Analyzer - A comprehensive real-time behavioral analysis system.

This package provides tools for analyzing human behavior through:
- Video analysis (facial expressions, eye tracking, posture, movement)
- Audio analysis (speech transcription, emotion detection, prosody analysis)
- Unified behavioral insights combining both modalities

Modules:
- video_analyzer: Video-based behavioral analysis
- audio_analyzer: Audio-based behavioral analysis  
- unified_analyzer: Combined video and audio analysis
- config: Configuration management
- utils: Shared utilities and helper functions
"""

__version__ = "1.0.0"
__author__ = "HackEVL 2025"

from .video_analyzer import VideoAnalyzer
from .audio_analyzer import AudioAnalyzer
from .unified_analyzer import UnifiedBehavioralAnalyzer
from .object_detector import ObjectDetector
from .web_ui import BehavioralWebUI
from .config import Config

__all__ = [
    'VideoAnalyzer',
    'AudioAnalyzer', 
    'UnifiedBehavioralAnalyzer',
    'ObjectDetector',
    'BehavioralWebUI',
    'Config'
]
