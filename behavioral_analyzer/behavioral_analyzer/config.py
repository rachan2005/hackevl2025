"""
Configuration management for the Behavioral Analyzer.

This module provides centralized configuration management for all components
of the behavioral analysis system, making it easy to adjust parameters and
enable/disable features.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class VideoConfig:
    """Configuration for video analysis components."""
    
    # Camera settings
    camera_id: int = 0
    resolution: tuple = (640, 480)
    
    # Feature toggles
    enable_emotion: bool = True
    enable_blink_detection: bool = True
    enable_attention_analysis: bool = True
    enable_posture_analysis: bool = True
    enable_movement_analysis: bool = True
    enable_fatigue_detection: bool = True
    enable_object_detection: bool = True
    
    # Blink detection parameters
    eye_ar_consec_frames_closed: int = 2
    eye_ar_consec_frames_open: int = 1
    blink_cooldown: float = 0.1
    calibration_frames: int = 30
    
    # Emotion detection parameters
    emotion_cooldown: float = 0.5
    emotion_backend: str = 'ssd'  # 'ssd', 'opencv', 'retinaface'
    
    # Visualization settings
    show_landmarks: bool = False
    show_graphs: bool = True
    debug_mode: bool = False
    
    # Performance settings
    plot_width: int = 200
    plot_height: int = 120
    
    # Object detection settings
    yolo_model_size: str = "n"  # 'n', 's', 'm', 'l', 'x'
    yolo_confidence_threshold: float = 0.5
    yolo_max_detections: int = 100
    yolo_device: str = "cpu"  # 'cpu', 'cuda', 'mps'
    show_object_detections: bool = True
    show_detection_confidence: bool = True
    show_detection_class: bool = True


@dataclass
class AudioConfig:
    """Configuration for audio analysis components."""
    
    # Model settings
    model: str = "tiny.en"  # 'tiny.en', 'small.en', 'base.en'
    device: str = "cpu"  # 'cpu', 'cuda'
    sample_rate: int = 16000
    
    # Audio processing parameters
    chunk_duration: float = 2.0  # seconds per chunk
    energy_threshold: float = 0.005
    
    # Feature toggles
    enable_transcription: bool = True
    enable_emotion_detection: bool = True
    enable_sentiment_analysis: bool = True
    enable_prosody_analysis: bool = True
    
    # Whisper parameters
    beam_size: int = 1
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


@dataclass
class OutputConfig:
    """Configuration for output and data collection."""
    
    # Output directory
    output_dir: str = "video"
    
    # Data collection settings
    save_json: bool = True
    save_video: bool = False
    save_audio: bool = False
    
    # JSON output settings
    json_filename: Optional[str] = None  # Auto-generated if None
    include_timestamps: bool = True
    include_performance_metrics: bool = True
    
    # Video recording settings
    video_codec: str = 'mp4v'
    video_fps: float = 20.0


@dataclass
class Config:
    """Main configuration class that combines all component configurations."""
    
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Global settings
    session_name: Optional[str] = None
    enable_logging: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure output directory exists
        os.makedirs(self.output.output_dir, exist_ok=True)
        
        # Set session name if not provided
        if self.session_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_name = f"session_{timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'video': self.video.__dict__,
            'audio': self.audio.__dict__,
            'output': self.output.__dict__,
            'session_name': self.session_name,
            'enable_logging': self.enable_logging,
            'log_level': self.log_level
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'video' in config_dict:
            for key, value in config_dict['video'].items():
                if hasattr(config.video, key):
                    setattr(config.video, key, value)
        
        if 'audio' in config_dict:
            for key, value in config_dict['audio'].items():
                if hasattr(config.audio, key):
                    setattr(config.audio, key, value)
        
        if 'output' in config_dict:
            for key, value in config_dict['output'].items():
                if hasattr(config.output, key):
                    setattr(config.output, key, value)
        
        if 'session_name' in config_dict:
            config.session_name = config_dict['session_name']
        if 'enable_logging' in config_dict:
            config.enable_logging = config_dict['enable_logging']
        if 'log_level' in config_dict:
            config.log_level = config_dict['log_level']
        
        return config
    
    def save_to_file(self, filename: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'Config':
        """Load configuration from JSON file."""
        import json
        with open(filename, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configuration instance
default_config = Config()
