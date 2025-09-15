"""
Tests for the configuration module.
"""

import pytest
import tempfile
import os
from behavioral_analyzer.config import Config, VideoConfig, AudioConfig, OutputConfig


class TestConfig:
    """Test configuration classes."""
    
    def test_video_config_defaults(self):
        """Test VideoConfig default values."""
        config = VideoConfig()
        
        assert config.camera_id == 0
        assert config.resolution == (640, 480)
        assert config.enable_emotion is True
        assert config.enable_blink_detection is True
        assert config.enable_attention_analysis is True
        assert config.enable_posture_analysis is True
        assert config.enable_movement_analysis is True
        assert config.enable_fatigue_detection is True
        assert config.show_landmarks is False
        assert config.debug_mode is False
    
    def test_audio_config_defaults(self):
        """Test AudioConfig default values."""
        config = AudioConfig()
        
        assert config.model == "tiny.en"
        assert config.device == "cpu"
        assert config.sample_rate == 16000
        assert config.chunk_duration == 2.0
        assert config.energy_threshold == 0.005
        assert config.enable_transcription is True
        assert config.enable_emotion_detection is True
        assert config.enable_sentiment_analysis is True
        assert config.enable_prosody_analysis is True
    
    def test_output_config_defaults(self):
        """Test OutputConfig default values."""
        config = OutputConfig()
        
        assert config.output_dir == "video"
        assert config.save_json is True
        assert config.save_video is False
        assert config.save_audio is False
        assert config.json_filename is None
        assert config.include_timestamps is True
        assert config.include_performance_metrics is True
        assert config.video_codec == 'mp4v'
        assert config.video_fps == 20.0
    
    def test_config_creation(self):
        """Test Config creation and initialization."""
        config = Config()
        
        assert isinstance(config.video, VideoConfig)
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.output, OutputConfig)
        assert config.session_name is not None
        assert config.enable_logging is True
        assert config.log_level == "INFO"
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = Config()
        config_dict = config.to_dict()
        
        assert 'video' in config_dict
        assert 'audio' in config_dict
        assert 'output' in config_dict
        assert 'session_name' in config_dict
        assert 'enable_logging' in config_dict
        assert 'log_level' in config_dict
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'video': {
                'camera_id': 1,
                'resolution': [1280, 720],
                'enable_emotion': False
            },
            'audio': {
                'model': 'small.en',
                'device': 'cuda'
            },
            'output': {
                'output_dir': 'results',
                'save_video': True
            },
            'session_name': 'test_session',
            'enable_logging': False,
            'log_level': 'DEBUG'
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.video.camera_id == 1
        assert config.video.resolution == (1280, 720)
        assert config.video.enable_emotion is False
        assert config.audio.model == 'small.en'
        assert config.audio.device == 'cuda'
        assert config.output.output_dir == 'results'
        assert config.output.save_video is True
        assert config.session_name == 'test_session'
        assert config.enable_logging is False
        assert config.log_level == 'DEBUG'
    
    def test_config_save_load(self):
        """Test configuration save and load from file."""
        config = Config()
        config.video.camera_id = 2
        config.audio.model = 'base.en'
        config.session_name = 'test_save_load'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save configuration
            config.save_to_file(temp_file)
            assert os.path.exists(temp_file)
            
            # Load configuration
            loaded_config = Config.load_from_file(temp_file)
            
            assert loaded_config.video.camera_id == 2
            assert loaded_config.audio.model == 'base.en'
            assert loaded_config.session_name == 'test_save_load'
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_config_output_directory_creation(self):
        """Test that output directory is created on initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, 'test_output')
            
            config = Config()
            config.output.output_dir = output_dir
            
            # The directory should be created in __post_init__
            assert os.path.exists(output_dir)
            assert os.path.isdir(output_dir)

