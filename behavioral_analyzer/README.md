# Behavioral Analyzer

A comprehensive real-time behavioral analysis system that combines video and audio analysis to provide deep insights into human behavior, emotions, attention, posture, and communication patterns.

## Features

### Video Analysis
- **Facial Expression Recognition**: Real-time emotion detection using DeepFace
- **Enhanced Eye Tracking & Blink Detection**: 
  - Advanced blink detection with adaptive thresholds
  - Real-time blink counting and rate analysis
  - Blink duration and interval tracking
  - Eye state visualization (open/closed with animation)
  - Drowsiness detection and alerts
  - Eye asymmetry analysis
  - Real-time EAR (Eye Aspect Ratio) graphs
- **Attention Analysis**: Face orientation and gaze tracking
- **Posture Analysis**: Body posture assessment using pose landmarks
- **Movement Analysis**: Body movement patterns and fidgeting detection
- **Fatigue Detection**: Multi-modal fatigue assessment

### Audio Analysis
- **Real-time Speech Transcription**: Using Whisper AI for accurate transcription
- **Audio-based Emotion Detection**: Emotion recognition from voice characteristics
- **Sentiment Analysis**: Text-based sentiment analysis of transcribed speech

### Web Dashboard
- **Real-time Video Feed**: Live camera feed with all analysis overlays
- **Interactive Charts**: Dynamic graphs for blink rates, emotions, and object detection
- **Remote Controls**: Toggle features like debug mode, landmarks, and object detection
- **Session Analytics**: Comprehensive statistics and performance metrics
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **WebSocket Communication**: Real-time data streaming with low latency

### Unified Analysis
- **Multi-modal Fusion**: Intelligent combination of video and audio insights
- **Real-time Processing**: Low-latency analysis with performance optimization
- **Comprehensive Reporting**: Detailed JSON output with session statistics
- **Configurable Features**: Easy enable/disable of analysis components

## Installation

### Prerequisites
- Python 3.12 or higher
- macOS (for TensorFlow compatibility)
- Webcam and microphone access

### Install with uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/hackevl2025/behavioral-analyzer.git
cd behavioral-analyzer

# Install with uv
uv sync
```

### Install with pip
```bash
pip install -e .
```

## Quick Start

### Basic Usage
```python
from behavioral_analyzer import UnifiedBehavioralAnalyzer, Config

# Create configuration
config = Config()

# Initialize analyzer
analyzer = UnifiedBehavioralAnalyzer(config)

# Start analysis
if analyzer.start_analysis():
    try:
        while True:
            success, frame = analyzer.process_frame()
            if success:
                cv2.imshow('Behavioral Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        analyzer.cleanup()
        cv2.destroyAllWindows()
```

### Command Line Interface
```bash
# Run with default settings
behavioral-analyzer

# Run with custom configuration
behavioral-analyzer --config config.json

# Run video-only analysis
behavioral-analyzer --no-audio

# Run audio-only analysis
behavioral-analyzer --no-video
```

### Web Dashboard
```bash
# Launch the web-based dashboard
uv run python run_web_ui.py
```

Then open: http://localhost:5000

**Features:**
- Real-time video feed with object detection
- Live emotion, attention, and posture monitoring
- Audio transcription and sentiment analysis
- Interactive charts and graphs
- Session statistics and analytics
- Remote control of analyzer settings
- Responsive design for mobile and desktop

## Configuration

The system is highly configurable through the `Config` class:

```python
from behavioral_analyzer import Config, VideoConfig, AudioConfig, OutputConfig

# Create custom configuration
config = Config()

# Video analysis settings
config.video.enable_emotion = True
config.video.enable_blink_detection = True
config.video.debug_mode = False

# Audio analysis settings
config.audio.model = "small.en"  # Whisper model size
config.audio.enable_transcription = True
config.audio.enable_emotion_detection = True

# Output settings
config.output.output_dir = "results"
config.output.save_json = True
config.output.save_video = False

# Save configuration
config.save_to_file("my_config.json")
```

## Architecture

The system is built with a modular architecture for easy extension:

```
behavioral_analyzer/
├── behavioral_analyzer/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── video_analyzer.py  # Video analysis module
│   ├── audio_analyzer.py  # Audio analysis module
│   ├── unified_analyzer.py # Unified analysis interface
│   └── utils.py           # Shared utilities
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── data/                  # Sample data
```

### Key Components

1. **VideoAnalyzer**: Handles all video-based analysis including facial expressions, eye tracking, posture, and movement
2. **AudioAnalyzer**: Manages audio processing, transcription, and audio-based emotion detection
3. **UnifiedBehavioralAnalyzer**: Combines both modalities for comprehensive behavioral insights
4. **Config**: Centralized configuration management
5. **Utils**: Shared utilities for data collection, performance tracking, and visualization

## Adding New Features

The modular architecture makes it easy to add new analysis features:

### Adding a New Video Feature
1. Add configuration options to `VideoConfig`
2. Implement the feature in `VideoAnalyzer`
3. Update the unified state in `UnifiedBehavioralAnalyzer`

### Adding a New Audio Feature
1. Add configuration options to `AudioConfig`
2. Implement the feature in `AudioAnalyzer`
3. Update the unified state in `UnifiedBehavioralAnalyzer`

### Example: Adding Heart Rate Detection
```python
# In VideoConfig
@dataclass
class VideoConfig:
    # ... existing fields ...
    enable_heart_rate: bool = True
    heart_rate_window: int = 30

# In VideoAnalyzer
def _detect_heart_rate(self, frame):
    # Implement heart rate detection
    pass

# In UnifiedBehavioralAnalyzer
def _update_unified_state(self):
    # ... existing code ...
    self.unified_state['heart_rate'] = self.video_analyzer.current_heart_rate
```

## Output Format

The system generates comprehensive JSON reports:

```json
{
  "session_info": {
    "name": "session_20250114_130225",
    "duration_seconds": 120.5,
    "start_time": 1705236145.123,
    "end_time": 1705236265.623
  },
  "analysis_data": {
    "video_data": [...],
    "audio_data": [...],
    "combined_data": [...]
  },
  "statistics": {
    "total_frames": 3600,
    "total_audio_chunks": 60,
    "session_duration": 120.5
  },
  "performance": {
    "average_fps": 29.8,
    "average_processing_time": 0.033
  }
}
```

## Performance Optimization

The system includes several performance optimizations:

- **Background Processing**: Emotion detection runs in separate threads
- **Adaptive Thresholds**: Dynamic calibration for better accuracy
- **Efficient Data Structures**: Optimized data collection and storage
- **Configurable Quality**: Adjustable processing parameters

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and try different camera indices
2. **Audio not working**: Verify microphone permissions and audio device selection
3. **Low performance**: Reduce video resolution or disable some features
4. **Memory issues**: Increase system memory or reduce analysis window sizes

### Debug Mode
Enable debug mode for detailed logging:
```python
config.video.debug_mode = True
config.audio.debug_mode = True
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
uv sync --extra dev

# Run tests
pytest

# Format code
black behavioral_analyzer/

# Type checking
mypy behavioral_analyzer/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose and face detection
- [DeepFace](https://github.com/serengil/deepface) for emotion recognition
- [Whisper](https://github.com/openai/whisper) for speech transcription
- [OpenCV](https://opencv.org/) for computer vision operations

## Support

For support, please open an issue on GitHub or contact the development team.

---

**HackEVL 2025** - Building the future of behavioral analysis
