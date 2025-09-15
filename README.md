# HackEVL 2025 - Behavioral Analyzer

A comprehensive real-time behavioral analysis system that combines computer vision, audio processing, and machine learning to analyze human behavior through video and audio streams.

## 🚀 Features

### Video Analysis
- **Multi-person Detection**: YOLO-based object detection to identify and track multiple people
- **Main Person Focus**: Automatically identifies and focuses analysis on the primary subject
- **Facial Emotion Recognition**: Real-time emotion detection using DeepFace
- **Eye Blink Detection**: Advanced blink detection with adaptive thresholds for fatigue analysis
- **Pose Analysis**: Body posture and movement tracking using MediaPipe
- **Attention Tracking**: Face orientation and gaze analysis

### Audio Analysis
- **Speech Transcription**: Real-time speech-to-text using Faster Whisper
- **Sentiment Analysis**: Text-based sentiment scoring and analysis
- **Audio Emotion Detection**: Prosodic feature-based emotion recognition
- **Speech Rate Analysis**: Speaking pace and rhythm analysis
- **Audio Feature Extraction**: Energy, pitch, and silence ratio analysis

### System Features
- **Real-time Processing**: Optimized for live video and audio streams
- **Modular Architecture**: Easy to extend with new analysis features
- **JSON Output**: Structured session data export
- **Clean Python Interface**: Command-line interface with live feedback
- **Performance Optimized**: Background processing threads for smooth operation

## 📋 Requirements

### System Requirements
- **Python**: 3.12 or higher
- **Operating System**: macOS (optimized for Apple Silicon)
- **Camera**: Built-in or external webcam
- **Microphone**: Built-in or external microphone
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 2GB free space for models and dependencies

### Python Dependencies
```bash
# Core computer vision and ML
numpy>=1.24.0
opencv-python>=4.8.0
mediapipe>=0.10.0
tensorflow-macos>=2.13.0
tf-keras>=2.13.0
deepface>=0.0.79

# Object detection and tracking
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0

# Audio processing and speech recognition
sounddevice>=0.4.6
librosa>=0.10.0
faster-whisper>=0.9.0
textblob>=0.17.1

# Data processing
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
pillow>=9.5.0

# Web interface (optional)
flask>=2.3.0
flask-socketio>=5.3.0
eventlet>=0.33.0
```

## 🛠️ Installation

### Option 1: Using uv (Recommended)
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/rachan2005/hackevl2025.git
cd hackevl2025/behavioral_analyzer

# Install dependencies
uv sync

# Run the behavioral analyzer
uv run python -m behavioral_analyzer
```

### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/rachan2005/hackevl2025.git
cd hackevl2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the behavioral analyzer
cd behavioral_analyzer
python -m behavioral_analyzer
```

## 🎮 Usage

### Basic Usage
```bash
# Run with default settings
uv run python -m behavioral_analyzer

# Or with pip
python -m behavioral_analyzer
```

### Controls
- **'q'** - Quit the application
- **'d'** - Toggle debug mode
- **'l'** - Toggle landmark display
- **'r'** - Start/stop recording
- **'s'** - Save session data
- **'h'** - Show help

### Output
- **Real-time Console**: Live analysis results and statistics
- **Session Data**: JSON files saved to `video/` directory
- **Video Recording**: MP4 files saved to `video/` directory (optional)

## 📊 Analysis Output

The system provides comprehensive behavioral analysis including:

### Video Metrics
- Person count and main person identification
- Real-time emotion detection (happy, sad, angry, fear, surprise, neutral)
- Eye blink rate and fatigue indicators
- Face orientation and attention level
- Pose landmarks and body movement

### Audio Metrics
- Speech transcription with word-level timing
- Sentiment analysis scores (-1.0 to 1.0)
- Audio emotion detection
- Speech rate and rhythm analysis
- Energy, pitch, and silence ratio

### Session Summary
- Overall session duration and statistics
- Final emotion and attention states
- Average processing FPS
- Comprehensive behavioral insights

## 🏗️ Architecture

```
behavioral_analyzer/
├── behavioral_analyzer/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── video_analyzer.py      # Video analysis engine
│   ├── audio_analyzer.py      # Audio analysis engine
│   ├── object_detector.py     # YOLO object detection
│   ├── unified_analyzer.py    # Main orchestrator
│   ├── web_ui.py             # Web dashboard (optional)
│   └── utils.py              # Utility functions
├── tests/                    # Test suite
├── examples/                 # Example scripts
├── static/                   # Web UI assets
├── templates/                # Web UI templates
├── pyproject.toml           # Project configuration
└── requirements.txt         # Dependencies
```

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# Video Analysis
enable_object_detection = True
yolo_model_size = "n"  # nano, small, medium, large, xlarge
yolo_confidence_threshold = 0.5

# Audio Analysis
whisper_model_size = "tiny.en"
audio_sample_rate = 16000

# Performance
max_fps = 30
emotion_detection_interval = 0.5
```

## 🚀 Performance

- **Processing Speed**: ~12-15 FPS on Apple M2
- **Memory Usage**: ~2-4GB RAM
- **Latency**: <100ms for real-time analysis
- **Accuracy**: >90% emotion detection accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** for facial landmark detection
- **DeepFace** for emotion recognition
- **Ultralytics YOLO** for object detection
- **Faster Whisper** for speech recognition
- **OpenCV** for computer vision operations

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the HackEVL 2025 team.

---

**Built with ❤️ for HackEVL 2025**
