# ADK Voice Agent + Behavioral Analyzer Integration Guide

This guide explains how to run the integrated system that combines the ADK Voice Agent with real-time behavioral analysis.

## 🎯 System Overview

The integrated system consists of two main components:

1. **Behavioral Analyzer** - Real-time video/audio analysis with HTTP API
2. **ADK Voice Agent** - Conversational AI with behavioral insights integration

## 🚀 Quick Start

### Option 1: Run Everything Together (Recommended)
```bash
python start_integrated_system.py
```

This will start both systems automatically with proper coordination.

### Option 2: Run Systems Separately

**Terminal 1 - Start Behavioral Analyzer:**
```bash
python start_behavioral_analyzer.py
```

**Terminal 2 - Start ADK Voice Agent:**
```bash
python start_adk_voice_agent.py
```

## 📋 Prerequisites

### 1. Install Dependencies

**Behavioral Analyzer:**
```bash
cd behavioral_analyzer
uv sync
# or
pip install -r requirements.txt
```

**ADK Voice Agent:**
```bash
cd server
uv sync
# or
pip install -r requirements.txt
```

### 2. Environment Setup

Make sure you have the required environment variables set for the ADK Voice Agent (Google API keys, etc.).

## 🔧 Configuration

### Behavioral Analyzer API Server
- **Port**: 8083 (configurable with `--api-port`)
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /api/behavioral-data` - Complete behavioral data
  - `GET /api/emotion` - Current emotion
  - `GET /api/attention` - Attention level
  - `GET /api/fatigue` - Fatigue detection
  - `GET /api/sentiment` - Audio sentiment
  - `GET /api/person-tracking` - Person detection

### ADK Voice Agent
- **WebSocket**: Port 8081
- **HTTP API**: Port 8082
- **Behavioral Integration**: Automatically connects to Behavioral Analyzer API

## 🎮 Usage

### 1. Start the Integrated System
```bash
python start_integrated_system.py
```

### 2. Open the Web Interface
Navigate to `http://localhost:8081` in your browser to access the ADK Voice Agent interface.

### 3. Enable Camera and Microphone
- Click the camera button to enable video analysis
- Click the microphone button to enable audio analysis
- The system will automatically start behavioral analysis

### 4. Start a Conversation
- Speak into your microphone
- The system will:
  - Transcribe your speech
  - Analyze your emotions, attention, and behavior
  - Provide contextual responses based on behavioral insights
  - Track multiple people and focus on the main speaker

## 🔍 Behavioral Features

The integrated system provides real-time analysis of:

### Video Analysis
- **Emotion Detection**: Happy, sad, angry, fear, surprise, neutral
- **Attention Tracking**: Focused, partially attentive, distracted
- **Fatigue Detection**: Based on blink rate and eye openness
- **Person Tracking**: Multi-person detection with main person identification
- **Posture Analysis**: Body position and movement

### Audio Analysis
- **Speech Transcription**: Real-time speech-to-text
- **Sentiment Analysis**: Positive, negative, neutral sentiment
- **Voice Emotion**: Emotion detection from audio features
- **Speech Rate**: Speaking pace and rhythm analysis

## 🛠️ API Integration

### Behavioral Data Structure
```json
{
  "emotion": "happy",
  "emotion_confidence": 0.85,
  "attention": "focused",
  "attention_score": 0.9,
  "fatigue": "normal",
  "blink_rate": 15.2,
  "transcription": "Hello, how are you today?",
  "sentiment": "positive",
  "sentiment_score": 0.7,
  "person_count": 1,
  "main_person_id": "person_1",
  "main_person_confidence": 0.95,
  "timestamp": 1234567890.123,
  "session_duration": 45.6
}
```

### ADK Voice Agent Integration
The ADK Voice Agent automatically:
- Fetches behavioral data every 500ms
- Correlates behavioral features with conversation timestamps
- Enriches Q&A pairs with behavioral insights
- Provides contextual responses based on detected emotions and attention

## 🐛 Troubleshooting

### Common Issues

**1. Behavioral Analyzer API Not Responding**
```bash
# Check if the API is running
curl http://localhost:8083/health
```

**2. Camera Not Working**
- Ensure camera permissions are granted
- Check if another application is using the camera
- Try different camera indices: `--camera 1`, `--camera 2`

**3. Audio Not Working**
- Check microphone permissions
- Ensure no other application is using the microphone
- Try different audio devices

**4. ADK Voice Agent Connection Issues**
- Verify Google API credentials are set
- Check network connectivity
- Ensure ports 8081 and 8082 are available

### Debug Mode
Run with debug mode for detailed logging:
```bash
# Behavioral Analyzer
python -m behavioral_analyzer --enable-api --debug

# ADK Voice Agent
LOG_LEVEL=DEBUG python server.py
```

## 📊 Performance

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Camera**: 720p or higher resolution
- **Microphone**: Good quality microphone recommended

### Performance Tips
- Use `--low-latency` flag for real-time applications
- Use `--high-quality` flag for detailed analysis
- Close unnecessary applications to free up resources
- Use SSD storage for better performance

## 🔧 Advanced Configuration

### Custom API Endpoint
If running the Behavioral Analyzer on a different machine:
```python
# In server/core/behavioral_extractors/real_extractor.py
extractor = RealBehavioralExtractor(
    api_endpoint="http://your-server:8083"
)
```

### Custom Ports
```bash
# Behavioral Analyzer on custom port
python -m behavioral_analyzer --enable-api --api-port 8084

# ADK Voice Agent on custom ports (modify server.py)
```

## 📈 Monitoring

### Health Checks
- **Behavioral Analyzer**: `GET http://localhost:8083/health`
- **ADK Voice Agent**: `GET http://localhost:8082/health`

### Logs
Both systems provide detailed logging:
- Behavioral Analyzer: Console output with timestamps
- ADK Voice Agent: Structured logging with different levels

## 🤝 Contributing

To extend the integration:

1. **Add New Behavioral Features**: Modify `behavioral_analyzer/api_server.py`
2. **Enhance ADK Integration**: Update `server/core/behavioral_extractors/real_extractor.py`
3. **Add New Tools**: Extend the ADK voice agent's tool configuration

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs for error messages
3. Ensure all dependencies are properly installed
4. Verify environment variables are set correctly

---

**🎉 Enjoy your integrated ADK Voice Agent + Behavioral Analyzer system!**
