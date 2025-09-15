# Project Shadow

A Google ADK-based conversational AI system designed for natural human conversation in interview scenarios. The system integrates behavioral analysis with real-time conversation to provide contextual insights and enhance the interview experience.

## 🎯 Overview

Project Shadow consists of two main agents working together:

- **Conversational Agent**: Handles speech-to-speech interaction with users, managing questions and answers
- **State Agent**: Processes behavioral features and correlates them with conversation timestamps to provide enriched insights

The system uses a shared state mechanism where both agents can access and update conversation history, behavioral insights, and contextual information in real-time.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client UI     │    │   Server        │    │  External       │
│   (WebSocket)   │◄──►│   (ADK System)  │◄──►│  Behavioral     │
│                 │    │                 │    │  Analysis       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Google        │
                       │   Gemini API    │
                       │   (Live API)    │
                       └─────────────────┘
```

### Core Components

- **Client**: Web-based UI with real-time audio/video capture
- **Server**: Python-based WebSocket server with ADK agents
- **Behavioral Extractors**: Modular system for behavioral feature analysis
- **Shared State**: Centralized state management between agents

## 🚀 Features

### Conversational Agent
- Real-time speech-to-speech interaction
- Question and answer timestamping
- Tool calling for state management
- Context-aware conversation flow

### State Agent
- Asynchronous behavioral feature processing
- Q&A pair enrichment with behavioral insights
- Progressive state updates
- Real-time WebSocket notifications

### Behavioral Analysis
- Facial expression detection
- Audio pitch analysis
- Sentiment analysis
- Attention level monitoring
- Stress indicators
- Word-level transcription

### Client Interface
- Real-time audio recording and playback
- Webcam integration
- Shared state visualization
- Progressive behavioral insight display

## 📁 Project Structure

```
project-shadow/
├── client/                     # Web client application
│   ├── src/
│   │   ├── api/               # WebSocket API client
│   │   ├── audio/             # Audio processing modules
│   │   ├── media/             # Media handling
│   │   └── utils/             # Utility functions
│   ├── styles/                # CSS stylesheets
│   └── index.html             # Main UI
├── server/                    # Python server
│   ├── core/
│   │   ├── behavioral_extractors/  # Behavioral analysis system
│   │   ├── conversational_agent.py # Conversational AI agent
│   │   ├── state_agent.py          # State processing agent
│   │   ├── session.py              # Session management
│   │   ├── websocket_handler.py    # WebSocket handling
│   │   └── tool_handler.py         # Tool execution
│   ├── config/                # Configuration files
│   ├── api_server.py          # HTTP API server
│   └── server.py              # Main server entry point
├── cloud-functions/           # Google Cloud Functions
│   └── weather-tools/         # Example tools
├── docs/                      # Documentation
└── assets/                    # Project assets
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Node.js (for client development)
- Google Cloud account with Gemini API access
- Virtual environment (recommended)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/project-shadow.git
   cd project-shadow
   ```

2. **Set up the server**
   ```bash
   cd server
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   export LOG_LEVEL="INFO"
   ```

4. **Start the server**
   ```bash
   python server.py
   ```

5. **Open the client**
   - Navigate to `client/index.html` in your browser
   - Or serve it with a local web server

### Detailed Setup

For detailed setup instructions, see:
- [Local Development Setup](docs/local_setup.md)
- [Cloud Deployment Guide](docs/cloud_deployment.md)

## 🔧 Configuration

### Behavioral Extractor Configuration

The system supports different behavioral extractors:

```python
# In server/core/session.py
extractor_type = "dummy"  # For testing
# extractor_type = "real"  # For production

extractor_config = {
    "api_endpoint": "https://your-behavioral-api.com/analyze",
    "api_key": "your-api-key"
}
```

### Available Extractors

- **Dummy Extractor**: Synthetic behavioral features for testing
- **Real Extractor**: Template for connecting to external behavioral analysis systems

## 🎮 Usage

### Starting a Conversation

1. Open the client interface
2. Click "Connect" to establish WebSocket connection
3. Click the microphone button to start recording
4. Ask questions or provide responses
5. View real-time behavioral insights in the shared state panel

### Understanding the Interface

- **Chat Area**: Shows conversation history
- **Shared State Panel**: Displays Q&A pairs with behavioral insights
- **Controls**: Microphone, webcam, and connection controls
- **Status Indicators**: Connection status and processing indicators

## 🔌 API Reference

### WebSocket Messages

#### Client to Server
```javascript
{
  "type": "text",
  "content": "Hello, how are you?"
}
```

#### Server to Client
```javascript
{
  "type": "state_agent_update",
  "data": {
    "type": "qa_pair_enriched",
    "qa_pair": {
      "question": "How are you?",
      "answer": "I'm doing well, thank you.",
      "behavioral_insights": "The interviewee showed confidence...",
      "confidence": 0.85
    }
  }
}
```

### HTTP API Endpoints

- `POST /api/behavioral-features` - Add behavioral features
- `GET /api/session/{session_id}` - Get session information

## 🧪 Testing

### Running Tests

```bash
cd server
python -m pytest tests/
```

### Manual Testing

1. Start the server
2. Open client interface
3. Test conversation flow
4. Verify behavioral insights appear
5. Check shared state updates

## 🚀 Deployment

### Local Development
- Server runs on `localhost:8081` (WebSocket) and `localhost:8082` (HTTP API)
- Client served from file system or local web server

### Cloud Deployment
- Deploy server to Google Cloud Run
- Deploy client to Google Cloud Storage or Cloud Run
- Configure environment variables in Cloud Run

See [Cloud Deployment Guide](docs/cloud_deployment.md) for detailed instructions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines

- Follow Python PEP 8 style guidelines
- Add type hints to Python functions
- Document new features
- Test behavioral extractor changes thoroughly

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini API for conversational AI capabilities
- Google ADK for agent development framework
- WebAudio API for real-time audio processing
- WebSocket API for real-time communication

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the behavioral extractor documentation in `server/core/behavioral_extractors/README.md`

## 🔮 Roadmap

- [ ] Enhanced behavioral feature extraction
- [ ] Multi-language support
- [ ] Advanced conversation analytics
- [ ] Integration with more external systems
- [ ] Mobile app development
- [ ] Advanced UI customization

---

**Project Shadow** - Bringing behavioral insights to conversational AI