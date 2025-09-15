# Music Recommendation Agent Setup Guide

## Overview

The Music Recommendation Agent provides intelligent music and video recommendations based on:
- User intent detection (music, video, or both)
- Behavioral analysis (emotion, mood, activity)
- YouTube Data API v3 integration
- Spotify Web API integration
- Personalized preferences and trending content

## API Keys Required

### 1. YouTube Data API v3
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the YouTube Data API v3
4. Create credentials (API Key)
5. Set environment variable: `YOUTUBE_API_KEY=your_api_key_here`

### 2. Spotify Web API
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Get your Client ID and Client Secret
4. Set environment variables:
   - `SPOTIFY_CLIENT_ID=your_client_id_here`
   - `SPOTIFY_CLIENT_SECRET=your_client_secret_here`

## Environment Variables

Create a `.env` file in the `server/` directory with:

```bash
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# YouTube Data API v3
YOUTUBE_API_KEY=your_youtube_api_key_here

# Spotify Web API
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here

# Behavioral Analyzer API
BEHAVIORAL_ANALYZER_URL=http://localhost:8083

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=8082
WEBSOCKET_PORT=8081
```

## Features

### Intent Detection
The agent automatically detects user intent for:
- **Music requests**: "play some music", "I want to listen to rock"
- **Video requests**: "show me videos", "I want to watch something"
- **Both**: "recommend something", "what's good to watch or listen to"

### Mood-Based Recommendations
Based on behavioral analysis and user input:
- **Happy**: Upbeat pop, dance, electronic music
- **Sad**: Blues, indie, emotional music
- **Energetic**: Rock, electronic, high-energy tracks
- **Calm**: Ambient, classical, peaceful music
- **Focused**: Instrumental, lo-fi, study music
- **Relaxed**: Jazz, acoustic, chill music

### Activity-Based Recommendations
- **Workout**: High-energy electronic, hip hop, rock
- **Study**: Classical, ambient, instrumental
- **Work**: Background music, ambient, jazz
- **Party**: Dance, electronic, pop
- **Relaxation**: Calm, peaceful, meditation music
- **Driving**: Road trip music, rock, pop

## Usage Examples

### User Input Examples
- "I'm feeling sad, play something emotional"
- "I need music for working out"
- "Show me trending videos"
- "I want to listen to jazz music"
- "What's popular right now?"
- "Play something by The Beatles"
- "I need focus music for studying"

### AI Response Examples
The agent will:
1. Detect intent and mood from user input
2. Analyze current behavioral data (emotion, attention, fatigue)
3. Search YouTube and Spotify APIs
4. Generate personalized recommendations
5. Provide formatted results with reasoning

## Tools Available

### 1. get_music_recommendations
- **Purpose**: Get personalized music/video recommendations
- **Parameters**: 
  - `user_input`: User's request
  - `behavioral_data`: Current behavioral analysis (optional)
- **Returns**: Formatted recommendations with reasoning

### 2. get_trending_content
- **Purpose**: Get trending music and videos
- **Parameters**:
  - `content_type`: "music", "video", or "both" (default: "both")
- **Returns**: Trending content from YouTube and Spotify

## Integration with Behavioral Analysis

The music agent integrates with the behavioral analyzer to:
- Use current emotional state for mood-based recommendations
- Consider attention levels for activity-appropriate content
- Factor in fatigue levels for energy-appropriate music
- Provide context-aware suggestions

## Error Handling

The system gracefully handles:
- Missing API keys (shows configuration message)
- API rate limits (implements throttling)
- Network errors (returns fallback recommendations)
- Invalid responses (filters out bad results)

## Performance

- **YouTube API**: 100 requests per 100 seconds
- **Spotify API**: 1000 requests per hour
- **Recommendation Generation**: < 2 seconds typical
- **Caching**: Results cached for 5 minutes

## Testing

Test the music agent by asking:
1. "Play some happy music"
2. "I need workout music"
3. "Show me trending videos"
4. "What's popular on Spotify?"
5. "I'm feeling sad, recommend something"

The agent should provide relevant, personalized recommendations based on your request and current behavioral state.
