# Behavioral Extractors

This module contains the behavioral feature extraction system for the ADK-based interview system. It provides a modular, extensible architecture for integrating different behavioral analysis systems.

## Architecture

### Base Classes

- **`BaseBehavioralExtractor`**: Abstract base class that all extractors must implement
- **`BehavioralFeature`**: Data class representing a single behavioral feature
- **`BehavioralExtractorFactory`**: Factory for creating and managing extractors

### Available Extractors

#### 1. Dummy Extractor (`DummyBehavioralExtractor`)
- **Purpose**: Synthetic behavioral feature generation for testing and development
- **Features**: facial_expression, audio_pitch, sentiment, attention, stress, transcription
- **Configuration**: None required
- **Use Case**: Development, testing, demos

#### 2. Real Extractor (`RealBehavioralExtractor`)
- **Purpose**: Template for connecting to your actual behavioral analysis system
- **Features**: All dummy features + eye_gaze, body_language, voice_emotion
- **Configuration**: api_endpoint, api_key
- **Use Case**: Production with external behavioral analysis systems

## Usage

### Creating an Extractor

```python
from core.behavioral_extractors import BehavioralExtractorFactory

# Create dummy extractor (for testing)
dummy_extractor = BehavioralExtractorFactory.create_extractor("dummy")

# Create real extractor (for production)
config = {
    "api_endpoint": "https://your-api.com/analyze",
    "api_key": "your-api-key"
}
real_extractor = BehavioralExtractorFactory.create_extractor("real", config)
```

### Using an Extractor

```python
# Initialize the extractor
await extractor.initialize()

# Extract features for a specific timestamp
features = await extractor.extract_features_for_timestamp(timestamp)

# Extract features for a time range
features = await extractor.extract_features_for_time_range(start_time, end_time)

# Get extractor information
info = extractor.get_extractor_info()

# Cleanup when done
await extractor.cleanup()
```

### Switching Extractors in StateAgent

In `core/session.py`, change the extractor type:

```python
# For testing/development
extractor_type = "dummy"

# For production with your external system
extractor_type = "real"
extractor_config = {
    "api_endpoint": "https://your-behavioral-api.com/analyze",
    "api_key": "your-api-key"
}
```

## Integrating Your External System

### Step 1: Update Real Extractor

Edit `real_extractor.py` and implement the following methods:

```python
async def initialize(self) -> bool:
    """Connect to your external system"""
    # Add your initialization code here
    pass

async def extract_features_for_timestamp(self, timestamp: float, context: Dict[str, Any] = None) -> List[BehavioralFeature]:
    """Extract features from your external system"""
    # Add your feature extraction logic here
    pass
```

### Step 2: Add Your API Calls

```python
async def _call_your_api(self, timestamp: float, context: Dict[str, Any]) -> List[BehavioralFeature]:
    """Call your external behavioral analysis API"""
    # Make HTTP request to your API
    # Process response
    # Convert to BehavioralFeature objects
    pass
```

### Step 3: Configure StateAgent

Update the extractor configuration in `core/session.py`:

```python
extractor_type = "real"
extractor_config = {
    "api_endpoint": "https://your-api.com/analyze",
    "api_key": "your-api-key",
    "additional_param": "value"
}
```

## Adding New Extractor Types

### Step 1: Create New Extractor Class

```python
class CustomBehavioralExtractor(BaseBehavioralExtractor):
    def __init__(self, custom_param: str):
        super().__init__("CustomBehavioralExtractor")
        self.custom_param = custom_param
    
    async def initialize(self) -> bool:
        # Your initialization logic
        pass
    
    # Implement other required methods...
```

### Step 2: Update Factory

Add your extractor to `extractor_factory.py`:

```python
elif extractor_type == "custom":
    custom_param = config.get("custom_param")
    extractor = CustomBehavioralExtractor(custom_param)
```

### Step 3: Update Available Extractors

Add your extractor info to `get_available_extractors()`:

```python
"custom": {
    "name": "CustomBehavioralExtractor",
    "description": "Your custom behavioral extractor",
    "supported_features": ["feature1", "feature2"],
    "config_required": True,
    "config_options": ["custom_param"]
}
```

## Feature Types

The system supports various behavioral feature types:

- **facial_expression**: Detected facial expressions (confident, nervous, etc.)
- **audio_pitch**: Voice pitch analysis
- **sentiment**: Sentiment analysis scores
- **attention**: Attention level indicators
- **stress**: Stress level indicators
- **transcription**: Word-level transcription with timing
- **eye_gaze**: Eye tracking data
- **body_language**: Body posture and movement analysis
- **voice_emotion**: Voice emotion analysis

## Testing

Run the extractor system test:

```bash
python test_extractor_system.py
```

This will test both dummy and real extractors, showing how the system works and what features are available.
