"""
Audio-based behavioral analysis module.

This module provides comprehensive audio analysis capabilities including:
- Real-time speech transcription
- Audio-based emotion detection
- Sentiment analysis
- Prosody analysis (pitch, energy, speech rate)
- Silence detection
"""

import numpy as np
import sounddevice as sd
import librosa
from faster_whisper import WhisperModel
from textblob import TextBlob
import threading
import queue
import time
import sys
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import torch
import torchaudio
try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False

from .config import AudioConfig
from .utils import (
    DataCollector, AudioUtils, ColorUtils, 
    ensure_directory_exists, create_timestamped_filename
)


class SileroVAD:
    """
    Silero Voice Activity Detection wrapper.
    
    Provides real-time voice activity detection to filter out
    background noise and only process actual speech segments.
    """
    
    def __init__(self, sample_rate: int = 16000, device: str = "cpu"):
        """
        Initialize Silero VAD.
        
        Args:
            sample_rate: Audio sample rate
            device: Device to run VAD on ('cpu', 'cuda')
        """
        self.sample_rate = sample_rate
        self.device = device
        self.model = None
        self.utils = None
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model."""
        try:
            # Method 1: Try using silero-vad package directly
            if SILERO_VAD_AVAILABLE:
                try:
                    self.model = load_silero_vad()
                    if self.device != "cpu":
                        self.model = self.model.to(self.device)
                    self.utils = None
                    print("Silero VAD model loaded successfully using silero-vad package")
                    return
                except Exception as e1:
                    print(f"Silero-vad package method failed: {e1}")
            
            # Method 2: Try torch.hub with different approaches
            try:
                # Try loading with explicit return handling
                result = torch.hub.load(
                    repo_or_dir='silero-models',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                
                # Handle different return formats
                if isinstance(result, tuple) and len(result) == 2:
                    self.model, self.utils = result
                elif isinstance(result, tuple) and len(result) == 1:
                    self.model = result[0]
                    self.utils = None
                else:
                    self.model = result
                    self.utils = None
                    
            except Exception as e2:
                print(f"Torch hub method 1 failed: {e2}")
                # Method 3: Try loading with different parameters
                try:
                    result = torch.hub.load(
                        repo_or_dir='silero-models',
                        model='silero_vad',
                        force_reload=True,
                        onnx=True
                    )
                    self.model = result
                    self.utils = None
                except Exception as e3:
                    print(f"Torch hub method 2 failed: {e3}")
                    # Method 4: Try loading with explicit model name
                    try:
                        result = torch.hub.load(
                            repo_or_dir='snakers4/silero-vad',
                            model='silero_vad',
                            force_reload=False,
                            onnx=False
                        )
                        self.model = result
                        self.utils = None
                    except Exception as e4:
                        print(f"Torch hub method 3 failed: {e4}")
                        # Method 5: Try loading with different repo
                        try:
                            result = torch.hub.load(
                                repo_or_dir='snakers4/silero-vad:master',
                                model='silero_vad',
                                force_reload=False,
                                onnx=False
                            )
                            self.model = result
                            self.utils = None
                        except Exception as e5:
                            print(f"Torch hub method 4 failed: {e5}")
                            raise e2  # Re-raise the original error
            
            if self.model is not None:
                # Only call .to() if it's not a tuple
                if not isinstance(self.model, tuple):
                    self.model.to(self.device)
                print("Silero VAD model loaded successfully using torch.hub")
            else:
                raise Exception("Model is None after loading")
                
        except Exception as e:
            print(f"Error loading Silero VAD model: {e}")
            print("VAD will be disabled - audio will be processed without voice activity detection")
            self.model = None
            self.utils = None
    
    def detect_voice_activity(self, audio_chunk: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Detect voice activity in audio chunk.
        
        Args:
            audio_chunk: Audio data as numpy array
            threshold: VAD threshold (0.0 to 1.0)
            
        Returns:
            True if voice activity detected, False otherwise
        """
        if self.model is None:
            return True  # Fallback to always process if VAD not available
        
        try:
            # Silero VAD expects specific chunk sizes: 512 for 16kHz, 256 for 8kHz
            expected_samples = 512 if self.sample_rate == 16000 else 256
            
            # Process audio in chunks of the expected size
            if len(audio_chunk) >= expected_samples:
                # Take the first chunk of the expected size
                chunk = audio_chunk[:expected_samples]
                audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0)  # Add batch dimension
                
                # Get voice probability
                voice_prob = self.model(audio_tensor, self.sample_rate).item()
                
                return voice_prob > threshold
            else:
                # If chunk is too small, pad it
                padded_chunk = np.pad(audio_chunk, (0, expected_samples - len(audio_chunk)), 'constant')
                audio_tensor = torch.from_numpy(padded_chunk).float().unsqueeze(0)
                
                voice_prob = self.model(audio_tensor, self.sample_rate).item()
                return voice_prob > threshold
                
        except Exception as e:
            print(f"VAD detection error: {e}")
            return True  # Fallback to process audio
    
    def get_voice_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Get voice probability for audio chunk.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Voice probability (0.0 to 1.0)
        """
        if self.model is None:
            return 0.5  # Default probability if VAD not available
        
        try:
            # Silero VAD expects specific chunk sizes: 512 for 16kHz, 256 for 8kHz
            expected_samples = 512 if self.sample_rate == 16000 else 256
            
            # Process audio in chunks of the expected size
            if len(audio_chunk) >= expected_samples:
                # Take the first chunk of the expected size
                chunk = audio_chunk[:expected_samples]
                audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0)  # Add batch dimension
                
                voice_prob = self.model(audio_tensor, self.sample_rate).item()
                return voice_prob
            else:
                # If chunk is too small, pad it
                padded_chunk = np.pad(audio_chunk, (0, expected_samples - len(audio_chunk)), 'constant')
                audio_tensor = torch.from_numpy(padded_chunk).float().unsqueeze(0)
                
                voice_prob = self.model(audio_tensor, self.sample_rate).item()
                return voice_prob
                
        except Exception as e:
            print(f"VAD probability error: {e}")
            return 0.5


class AudioAnalyzer:
    """Comprehensive audio-based behavioral analysis."""
    
    def __init__(self, config: AudioConfig, data_collector: DataCollector = None):
        self.config = config
        self.data_collector = data_collector
        
        # Initialize audio processing parameters
        self._initialize_audio_parameters()
        
        # Initialize Whisper model
        self._initialize_whisper_model()
        
        # Initialize VAD
        if self.config.enable_vad:
            self._initialize_vad()
            # If VAD failed to initialize, disable it
            if not hasattr(self, 'vad') or self.vad is None or self.vad.model is None:
                print("VAD initialization failed - disabling VAD")
                self.config.enable_vad = False
        
        # Initialize audio analysis
        self._initialize_audio_analysis()
        
        # Control flags
        self.running = False
        self._audio_thread = None
        self._transcribe_thread = None
        
        print("Audio analyzer initialized successfully")
    
    def _initialize_audio_parameters(self):
        """Initialize audio processing parameters."""
        self.chunk_samples = int(self.config.chunk_duration * self.config.sample_rate)
        
        # Queues and buffers
        self.audio_queue = queue.Queue()
        self.buffer = np.zeros(0, dtype=np.float32)
        
        # Audio analysis history
        self.pitch_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)
        self.speech_rate_history = deque(maxlen=100)
        
        # Session statistics
        self.session_stats = {
            'total_chunks': 0,
            'total_words': 0,
            'emotion_counts': {},
            'sentiment_scores': [],
            'silence_ratio_history': []
        }
    
    def _initialize_whisper_model(self):
        """Initialize Whisper model for transcription."""
        if not self.config.enable_transcription:
            return
            
        print(f"Loading Whisper model {self.config.model} on {self.config.device}...")
        self.model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type="int8" if self.config.device == "cpu" else "int8_float16"
        )
        print("Whisper model loaded successfully")
    
    def _initialize_vad(self):
        """Initialize Voice Activity Detection."""
        try:
            self.vad = SileroVAD(
                sample_rate=self.config.sample_rate,
                device=self.config.device
            )
            
            # VAD state tracking
            self.vad_state = {
                'is_speaking': False,
                'speech_start_time': None,
                'silence_start_time': None,
                'current_speech_segment': [],
                'voice_probability': 0.0,
                'speech_segments': [],
                'total_speech_time': 0.0,
                'total_silence_time': 0.0
            }
            
            print("VAD initialized successfully")
        except Exception as e:
            print(f"Error initializing VAD: {e}")
            self.vad = None
    
    def _initialize_audio_analysis(self):
        """Initialize audio analysis components."""
        # Current analysis state
        self.current_transcription = ""
        self.current_emotion = "neutral"
        self.current_sentiment = 0.0
        self.current_confidence = "low"
        
        # Audio feature tracking
        self.audio_features = {
            'energy': 0.0,
            'pitch': 0.0,
            'speech_rate': 0.0,
            'silence_ratio': 0.0,
            'energy_z_score': 0.0,
            'pitch_z_score': 0.0,
            'rate_z_score': 0.0
        }
        
        # Emotion detection parameters
        self.emotion_thresholds = {
            'excited': {'energy': 1.0, 'pitch': 0.5, 'rate': 0.5},
            'happy': {'energy': 0.5, 'pitch': 0.3, 'rate': 0.0},
            'sad': {'energy': -0.5, 'pitch': 0.0, 'rate': -0.5, 'silence': 0.3},
            'angry': {'energy': 1.0, 'pitch': 0.0, 'rate': 0.0},
            'calm': {'energy': 0.0, 'pitch': 0.5, 'silence': 0.2}
        }
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice's InputStream."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        
        # Convert to mono if needed and ensure float32
        if indata.ndim > 1:
            audio = np.mean(indata, axis=1)
        else:
            audio = indata
        
        self.audio_queue.put(audio.astype(np.float32))
    
    def _process_audio(self):
        """Main audio processing loop."""
        while self.running:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to rolling buffer
                self.buffer = np.concatenate([self.buffer, audio_chunk])
                
                # Process if we have enough audio
                if len(self.buffer) >= self.chunk_samples:
                    # Get audio chunk and update buffer
                    audio_to_process = self.buffer[:self.chunk_samples]
                    self.buffer = self.buffer[self.chunk_samples//2:]  # 50% overlap
                    
                    # Apply VAD filtering if enabled
                    if self.vad and self.config.enable_vad:
                        self._process_with_vad(audio_to_process)
                    else:
                        # Process without VAD (original behavior)
                        self._process_audio_directly(audio_to_process)
                    
                    # Update session statistics
                    self.session_stats['total_chunks'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}", file=sys.stderr)
    
    def _process_with_vad(self, audio_data: np.ndarray):
        """Process audio chunk with VAD filtering."""
        # Get voice probability
        voice_prob = self.vad.get_voice_probability(audio_data)
        self.vad_state['voice_probability'] = voice_prob
        
        # Check if voice activity detected
        has_voice = voice_prob > self.config.vad_threshold
        current_time = time.time()
        
        if has_voice:
            # Voice detected
            if not self.vad_state['is_speaking']:
                # Start of speech
                self.vad_state['is_speaking'] = True
                self.vad_state['speech_start_time'] = current_time
                self.vad_state['current_speech_segment'] = []
                if self.vad_state['silence_start_time']:
                    silence_duration = current_time - self.vad_state['silence_start_time']
                    self.vad_state['total_silence_time'] += silence_duration
            
            # Add to current speech segment
            self.vad_state['current_speech_segment'].append(audio_data)
            
        else:
            # No voice detected
            if self.vad_state['is_speaking']:
                # End of speech
                self.vad_state['is_speaking'] = False
                self.vad_state['silence_start_time'] = current_time
                
                if self.vad_state['speech_start_time']:
                    speech_duration = current_time - self.vad_state['speech_start_time']
                    
                    # Only process if speech duration meets minimum threshold
                    if speech_duration >= self.config.vad_min_speech_duration:
                        # Combine speech segment
                        combined_audio = np.concatenate(self.vad_state['current_speech_segment'])
                        
                        # Process the complete speech segment
                        self._process_audio_directly(combined_audio)
                        
                        # Update statistics
                        self.vad_state['total_speech_time'] += speech_duration
                        self.vad_state['speech_segments'].append({
                            'start_time': self.vad_state['speech_start_time'],
                            'duration': speech_duration,
                            'voice_probability': voice_prob
                        })
                
                # Clear current segment
                self.vad_state['current_speech_segment'] = []
        
        # Also process audio directly if voice probability is above a lower threshold
        # This ensures we don't miss speech due to VAD filtering
        if voice_prob > 0.1:  # Very low threshold to catch any potential speech
            self._process_audio_directly(audio_data)
    
    def _process_audio_directly(self, audio_data: np.ndarray):
        """Process audio chunk directly without VAD filtering."""
        # Analyze audio features
        self._analyze_audio_features(audio_data)
        
        # Skip processing if energy is too low
        if self.audio_features['energy'] < self.config.energy_threshold:
            return
        
        # Transcription
        if self.config.enable_transcription:
            self._process_transcription(audio_data)
        
        # Collect data
        if self.data_collector:
            self._collect_audio_data(audio_data)
    
    def _analyze_audio_features(self, audio_chunk: np.ndarray):
        """Extract and analyze audio features for emotion detection."""
        # Convert to float32 if not already
        audio_chunk = audio_chunk.astype(np.float32)
        
        # Calculate energy (loudness)
        energy = AudioUtils.calculate_energy(audio_chunk)
        self.energy_history.append(energy)
        
        # Extract pitch using librosa
        pitch = 0.0
        if len(audio_chunk) >= 2048:  # Minimum length for pitch detection
            try:
                pitches, _ = librosa.piptrack(y=audio_chunk, sr=self.config.sample_rate)
                pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
                self.pitch_history.append(pitch)
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Pitch extraction error: {e}")
        
        # Calculate silence ratio
        silence_ratio = AudioUtils.calculate_silence_ratio(audio_chunk)
        
        # Calculate speech rate (using zero-crossing rate as proxy)
        try:
            zcr = librosa.feature.zero_crossing_rate(audio_chunk)[0]
            speech_rate = np.mean(zcr)
            self.speech_rate_history.append(speech_rate)
        except Exception as e:
            speech_rate = 0.0
            if self.config.debug_mode:
                print(f"Speech rate calculation error: {e}")
        
        # Calculate z-scores for relative measures
        energy_z_score = self._calculate_z_score(energy, self.energy_history)
        pitch_z_score = self._calculate_z_score(pitch, self.pitch_history)
        rate_z_score = self._calculate_z_score(speech_rate, self.speech_rate_history)
        
        # Update audio features
        self.audio_features.update({
            'energy': energy,
            'pitch': pitch,
            'speech_rate': speech_rate,
            'silence_ratio': silence_ratio,
            'energy_z_score': energy_z_score,
            'pitch_z_score': pitch_z_score,
            'rate_z_score': rate_z_score
        })
        
        # Detect emotion from audio features
        if self.config.enable_emotion_detection:
            self.current_emotion = self._detect_emotion_from_audio()
        
        # Update session statistics
        self.session_stats['silence_ratio_history'].append(silence_ratio)
    
    def _calculate_z_score(self, value: float, history: deque) -> float:
        """Calculate z-score for a value given its history."""
        if len(history) <= 1:
            return 0.0
        
        mean_val = sum(history) / len(history)
        std_val = np.std(list(history))
        
        if std_val == 0:
            return 0.0
        
        return (value - mean_val) / std_val
    
    def _detect_emotion_from_audio(self) -> str:
        """Detect emotion based on audio features."""
        energy_z = self.audio_features['energy_z_score']
        pitch_z = self.audio_features['pitch_z_score']
        rate_z = self.audio_features['rate_z_score']
        silence_ratio = self.audio_features['silence_ratio']
        
        # Check emotion conditions
        if (energy_z > 1 and pitch_z > 0.5 and rate_z > 0.5):
            return "excited"
        elif (energy_z > 0.5 and pitch_z > 0.3 and rate_z > 0):
            return "happy"
        elif (energy_z < -0.5 and pitch_z < 0 and rate_z < 0 and silence_ratio > 0.3):
            return "sad"
        elif (energy_z > 1 and pitch_z > 0 and rate_z > 0):
            return "angry"
        elif (energy_z < 0 and abs(pitch_z) < 0.5 and silence_ratio > 0.2):
            return "calm"
        else:
            return "neutral"
    
    def _process_transcription(self, audio_chunk: np.ndarray):
        """Process audio chunk for transcription."""
        try:
            # Transcribe using Whisper
            segments, info = self.model.transcribe(
                audio_chunk,
                beam_size=self.config.beam_size,
                word_timestamps=self.config.word_timestamps,
                condition_on_previous_text=self.config.condition_on_previous_text,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
                log_prob_threshold=self.config.log_prob_threshold,
                no_speech_threshold=self.config.no_speech_threshold
            )
            
            # Process transcription results
            current_time = time.time()
            for segment in segments:
                text = segment.text.strip()
                if text:
                    # Calculate confidence metrics
                    confidence = segment.avg_logprob
                    speech_clarity = 1.0 - segment.no_speech_prob
                    
                    # Text-based sentiment analysis
                    if self.config.enable_sentiment_analysis:
                        blob = TextBlob(text)
                        text_sentiment = blob.sentiment.polarity
                        text_subjectivity = blob.sentiment.subjectivity
                        
                        # Update current sentiment
                        self.current_sentiment = text_sentiment
                        self.session_stats['sentiment_scores'].append(text_sentiment)
                    
                    # Determine confidence level
                    if (abs(self.audio_features['energy_z_score']) > 1 and 
                        abs(self.audio_features['pitch_z_score']) > 1):
                        self.current_confidence = "high"
                    elif (abs(self.audio_features['energy_z_score']) > 0.5 or 
                          abs(self.audio_features['pitch_z_score']) > 0.5):
                        self.current_confidence = "medium"
                    else:
                        self.current_confidence = "low"
                    
                    # Update current transcription
                    self.current_transcription = text
                    
                    # Update session statistics
                    word_count = len(text.split())
                    self.session_stats['total_words'] += word_count
                    
                    # Print results
                    self._print_transcription_results(text, current_time, segment)
        
        except Exception as e:
            print(f"Transcription error: {e}", file=sys.stderr)
    
    def _print_transcription_results(self, text: str, timestamp: float, segment):
        """Print transcription results with analysis."""
        print(f"[{time.strftime('%H:%M:%S')}] {text}")
        print(f"  Audio Emotion: {self.current_emotion} ({self.current_confidence} confidence)")
        print(f"  Audio Features:")
        print(f"    - Energy: {self.audio_features['energy']:.3f} (z-score: {self.audio_features['energy_z_score']:.2f})")
        print(f"    - Pitch: {self.audio_features['pitch']:.1f} (z-score: {self.audio_features['pitch_z_score']:.2f})")
        print(f"    - Speech Rate: {self.audio_features['speech_rate']:.3f} (z-score: {self.audio_features['rate_z_score']:.2f})")
        print(f"    - Silence Ratio: {self.audio_features['silence_ratio']:.2f}")
        
        # VAD Status
        if hasattr(self, 'vad_state') and self.vad_state:
            vad_info = self._get_vad_info()
            print(f"  VAD Status:")
            print(f"    - Enabled: {vad_info['enabled']}")
            print(f"    - Voice Probability: {vad_info['voice_probability']:.3f}")
            print(f"    - Currently Speaking: {vad_info['is_speaking']}")
            print(f"    - Speech Segments: {vad_info['speech_segments_count']}")
        else:
            print(f"  VAD Status: Disabled (model not loaded)")
        
        if self.config.enable_sentiment_analysis:
            print(f"  Text Analysis:")
            print(f"    - Sentiment: {self.current_sentiment:.2f} (-1.0 to 1.0)")
            print(f"    - Start Time: {segment.start:.2f}s")
            
            if hasattr(segment, 'words') and segment.words:
                words_info = " ".join([f"{w.word}({w.start:.1f}s)" for w in segment.words])
                print(f"    - Words: {words_info}")
    
    def _collect_audio_data(self, audio_chunk: np.ndarray):
        """Collect audio analysis data for output."""
        current_time = time.time()
        
        # Create data point
        data_point = {
            'timestamp': current_time,
            'transcription': self.current_transcription,
            'emotion': self.current_emotion,
            'sentiment': self.current_sentiment,
            'confidence': self.current_confidence,
            'audio_features': self.audio_features.copy(),
            'chunk_duration': self.config.chunk_duration,
            'sample_rate': self.config.sample_rate,
            'vad_info': self._get_vad_info()
        }
        
        self.data_collector.add_audio_data(data_point)
    
    def _get_vad_info(self) -> Dict[str, Any]:
        """Get VAD information for data collection."""
        if not hasattr(self, 'vad_state') or not self.vad_state:
            return {
                'enabled': False,
                'voice_probability': 0.0,
                'is_speaking': False,
                'total_speech_time': 0.0,
                'total_silence_time': 0.0,
                'speech_segments_count': 0
            }
        
        return {
            'enabled': self.config.enable_vad,
            'voice_probability': self.vad_state.get('voice_probability', 0.0),
            'is_speaking': self.vad_state.get('is_speaking', False),
            'total_speech_time': self.vad_state.get('total_speech_time', 0.0),
            'total_silence_time': self.vad_state.get('total_silence_time', 0.0),
            'speech_segments_count': len(self.vad_state.get('speech_segments', []))
        }
    
    def start(self):
        """Start the audio analysis system."""
        if self.running:
            return
        
        if not self.config.enable_transcription:
            print("Transcription is disabled in configuration")
            return
        
        self.running = True
        self.buffer = np.zeros(0, dtype=np.float32)
        
        # Start audio input stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.config.sample_rate,
            blocksize=int(0.1 * self.config.sample_rate),  # 100ms chunks
            callback=self._audio_callback
        )
        self.stream.start()
        
        # Start processing thread
        self._transcribe_thread = threading.Thread(target=self._process_audio)
        self._transcribe_thread.daemon = True
        self._transcribe_thread.start()
        
        print("Audio analysis started - listening...")
    
    def stop(self):
        """Stop the audio analysis system."""
        self.running = False
        
        if self._transcribe_thread:
            self._transcribe_thread.join(timeout=2.0)
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        print("Audio analysis stopped")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current audio analysis state."""
        return {
            'transcription': self.current_transcription,
            'emotion': self.current_emotion,
            'sentiment': self.current_sentiment,
            'confidence': self.current_confidence,
            'audio_features': self.audio_features.copy(),
            'session_stats': self.session_stats.copy()
        }
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        if not self.session_stats['sentiment_scores']:
            avg_sentiment = 0.0
        else:
            avg_sentiment = sum(self.session_stats['sentiment_scores']) / len(self.session_stats['sentiment_scores'])
        
        if not self.session_stats['silence_ratio_history']:
            avg_silence = 0.0
        else:
            avg_silence = sum(self.session_stats['silence_ratio_history']) / len(self.session_stats['silence_ratio_history'])
        
        return {
            'total_chunks': self.session_stats['total_chunks'],
            'total_words': self.session_stats['total_words'],
            'average_sentiment': avg_sentiment,
            'average_silence_ratio': avg_silence,
            'emotion_distribution': self._get_emotion_distribution(),
            'audio_quality_metrics': self._get_audio_quality_metrics()
        }
    
    def _get_emotion_distribution(self) -> Dict[str, int]:
        """Get distribution of detected emotions."""
        # This would be populated during analysis
        # For now, return empty distribution
        return {
            'neutral': 0,
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'excited': 0,
            'calm': 0
        }
    
    def _get_audio_quality_metrics(self) -> Dict[str, float]:
        """Get audio quality metrics."""
        if not self.energy_history:
            return {
                'average_energy': 0.0,
                'energy_stability': 0.0,
                'average_pitch': 0.0,
                'pitch_stability': 0.0
            }
        
        avg_energy = sum(self.energy_history) / len(self.energy_history)
        energy_stability = 1.0 - np.std(list(self.energy_history)) / max(avg_energy, 0.001)
        
        if self.pitch_history:
            avg_pitch = sum(self.pitch_history) / len(self.pitch_history)
            pitch_stability = 1.0 - np.std(list(self.pitch_history)) / max(avg_pitch, 0.001)
        else:
            avg_pitch = 0.0
            pitch_stability = 0.0
        
        return {
            'average_energy': avg_energy,
            'energy_stability': energy_stability,
            'average_pitch': avg_pitch,
            'pitch_stability': pitch_stability
        }
    
    def reset_audio_stats(self):
        """Reset audio statistics."""
        self.pitch_history.clear()
        self.energy_history.clear()
        self.speech_rate_history.clear()
        self.session_stats = {
            'total_chunks': 0,
            'total_words': 0,
            'emotion_counts': {},
            'sentiment_scores': [],
            'silence_ratio_history': []
        }
        print("Audio statistics reset")
    
    def cleanup(self):
        """Clean up audio analyzer resources."""
        self.stop()
        print("Audio analyzer cleaned up")
