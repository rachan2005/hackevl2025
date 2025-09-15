"""
Calibration module for behavioral analysis baseline establishment.

This module handles the collection of baseline behavioral data during a 10-second
calibration period and provides methods to calculate calibrated deviations from
the established baseline.
"""

import time
import json
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class BaselineMetrics:
    """Stores baseline metrics for a user."""
    # Video metrics
    blink_rate_mean: float = 0.0
    blink_rate_std: float = 0.0
    dominant_emotion: str = "neutral"
    emotion_confidence_mean: float = 0.0
    attention_score_mean: float = 0.0
    attention_score_std: float = 0.0
    
    # Audio metrics
    speech_confidence_mean: float = 0.0
    speech_confidence_std: float = 0.0
    sentiment_mean: float = 0.0
    sentiment_std: float = 0.0
    dominant_audio_emotion: str = "neutral"
    
    # Session info
    calibration_timestamp: float = 0.0
    calibration_duration: float = 10.0
    sample_count: int = 0


class CalibrationManager:
    """Manages the calibration process and baseline calculations."""
    
    def __init__(self, calibration_duration: float = 10.0):
        self.calibration_duration = calibration_duration
        self.is_calibrating = False
        self.calibration_start_time = 0.0
        self.calibration_data = []
        self.baseline_metrics = BaselineMetrics()
        self.calibration_complete = False
        
    def start_calibration(self) -> None:
        """Start the calibration process."""
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_data = []
        self.calibration_complete = False
        print(f"Starting {self.calibration_duration}s calibration to establish baseline behavior...")
        
    def add_calibration_sample(self, data: Dict[str, Any]) -> None:
        """Add a data sample during calibration period."""
        if not self.is_calibrating:
            return
            
        current_time = time.time()
        elapsed = current_time - self.calibration_start_time
        
        if elapsed <= self.calibration_duration:
            # Add timestamp to the data
            sample = data.copy()
            sample['timestamp'] = current_time
            sample['elapsed'] = elapsed
            self.calibration_data.append(sample)
        else:
            # Calibration period complete
            self._complete_calibration()
    
    def _complete_calibration(self) -> None:
        """Complete calibration and calculate baseline metrics."""
        if not self.calibration_data:
            print("Warning: No calibration data collected!")
            return
            
        self.is_calibrating = False
        self.calibration_complete = True
        
        # Extract metrics from calibration data
        video_data = [sample.get('video', {}) for sample in self.calibration_data if sample.get('video')]
        audio_data = [sample.get('audio', {}) for sample in self.calibration_data if sample.get('audio')]
        
        # Calculate video baseline metrics
        if video_data:
            blink_rates = [d.get('blink_rate', 0) for d in video_data if d.get('blink_rate') is not None]
            emotions = [d.get('emotion', 'neutral') for d in video_data if d.get('emotion')]
            attention_scores = [self._attention_to_score(d.get('attention_state', 'Unknown')) for d in video_data]
            
            if blink_rates:
                self.baseline_metrics.blink_rate_mean = statistics.mean(blink_rates)
                self.baseline_metrics.blink_rate_std = statistics.stdev(blink_rates) if len(blink_rates) > 1 else 0.0
            
            if emotions:
                # Find most common emotion
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                self.baseline_metrics.dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            
            if attention_scores:
                valid_scores = [s for s in attention_scores if s is not None]
                if valid_scores:
                    self.baseline_metrics.attention_score_mean = statistics.mean(valid_scores)
                    self.baseline_metrics.attention_score_std = statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0
        
        # Calculate audio baseline metrics
        if audio_data:
            confidences = [d.get('confidence', 0) for d in audio_data if d.get('confidence') is not None]
            sentiments = [d.get('sentiment', 0) for d in audio_data if d.get('sentiment') is not None]
            audio_emotions = [d.get('emotion', 'neutral') for d in audio_data if d.get('emotion')]
            
            if confidences:
                self.baseline_metrics.speech_confidence_mean = statistics.mean(confidences)
                self.baseline_metrics.speech_confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            
            if sentiments:
                self.baseline_metrics.sentiment_mean = statistics.mean(sentiments)
                self.baseline_metrics.sentiment_std = statistics.stdev(sentiments) if len(sentiments) > 1 else 0.0
            
            if audio_emotions:
                # Find most common audio emotion
                emotion_counts = {}
                for emotion in audio_emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                self.baseline_metrics.dominant_audio_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Set metadata
        self.baseline_metrics.calibration_timestamp = self.calibration_start_time
        self.baseline_metrics.sample_count = len(self.calibration_data)
        
        print(f"Calibration complete! Collected {len(self.calibration_data)} samples.")
        print(f"Baseline established:")
        print(f"  - Blink rate: {self.baseline_metrics.blink_rate_mean:.1f} ± {self.baseline_metrics.blink_rate_std:.1f} blinks/min")
        print(f"  - Dominant emotion: {self.baseline_metrics.dominant_emotion}")
        print(f"  - Average attention: {self.baseline_metrics.attention_score_mean:.2f}")
        print(f"  - Speech confidence: {self.baseline_metrics.speech_confidence_mean:.2f}")
        print(f"  - Sentiment baseline: {self.baseline_metrics.sentiment_mean:.2f}")
    
    def get_calibration_progress(self) -> Dict[str, Any]:
        """Get current calibration progress."""
        if not self.is_calibrating:
            return {
                'is_calibrating': False,
                'progress_percent': 100.0 if self.calibration_complete else 0.0,
                'time_remaining': 0.0,
                'status': 'Complete' if self.calibration_complete else 'Not Started'
            }
        
        elapsed = time.time() - self.calibration_start_time
        progress_percent = min(100.0, (elapsed / self.calibration_duration) * 100.0)
        time_remaining = max(0.0, self.calibration_duration - elapsed)
        
        return {
            'is_calibrating': True,
            'progress_percent': progress_percent,
            'time_remaining': time_remaining,
            'status': f'Calibrating... {time_remaining:.1f}s remaining',
            'samples_collected': len(self.calibration_data)
        }
    
    def calculate_calibrated_values(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate calibrated values based on baseline."""
        if not self.calibration_complete:
            return current_data  # Return raw data if not calibrated
        
        calibrated_data = current_data.copy()
        
        # Add calibrated video metrics
        if 'video' in current_data:
            video = current_data['video']
            calibrated_video = video.copy()
            
            # Calibrated blink rate
            if 'blink_rate' in video and self.baseline_metrics.blink_rate_mean > 0:
                current_blink = video['blink_rate']
                baseline_blink = self.baseline_metrics.blink_rate_mean
                deviation_percent = ((current_blink - baseline_blink) / baseline_blink) * 100
                calibrated_video['blink_rate_calibrated'] = {
                    'raw': current_blink,
                    'baseline': baseline_blink,
                    'deviation_percent': deviation_percent,
                    'status': self._get_deviation_status(deviation_percent, 'blink_rate')
                }
            
            # Calibrated attention
            if 'attention_state' in video:
                current_attention_score = self._attention_to_score(video['attention_state'])
                if current_attention_score is not None and self.baseline_metrics.attention_score_mean > 0:
                    baseline_attention = self.baseline_metrics.attention_score_mean
                    deviation_percent = ((current_attention_score - baseline_attention) / baseline_attention) * 100
                    calibrated_video['attention_calibrated'] = {
                        'raw': video['attention_state'],
                        'raw_score': current_attention_score,
                        'baseline': self.baseline_metrics.attention_score_mean,
                        'deviation_percent': deviation_percent,
                        'status': self._get_deviation_status(deviation_percent, 'attention')
                    }
            
            # Emotion comparison
            if 'emotion' in video:
                current_emotion = video['emotion']
                baseline_emotion = self.baseline_metrics.dominant_emotion
                calibrated_video['emotion_calibrated'] = {
                    'raw': current_emotion,
                    'baseline': baseline_emotion,
                    'is_baseline': current_emotion == baseline_emotion,
                    'status': 'Normal' if current_emotion == baseline_emotion else f'Changed from {baseline_emotion}'
                }
            
            calibrated_data['video'] = calibrated_video
        
        # Add calibrated audio metrics
        if 'audio' in current_data:
            audio = current_data['audio']
            calibrated_audio = audio.copy()
            
            # Calibrated confidence
            if 'confidence' in audio and self.baseline_metrics.speech_confidence_mean > 0:
                current_conf = audio['confidence']
                baseline_conf = self.baseline_metrics.speech_confidence_mean
                deviation_percent = ((current_conf - baseline_conf) / baseline_conf) * 100
                calibrated_audio['confidence_calibrated'] = {
                    'raw': current_conf,
                    'baseline': baseline_conf,
                    'deviation_percent': deviation_percent,
                    'status': self._get_deviation_status(deviation_percent, 'confidence')
                }
            
            # Calibrated sentiment
            if 'sentiment' in audio:
                current_sentiment = audio['sentiment']
                baseline_sentiment = self.baseline_metrics.sentiment_mean
                deviation_percent = ((current_sentiment - baseline_sentiment) / max(abs(baseline_sentiment), 0.1)) * 100
                calibrated_audio['sentiment_calibrated'] = {
                    'raw': current_sentiment,
                    'baseline': baseline_sentiment,
                    'deviation_percent': deviation_percent,
                    'status': self._get_deviation_status(deviation_percent, 'sentiment')
                }
            
            # Audio emotion comparison
            if 'emotion' in audio:
                current_emotion = audio['emotion']
                baseline_emotion = self.baseline_metrics.dominant_audio_emotion
                calibrated_audio['emotion_calibrated'] = {
                    'raw': current_emotion,
                    'baseline': baseline_emotion,
                    'is_baseline': current_emotion == baseline_emotion,
                    'status': 'Normal' if current_emotion == baseline_emotion else f'Changed from {baseline_emotion}'
                }
            
            calibrated_data['audio'] = calibrated_audio
        
        # Add calibration metadata - preserve original structure
        calibrated_data['calibration'] = {
            'is_calibrated': True,
            'is_calibrating': False,  # Keep UI compatibility
            'progress_percent': 100.0,
            'time_remaining': 0.0,
            'status': 'Complete',
            'baseline_timestamp': self.baseline_metrics.calibration_timestamp,
            'sample_count': self.baseline_metrics.sample_count
        }
        
        return calibrated_data
    
    def _attention_to_score(self, attention_state: str) -> Optional[float]:
        """Convert attention state to numeric score."""
        attention_map = {
            'Focused': 1.0,
            'Attentive': 0.8,
            'Normal': 0.6,
            'Distracted': 0.4,
            'Very Distracted': 0.2,
            'Unknown': 0.5
        }
        return attention_map.get(attention_state)
    
    def _get_deviation_status(self, deviation_percent: float, metric_type: str) -> str:
        """Get human-readable status for deviation."""
        abs_dev = abs(deviation_percent)
        
        if abs_dev < 10:
            return "Normal"
        elif abs_dev < 25:
            direction = "above" if deviation_percent > 0 else "below"
            return f"Slightly {direction} normal"
        elif abs_dev < 50:
            direction = "above" if deviation_percent > 0 else "below"
            return f"Moderately {direction} normal"
        else:
            direction = "above" if deviation_percent > 0 else "below"
            return f"Significantly {direction} normal"
    
    def save_baseline(self, filepath: str) -> bool:
        """Save baseline metrics to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(self.baseline_metrics), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving baseline: {e}")
            return False
    
    def load_baseline(self, filepath: str) -> bool:
        """Load baseline metrics from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.baseline_metrics = BaselineMetrics(**data)
            self.calibration_complete = True
            return True
        except Exception as e:
            print(f"Error loading baseline: {e}")
            return False
