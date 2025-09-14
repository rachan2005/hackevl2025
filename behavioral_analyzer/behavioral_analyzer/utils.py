"""
Utility functions and helper classes for the Behavioral Analyzer.

This module provides shared functionality used across different components
of the behavioral analysis system.
"""

import time
import json
import datetime
import platform
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import numpy as np
import cv2


class PerformanceTracker:
    """Track performance metrics for analysis components."""
    
    def __init__(self, max_history: int = 30):
        self.fps_history = deque(maxlen=max_history)
        self.processing_times = deque(maxlen=max_history)
        self.last_frame_time = time.time()
        self.current_fps = 0.0
        
    def update(self, processing_time: float) -> None:
        """Update performance metrics with new frame processing time."""
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if dt > 0:
            fps = 1 / dt
            self.fps_history.append(fps)
            self.current_fps = sum(self.fps_history) / len(self.fps_history)
        
        self.processing_times.append(processing_time)
    
    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        return {
            'current_fps': self.current_fps,
            'average_fps': sum(self.fps_history) / max(1, len(self.fps_history)),
            'average_processing_time': sum(self.processing_times) / max(1, len(self.processing_times)),
            'max_processing_time': max(self.processing_times) if self.processing_times else 0,
            'min_processing_time': min(self.processing_times) if self.processing_times else 0
        }


class DataCollector:
    """Collect and manage session data for JSON output."""
    
    def __init__(self, session_name: str, start_time: Optional[float] = None):
        self.session_name = session_name
        self.start_time = start_time or time.time()
        
        # Initialize data structure
        self.data = {
            'session_info': {
                'name': session_name,
                'start_time': self.start_time,
                'system_info': self._get_system_info()
            },
            'analysis_data': {
                'video_data': [],
                'audio_data': [],
                'combined_data': []
            },
            'statistics': {
                'total_frames': 0,
                'total_audio_chunks': 0,
                'session_duration': 0
            },
            'performance': {}
        }
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "opencv_version": cv2.__version__
        }
    
    def add_video_data(self, data_point: Dict[str, Any]) -> None:
        """Add video analysis data point."""
        self.data['analysis_data']['video_data'].append(data_point)
        self.data['statistics']['total_frames'] += 1
    
    def add_audio_data(self, data_point: Dict[str, Any]) -> None:
        """Add audio analysis data point."""
        self.data['analysis_data']['audio_data'].append(data_point)
        self.data['statistics']['total_audio_chunks'] += 1
    
    def add_combined_data(self, data_point: Dict[str, Any]) -> None:
        """Add combined analysis data point."""
        self.data['analysis_data']['combined_data'].append(data_point)
    
    def update_performance(self, performance_stats: Dict[str, float]) -> None:
        """Update performance statistics."""
        self.data['performance'] = performance_stats
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize data collection and return complete dataset."""
        end_time = time.time()
        session_duration = end_time - self.start_time
        
        self.data['session_info']['end_time'] = end_time
        self.data['session_info']['duration_seconds'] = session_duration
        self.data['session_info']['duration_formatted'] = self._format_duration(session_duration)
        self.data['statistics']['session_duration'] = session_duration
        
        return self.data
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def save_to_file(self, filename: str) -> str:
        """Save data to JSON file."""
        try:
            final_data = self.finalize()
            with open(filename, 'w') as f:
                json.dump(final_data, f, indent=4)
            return filename
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")
            return None


class FrameProcessor:
    """Utility class for frame processing operations."""
    
    @staticmethod
    def extract_face_region(frame: np.ndarray, landmarks, expand_ratio: float = 1.2) -> Optional[np.ndarray]:
        """Extract face region from frame using landmarks."""
        try:
            h, w, _ = frame.shape
            
            # Get bounding box of face
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            
            for landmark in landmarks.landmark:
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                x_min = min(x_min, px)
                y_min = min(y_min, py)
                x_max = max(x_max, px)
                y_max = max(y_max, py)
            
            # Expand bounding box
            width = x_max - x_min
            height = y_max - y_min
            
            x_min = max(0, int(x_min - width * (expand_ratio - 1) / 2))
            y_min = max(0, int(y_min - height * (expand_ratio - 1) / 2))
            x_max = min(w, int(x_max + width * (expand_ratio - 1) / 2))
            y_max = min(h, int(y_max + height * (expand_ratio - 1) / 2))
            
            # Extract face region
            face_region = frame[y_min:y_max, x_min:x_max]
            
            if face_region.shape[0] > 0 and face_region.shape[1] > 0:
                return face_region
                
        except Exception as e:
            print(f"Error extracting face region: {e}")
        
        return None
    
    @staticmethod
    def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int], 
                                 font_scale: float = 0.6, thickness: int = 2, 
                                 text_color: Tuple[int, int, int] = (255, 255, 255),
                                 bg_color: Tuple[int, int, int] = (0, 0, 0)) -> None:
        """Draw text with background for better visibility."""
        x, y = position
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(frame, (x, y - text_height - baseline), 
                     (x + text_width, y + baseline), bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, text_color, thickness)


class AudioUtils:
    """Utility functions for audio processing."""
    
    @staticmethod
    def calculate_energy(audio_chunk: np.ndarray) -> float:
        """Calculate energy (loudness) of audio chunk."""
        return np.sqrt(np.mean(audio_chunk**2))
    
    @staticmethod
    def calculate_silence_ratio(audio_chunk: np.ndarray, threshold: float = 0.001) -> float:
        """Calculate ratio of silence in audio chunk."""
        energy = AudioUtils.calculate_energy(audio_chunk)
        return np.mean(energy < threshold)
    
    @staticmethod
    def normalize_audio(audio_chunk: np.ndarray) -> np.ndarray:
        """Normalize audio chunk to [-1, 1] range."""
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 0:
            return audio_chunk / max_val
        return audio_chunk


class MathUtils:
    """Mathematical utility functions."""
    
    @staticmethod
    def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], 
                       p3: Tuple[float, float]) -> float:
        """Calculate angle between three points."""
        import math
        
        # Vector from p2 to p1
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        # Vector from p2 to p3
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate angle
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        
        return math.degrees(math.acos(cos_angle))
    
    @staticmethod
    def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        import math
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def smooth_signal(signal: List[float], window_size: int = 5) -> List[float]:
        """Apply simple moving average smoothing to signal."""
        if len(signal) < window_size:
            return signal
        
        smoothed = []
        for i in range(len(signal)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(signal), i + window_size // 2 + 1)
            smoothed.append(sum(signal[start_idx:end_idx]) / (end_idx - start_idx))
        
        return smoothed


class ColorUtils:
    """Color utility functions for visualization."""
    
    # Color definitions
    COLORS = {
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'orange': (0, 165, 255),
        'purple': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
        'light_gray': (200, 200, 200),
        'dark_gray': (50, 50, 50)
    }
    
    @staticmethod
    def get_emotion_color(emotion: str) -> Tuple[int, int, int]:
        """Get color for emotion visualization."""
        emotion_colors = {
            'angry': ColorUtils.COLORS['red'],
            'disgust': ColorUtils.COLORS['orange'],
            'fear': ColorUtils.COLORS['purple'],
            'happy': ColorUtils.COLORS['green'],
            'sad': ColorUtils.COLORS['blue'],
            'surprise': ColorUtils.COLORS['yellow'],
            'neutral': ColorUtils.COLORS['gray']
        }
        return emotion_colors.get(emotion, ColorUtils.COLORS['white'])
    
    @staticmethod
    def get_attention_color(attention_state: str) -> Tuple[int, int, int]:
        """Get color for attention state visualization."""
        attention_colors = {
            'Focused': ColorUtils.COLORS['green'],
            'Partially Attentive': ColorUtils.COLORS['orange'],
            'Distracted': ColorUtils.COLORS['red'],
            'Unknown': ColorUtils.COLORS['gray']
        }
        return attention_colors.get(attention_state, ColorUtils.COLORS['white'])
    
    @staticmethod
    def get_posture_color(posture_state: str) -> Tuple[int, int, int]:
        """Get color for posture state visualization."""
        posture_colors = {
            'Excellent': ColorUtils.COLORS['green'],
            'Good': ColorUtils.COLORS['green'],
            'Fair': ColorUtils.COLORS['orange'],
            'Poor': ColorUtils.COLORS['red'],
            'Unknown': ColorUtils.COLORS['gray']
        }
        return posture_colors.get(posture_state, ColorUtils.COLORS['white'])


def create_timestamped_filename(prefix: str, extension: str, directory: str = ".") -> str:
    """Create a timestamped filename."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, f"{prefix}_{timestamp}.{extension}")


def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)
