"""
Unified Behavioral Analyzer - Combines video and audio analysis.

This module provides a unified interface for comprehensive behavioral analysis
combining both video and audio modalities for enhanced insights.
"""

import cv2
import time
import threading
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .config import Config
from .db import MongoPersistence
from .video_analyzer import VideoAnalyzer
from .audio_analyzer import AudioAnalyzer
from .utils import (
    DataCollector, PerformanceTracker, ColorUtils, 
    ensure_directory_exists, create_timestamped_filename
)


class UnifiedBehavioralAnalyzer:
    """Unified analyzer combining video and audio behavioral analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize data collection
        self.data_collector = DataCollector(
            session_name=config.session_name,
            start_time=time.time()
        )
        
        # Initialize performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Initialize analyzers
        self.video_analyzer = VideoAnalyzer(config.video, self.data_collector)
        self.audio_analyzer = AudioAnalyzer(config.audio, self.data_collector)
        
        # Unified state
        self.unified_state = {
            'emotion': 'unknown',
            'attention': 'unknown',
            'posture': 'unknown',
            'movement': 'unknown',
            'fatigue': 'normal',
            'transcription': '',
            'sentiment': 0.0,
            'confidence': 'low'
        }
        
        # Session management
        self.session_start_time = time.time()
        self.is_running = False
        
        # Video recording
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        # Optional MongoDB persistence
        self.mongo: Optional[MongoPersistence] = None
        if getattr(self.config, 'db', None) and self.config.db.enabled and self.config.db.uri:
            try:
                self.mongo = MongoPersistence(
                    uri=self.config.db.uri,
                    database=self.config.db.database,
                    combined_collection=self.config.db.combined_collection,
                    sessions_collection=self.config.db.sessions_collection,
                )
                print("MongoDB persistence enabled")
            except Exception as e:
                print(f"MongoDB disabled due to error: {e}")

        print("Unified Behavioral Analyzer initialized successfully")
    
    def start_analysis(self, camera_id: Optional[int] = None) -> bool:
        """Start the unified analysis system."""
        try:
            # Set camera ID if provided
            if camera_id is not None:
                self.config.video.camera_id = camera_id
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.config.video.camera_id)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.config.video.camera_id}")
                return False
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video.resolution[1])
            
            # Start audio analysis
            if self.config.audio.enable_transcription:
                self.audio_analyzer.start()
            
            self.is_running = True
            print("Unified analysis started successfully")
            return True
            
        except Exception as e:
            print(f"Error starting analysis: {e}")
            return False
    
    def process_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Process a single frame and return annotated frame."""
        if not self.is_running:
            return False, None
        
        try:
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                return False, None
            
            # Mirror frame horizontally for natural webcam view
            frame = cv2.flip(frame, 1)
            
            # Process with video analyzer
            annotated_frame = self.video_analyzer.analyze_frame(frame)
            
            # Update unified state
            self._update_unified_state()
            
            # Add unified analysis overlay
            self._add_unified_overlay(annotated_frame)
            
            # Handle video recording
            if self.recording and self.video_writer is not None:
                self.video_writer.write(annotated_frame)
            
            # Update performance metrics
            process_time = time.time() - self.session_start_time
            self.performance_tracker.update(process_time)
            
            return True, annotated_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return False, None
    
    def _update_unified_state(self):
        """Update unified state from individual analyzers."""
        # Get video analyzer state
        video_state = {
            'emotion': self.video_analyzer.current_emotion,
            'attention': self.video_analyzer.attention_state,
            'posture': self.video_analyzer.posture_state,
            'movement': self.video_analyzer.movement_level,
            'fatigue': self.video_analyzer.fatigue_level
        }
        
        # Get audio analyzer state
        audio_state = self.audio_analyzer.get_current_state()
        
        # Combine states with priority logic
        self.unified_state.update({
            'emotion': self._combine_emotions(video_state['emotion'], audio_state['emotion']),
            'attention': video_state['attention'],
            'posture': video_state['posture'],
            'movement': video_state['movement'],
            'fatigue': self._combine_fatigue(video_state['fatigue'], audio_state),
            'transcription': audio_state['transcription'],
            'sentiment': audio_state['sentiment'],
            'confidence': audio_state['confidence']
        })
        
        # Collect combined data
        self._collect_combined_data()
    
    def _combine_emotions(self, video_emotion: str, audio_emotion: str) -> str:
        """Combine video and audio emotions with intelligent fusion."""
        # If both are unknown, return unknown
        if video_emotion == "Unknown" and audio_emotion == "neutral":
            return "unknown"
        
        # If video emotion is unknown, use audio
        if video_emotion == "Unknown":
            return audio_emotion
        
        # If audio emotion is neutral, use video
        if audio_emotion == "neutral":
            return video_emotion
        
        # If both are detected, use video as primary (more reliable for facial expressions)
        # but consider audio as confirmation
        if video_emotion == audio_emotion:
            return video_emotion  # High confidence
        
        # If they differ, use video as primary but note the discrepancy
        return video_emotion
    
    def _combine_fatigue(self, video_fatigue: str, audio_state: Dict[str, Any]) -> str:
        """Combine fatigue indicators from video and audio."""
        # Video fatigue is primary
        base_fatigue = video_fatigue
        
        # Audio can provide additional context
        audio_energy = audio_state.get('audio_features', {}).get('energy', 0)
        audio_silence = audio_state.get('audio_features', {}).get('silence_ratio', 0)
        
        # Low energy and high silence might indicate fatigue
        if audio_energy < 0.01 and audio_silence > 0.5:
            if base_fatigue == "Normal":
                return "Mild"
            elif base_fatigue == "Mild":
                return "Moderate"
        
        return base_fatigue
    
    def _collect_combined_data(self):
        """Collect combined analysis data."""
        current_time = time.time()
        
        combined_data = {
            'timestamp': current_time,
            'unified_state': self.unified_state.copy(),
            'video_metrics': {
                'emotion': self.video_analyzer.current_emotion,
                'attention': self.video_analyzer.attention_state,
                'posture': self.video_analyzer.posture_state,
                'movement': self.video_analyzer.movement_level,
                'fatigue': self.video_analyzer.fatigue_level,
                'blink_rate': self.video_analyzer.blink_count / max(1, time.time() - self.video_analyzer.last_reset_time) * 60.0,
                'total_blinks': self.video_analyzer.total_blink_count
            },
            'audio_metrics': self.audio_analyzer.get_current_state(),
            'performance': self.performance_tracker.get_stats()
        }
        
        self.data_collector.add_combined_data(combined_data)
        # Persist to Mongo if enabled
        if self.mongo is not None:
            try:
                session_name = getattr(self.config, 'session_name', None)
                doc = {
                    'session_name': session_name,
                    **combined_data,
                }
                self.mongo.insert_combined(doc)
            except Exception as e:
                print(f"Mongo combined insert error: {e}")
    
    def _add_unified_overlay(self, frame: np.ndarray):
        """Add unified analysis overlay to frame."""
        h, w, _ = frame.shape
        
        # Create unified status panel
        panel_x = w - 300
        panel_y = 10
        panel_width = 290
        panel_height = 200
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (100, 100, 100), 1)
        
        # Title
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + 30), (40, 40, 40), -1)
        cv2.putText(frame, "Unified Analysis", (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display unified state
        y_offset = panel_y + 50
        
        # Emotion
        emotion_color = ColorUtils.get_emotion_color(self.unified_state['emotion'])
        cv2.putText(frame, f"Emotion: {self.unified_state['emotion']}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 1)
        y_offset += 25
        
        # Attention
        attention_color = ColorUtils.get_attention_color(self.unified_state['attention'])
        cv2.putText(frame, f"Attention: {self.unified_state['attention']}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, attention_color, 1)
        y_offset += 25
        
        # Fatigue
        fatigue_color = ColorUtils.get_emotion_color(self.unified_state['fatigue'])
        cv2.putText(frame, f"Fatigue: {self.unified_state['fatigue']}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fatigue_color, 1)
        y_offset += 25
        
        # Transcription (if available)
        if self.unified_state['transcription']:
            transcription_text = self.unified_state['transcription'][:30] + "..." if len(self.unified_state['transcription']) > 30 else self.unified_state['transcription']
            cv2.putText(frame, f"Speech: {transcription_text}", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
        
        # Sentiment
        sentiment = self.unified_state['sentiment']
        sentiment_color = (0, 255, 0) if sentiment > 0 else (0, 0, 255) if sentiment < 0 else (128, 128, 128)
        cv2.putText(frame, f"Sentiment: {sentiment:.2f}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, sentiment_color, 1)
        y_offset += 20
        
        # Confidence
        confidence_color = (0, 255, 0) if self.unified_state['confidence'] == 'high' else (0, 165, 255) if self.unified_state['confidence'] == 'medium' else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {self.unified_state['confidence']}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, confidence_color, 1)
        
        # Add recording indicator
        if self.recording:
            rec_time = time.time() - self.recording_start_time
            minutes, seconds = divmod(int(rec_time), 60)
            
            # Flashing red circle
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            
            cv2.putText(frame, f"REC {minutes:02d}:{seconds:02d}", 
                       (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def toggle_recording(self, frame: np.ndarray):
        """Start or stop video recording."""
        if not self.recording:
            # Start recording
            try:
                ensure_directory_exists(self.config.output.output_dir)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_file = os.path.join(
                    self.config.output.output_dir, 
                    f"unified_recording_{timestamp}.mp4"
                )
                
                h, w, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*self.config.output.video_codec)
                self.video_writer = cv2.VideoWriter(
                    video_file, fourcc, self.config.output.video_fps, (w, h)
                )
                
                self.recording = True
                self.recording_start_time = time.time()
                print(f"Recording started: {video_file}")
                
            except Exception as e:
                print(f"Error starting recording: {e}")
                self.recording = False
        else:
            # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            recording_duration = time.time() - self.recording_start_time
            print(f"Recording stopped after {recording_duration:.1f} seconds")
            self.recording = False
    
    def save_session_data(self) -> Optional[str]:
        """Save session data to JSON file."""
        try:
            # Update performance metrics
            self.data_collector.update_performance(self.performance_tracker.get_stats())
            
            # Generate filename
            if self.config.output.json_filename:
                filename = os.path.join(self.config.output.output_dir, self.config.output.json_filename)
            else:
                filename = create_timestamped_filename(
                    "unified_session", "json", self.config.output.output_dir
                )
            
            # Save data
            saved_file = self.data_collector.save_to_file(filename)
            if saved_file:
                print(f"Session data saved to: {saved_file}")
                return saved_file
            else:
                print("Failed to save session data")
                return None
                
        except Exception as e:
            print(f"Error saving session data: {e}")
            return None
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        session_duration = time.time() - self.session_start_time
        
        # Get video statistics
        video_stats = {
            'total_blinks': self.video_analyzer.total_blink_count,
            'blink_rate': self.video_analyzer.total_blink_count / max(1, session_duration / 60),
            'fatigue_level': self.video_analyzer.fatigue_level,
            'attention_distribution': self._get_attention_distribution(),
            'posture_distribution': self._get_posture_distribution()
        }
        
        # Get audio statistics
        audio_stats = self.audio_analyzer.get_session_statistics()
        
        # Get performance metrics
        performance_stats = self.performance_tracker.get_stats()
        
        return {
            'session_info': {
                'name': self.config.session_name,
                'duration_seconds': session_duration,
                'duration_formatted': self._format_duration(session_duration),
                'start_time': self.session_start_time,
                'end_time': time.time()
            },
            'video_analysis': video_stats,
            'audio_analysis': audio_stats,
            'unified_analysis': {
                'final_emotion': self.unified_state['emotion'],
                'final_attention': self.unified_state['attention'],
                'final_fatigue': self.unified_state['fatigue'],
                'overall_sentiment': self.unified_state['sentiment']
            },
            'performance': performance_stats,
            'configuration': self.config.to_dict()
        }
    
    def _get_attention_distribution(self) -> Dict[str, int]:
        """Get distribution of attention states."""
        # This would be populated from video analyzer history
        return {
            'Focused': 0,
            'Partially Attentive': 0,
            'Distracted': 0,
            'Unknown': 0
        }
    
    def _get_posture_distribution(self) -> Dict[str, int]:
        """Get distribution of posture states."""
        # This would be populated from video analyzer history
        return {
            'Excellent': 0,
            'Good': 0,
            'Fair': 0,
            'Poor': 0,
            'Unknown': 0
        }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def stop_analysis(self):
        """Stop the unified analysis system."""
        self.is_running = False
        
        # Stop audio analysis
        self.audio_analyzer.stop()
        
        # Clean up video analyzer
        self.video_analyzer.cleanup()
        
        # Release camera
        if hasattr(self, 'cap'):
            self.cap.release()
        
        # Stop recording if active
        if self.recording:
            self.toggle_recording(None)
        
        # Save session data
        saved_file = self.save_session_data()
        
        # Persist session summary to Mongo
        if self.mongo is not None:
            try:
                summary = self.get_session_summary()
                summary['artifacts'] = {'json_file': saved_file}
                self.mongo.insert_session(summary)
            except Exception as e:
                print(f"Mongo session insert error: {e}")
        
        print("Unified analysis stopped")
    
    def cleanup(self):
        """Clean up all resources."""
        self.stop_analysis()
        self.audio_analyzer.cleanup()
        print("Unified analyzer cleaned up")
