"""
Web UI Module for Behavioral Analyzer.

This module provides a real-time web dashboard for monitoring
behavioral analysis results including video, audio, and object detection data.
"""

import json
import time
import threading
from typing import Dict, Any, Optional
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np

from .config import Config
from .unified_analyzer import UnifiedBehavioralAnalyzer


class BehavioralWebUI:
    """
    Web-based dashboard for real-time behavioral analysis monitoring.
    
    Features:
    - Real-time video feed
    - Live emotion, attention, and posture data
    - Audio transcription and sentiment analysis
    - Object detection results
    - Interactive charts and graphs
    - Session statistics
    """
    
    def __init__(self, config: Config, analyzer: Optional[UnifiedBehavioralAnalyzer] = None):
        """
        Initialize the web UI.
        
        Args:
            config: Configuration object
            analyzer: Optional existing analyzer instance
        """
        self.config = config
        self.analyzer = analyzer
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'behavioral_analyzer_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Data storage
        self.latest_data = {
            'video': {},
            'audio': {},
            'objects': {},
            'session_stats': {},
            'timestamp': time.time()
        }
        
        # Video streaming
        self.video_streaming = False
        self.video_thread = None
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
        # Start data collection thread
        self.data_thread = None
        self.data_thread_active = False
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/data')
        def get_data():
            """Get current analysis data."""
            return jsonify(self.latest_data)
        
        @self.app.route('/api/config')
        def get_config():
            """Get current configuration."""
            return jsonify({
                'video_enabled': self.config.video.enable_emotion,
                'audio_enabled': self.config.audio.enable_transcription,
                'object_detection_enabled': self.config.video.enable_object_detection,
                'debug_mode': self.config.video.debug_mode
            })
        
        @self.app.route('/api/controls', methods=['POST'])
        def update_controls():
            """Update analyzer controls."""
            data = request.get_json()
            
            if 'debug_mode' in data:
                self.config.video.debug_mode = data['debug_mode']
            
            if 'show_landmarks' in data:
                self.config.video.show_landmarks = data['show_landmarks']
            
            if 'show_objects' in data:
                self.config.video.show_object_detections = data['show_objects']
            
            return jsonify({'status': 'success'})
    
    def _setup_socketio_events(self):
        """Setup SocketIO events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            print(f"Client connected: {request.sid}")
            emit('status', {'message': 'Connected to Behavioral Analyzer'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_data')
        def handle_data_request():
            """Handle data request from client."""
            emit('data_update', self.latest_data)
        
        @self.socketio.on('toggle_video_stream')
        def handle_video_toggle(data):
            """Toggle video streaming."""
            self.video_streaming = data.get('enabled', False)
            if self.video_streaming and not self.video_thread:
                self._start_video_stream()
            elif not self.video_streaming and self.video_thread:
                self._stop_video_stream()
    
    def _start_video_stream(self):
        """Start video streaming thread."""
        if self.analyzer and hasattr(self.analyzer, 'video_analyzer'):
            self.video_thread = threading.Thread(target=self._video_stream_worker, daemon=True)
            self.video_thread.start()
    
    def _stop_video_stream(self):
        """Stop video streaming thread."""
        self.video_streaming = False
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
            self.video_thread = None
    
    def _video_stream_worker(self):
        """Video streaming worker thread."""
        while self.video_streaming and self.analyzer:
            try:
                # Get current frame from analyzer
                if hasattr(self.analyzer, 'video_analyzer') and hasattr(self.analyzer.video_analyzer, 'current_frame'):
                    frame = self.analyzer.video_analyzer.current_frame
                    if frame is not None:
                        # Encode frame as JPEG
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Emit frame to all connected clients
                        self.socketio.emit('video_frame', {
                            'frame': frame_base64,
                            'timestamp': time.time()
                        })
                
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Video streaming error: {e}")
                time.sleep(0.1)
    
    def _start_data_collection(self):
        """Start data collection thread."""
        self.data_thread_active = True
        self.data_thread = threading.Thread(target=self._data_collection_worker, daemon=True)
        self.data_thread.start()
    
    def _stop_data_collection(self):
        """Stop data collection thread."""
        self.data_thread_active = False
        if self.data_thread:
            self.data_thread.join(timeout=1.0)
            self.data_thread = None
    
    def _data_collection_worker(self):
        """Data collection worker thread."""
        while self.data_thread_active and self.analyzer:
            try:
                # Collect data from analyzer
                self._collect_analyzer_data()
                
                # Emit data to connected clients
                self.socketio.emit('data_update', self.latest_data)
                
                time.sleep(0.1)  # 10 Hz update rate
            except Exception as e:
                print(f"Data collection error: {e}")
                time.sleep(0.1)
    
    def _collect_analyzer_data(self):
        """Collect data from the analyzer."""
        if not self.analyzer:
            return
        
        current_time = time.time()
        
        # Video analysis data
        video_data = {}
        if hasattr(self.analyzer, 'video_analyzer'):
            va = self.analyzer.video_analyzer
            video_data = {
                'emotion': getattr(va, 'current_emotion', 'Unknown'),
                'emotion_scores': getattr(va, 'emotion_scores', {}),
                'attention_state': getattr(va, 'attention_state', 'Unknown'),
                'posture_state': getattr(va, 'posture_state', 'Unknown'),
                'movement_level': getattr(va, 'movement_level', 'Unknown'),
                'fatigue_level': getattr(va, 'fatigue_level', 'Normal'),
                'blink_rate': getattr(va, 'blink_count', 0) / max(1, current_time - getattr(va, 'last_reset_time', current_time)) * 60.0,
                'total_blinks': getattr(va, 'total_blink_count', 0),
                'fps': getattr(va.performance_tracker, 'current_fps', 0.0) if hasattr(va, 'performance_tracker') else 0.0,
                'person_tracking': getattr(va, '_get_person_tracking_data', lambda: {})(),
                'main_person': getattr(va, 'main_person', None)
            }
        
        # Audio analysis data
        audio_data = {}
        if hasattr(self.analyzer, 'audio_analyzer'):
            aa = self.analyzer.audio_analyzer
            audio_data = {
                'transcription': getattr(aa, 'current_transcription', ''),
                'emotion': getattr(aa, 'current_emotion', 'neutral'),
                'sentiment': getattr(aa, 'current_sentiment', 0.0),
                'confidence': getattr(aa, 'current_confidence', 0.0),
                'audio_features': getattr(aa, 'audio_features', {}),
                'session_stats': getattr(aa, 'session_stats', {})
            }
        
        # Object detection data
        objects_data = {}
        if hasattr(self.analyzer, 'video_analyzer') and hasattr(self.analyzer.video_analyzer, 'object_detector'):
            od = self.analyzer.video_analyzer.object_detector
            if od:
                objects_data = {
                    'detections': getattr(self.analyzer.video_analyzer, 'current_detections', []),
                    'summary': od.get_detection_summary(),
                    'object_counts': getattr(od, 'object_counts', {})
                }
        
        # Session statistics
        session_stats = {
            'session_duration': current_time - getattr(self.analyzer, 'session_start_time', current_time),
            'total_frames_processed': getattr(self.analyzer, 'total_frames_processed', 0),
            'timestamp': current_time
        }
        
        # Update latest data
        self.latest_data = {
            'video': video_data,
            'audio': audio_data,
            'objects': objects_data,
            'session_stats': session_stats,
            'timestamp': current_time
        }
    
    def start_analyzer(self):
        """Start the behavioral analyzer."""
        if not self.analyzer:
            self.analyzer = UnifiedBehavioralAnalyzer(self.config)
        
        if not self.analyzer.start_analysis():
            print("Failed to start analyzer")
            return False
        
        # Start data collection
        self._start_data_collection()
        
        return True
    
    def stop_analyzer(self):
        """Stop the behavioral analyzer."""
        if self.analyzer:
            self.analyzer.cleanup()
        
        # Stop data collection
        self._stop_data_collection()
        
        # Stop video streaming
        self._stop_video_stream()
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """
        Run the web UI server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        print(f"Starting Behavioral Analyzer Web UI on http://{host}:{port}")
        print("Press Ctrl+C to stop")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            print("\nShutting down web UI...")
            self.stop_analyzer()


def create_web_ui(config: Config, analyzer: Optional[UnifiedBehavioralAnalyzer] = None) -> BehavioralWebUI:
    """
    Create a web UI instance.
    
    Args:
        config: Configuration object
        analyzer: Optional existing analyzer instance
        
    Returns:
        BehavioralWebUI instance
    """
    return BehavioralWebUI(config, analyzer)
