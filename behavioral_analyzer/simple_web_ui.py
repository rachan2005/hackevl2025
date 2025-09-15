#!/usr/bin/env python3
"""
Simple Decoupled Web UI for Behavioral Analyzer

This web UI reads data from a JSON file that the behavioral analyzer writes to,
providing a clean separation between the analyzer and web interface.
"""

import json
import time
import os
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

class SimpleWebUI:
    """Simple web UI that reads data from analyzer output file."""
    
    def __init__(self, data_file='analyzer_data.json'):
        self.data_file = data_file
        self.latest_data = {}
        self.app = Flask(__name__, 
                         template_folder='templates',
                         static_folder='static')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes and events
        self._setup_routes()
        self._setup_socketio_events()
        
        # Start data reading thread
        self.data_thread_active = True
        self.data_thread = threading.Thread(target=self._data_reader_worker, daemon=True)
        self.data_thread.start()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/data')
        def get_data():
            """API endpoint to get current data."""
            return jsonify(self.latest_data)
    
    def _setup_socketio_events(self):
        """Setup SocketIO events."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            print(f"Client connected: {request.sid}")
            emit('status', {'message': 'Connected to Behavioral Analyzer Dashboard'})
            # Send current data immediately
            emit('data_update', self.latest_data)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_data')
        def handle_data_request():
            """Handle data request from client."""
            emit('data_update', self.latest_data)
    
    def _data_reader_worker(self):
        """Worker thread that reads data from the analyzer output file."""
        print("Data reader worker started")
        
        while self.data_thread_active:
            try:
                if os.path.exists(self.data_file):
                    # Read data from file
                    with open(self.data_file, 'r') as f:
                        file_data = json.load(f)
                    
                    # Update latest data
                    self.latest_data = file_data
                    
                    # Emit to connected clients
                    self.socketio.emit('data_update', self.latest_data)
                    print(f"Data updated: {datetime.now().strftime('%H:%M:%S')}")
                else:
                    # Create dummy data if file doesn't exist
                    self.latest_data = {
                        'video': {
                            'emotion': 'Waiting for analyzer...',
                            'fps': 0,
                            'blink_rate': 0,
                            'attention_state': 'Unknown',
                            'posture_state': 'Unknown',
                            'fatigue_level': 'Normal'
                        },
                        'audio': {
                            'transcription': 'Waiting for analyzer...',
                            'emotion': 'neutral',
                            'sentiment': 0.0,
                            'confidence': 0.0
                        },
                        'objects': {
                            'detections': []
                        },
                        'session_stats': {
                            'session_duration': 0,
                            'timestamp': time.time()
                        }
                    }
                    self.socketio.emit('data_update', self.latest_data)
                
                time.sleep(0.5)  # Read every 500ms
                
            except Exception as e:
                print(f"Data reader error: {e}")
                time.sleep(1.0)
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the web UI server."""
        print(f"Starting Simple Web UI on http://{host}:{port}")
        print("Make sure the behavioral analyzer is writing to:", os.path.abspath(self.data_file))
        self.socketio.run(self.app, host=host, port=port, debug=debug)
    
    def stop(self):
        """Stop the web UI."""
        self.data_thread_active = False
        if self.data_thread:
            self.data_thread.join(timeout=1.0)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Behavioral Analyzer Web UI')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--data-file', default='analyzer_data.json', help='Data file to read from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🌐 Simple Behavioral Analyzer Web UI")
    print("="*50)
    
    web_ui = SimpleWebUI(data_file=args.data_file)
    
    try:
        web_ui.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        web_ui.stop()
        print("Web UI stopped.")


if __name__ == "__main__":
    main()
