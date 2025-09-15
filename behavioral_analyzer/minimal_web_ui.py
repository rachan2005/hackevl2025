#!/usr/bin/env python3
"""
Minimal Web UI for Behavioral Analyzer

Uses simple Flask with AJAX polling instead of SocketIO to avoid event handler issues.
"""

import json
import time
import os
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_cors import CORS

class MinimalWebUI:
    """Minimal web UI that reads data from analyzer output file."""
    
    def __init__(self, data_file='video/analyzer_data.json'):
        self.data_file = data_file
        self.latest_data = {}
        self.app = Flask(__name__, 
                         template_folder='templates',
                         static_folder='static')
        CORS(self.app)  # Enable CORS for AJAX requests
        
        # Setup routes
        self._setup_routes()
        
        # Start data reading thread
        self.data_thread_active = True
        self.data_thread = threading.Thread(target=self._data_reader_worker, daemon=True)
        self.data_thread.start()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template('minimal_dashboard.html')
        
        @self.app.route('/api/data')
        def get_data():
            """API endpoint to get current data."""
            return jsonify(self.latest_data)
        
        @self.app.route('/api/status')
        def get_status():
            """API endpoint to get status."""
            file_exists = os.path.exists(self.data_file)
            return jsonify({
                'status': 'running' if file_exists else 'waiting',
                'data_file': self.data_file,
                'file_exists': file_exists,
                'timestamp': time.time()
            })

        @self.app.route('/api/calibration')
        def get_calibration():
            """Get detailed calibration information."""
            try:
                if os.path.exists(self.data_file):
                    with open(self.data_file, 'r') as f:
                        data = json.load(f)
                    
                    calibration_data = data.get('calibration', {})
                    return jsonify(calibration_data)
                else:
                    return jsonify({
                        'is_calibrating': False,
                        'progress_percent': 0.0,
                        'status': 'No data available'
                    })
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'is_calibrating': False,
                    'progress_percent': 0.0,
                    'status': 'Error'
                })
    
    def _data_reader_worker(self):
        """Worker thread that reads data from the analyzer output file."""
        print("Data reader worker started")
        
        while self.data_thread_active:
            try:
                if os.path.exists(self.data_file):
                    # Read data from file
                    with open(self.data_file, 'r') as f:
                        file_data = json.load(f)
                    
                    # Debug: Log complete data structure
                    print(f"Complete data structure: {file_data}")
                    
                    # Debug: Log calibration data
                    if 'calibration' in file_data:
                        print(f"Calibration data found: {file_data['calibration']}")
                    else:
                        print("No calibration data in file")
                    
                    # Debug: Log video calibrated data
                    if 'video' in file_data and 'blink_rate_calibrated' in file_data['video']:
                        print(f"Video calibrated data: {file_data['video']['blink_rate_calibrated']}")
                    else:
                        print("No video calibrated data found")
                    
                    # Debug: Log audio calibrated data  
                    if 'audio' in file_data and 'sentiment_calibrated' in file_data['audio']:
                        print(f"Audio calibrated data: {file_data['audio']['sentiment_calibrated']}")
                    else:
                        print("No audio calibrated data found")
                    
                    # Update latest data
                    self.latest_data = file_data
                    print(f"Data updated: {datetime.now().strftime('%H:%M:%S')}")
                else:
                    # Create dummy data if file doesn't exist
                    print(f"Data file does not exist: {self.data_file}")
                    self.latest_data = {
                        'video': {
                            'emotion': 'Waiting for analyzer...',
                            'fps': 0,
                            'blink_rate': 0,
                            'attention_state': 'Unknown',
                            'posture_state': 'Unknown',
                            'fatigue_level': 'Normal',
                            'total_blinks': 0
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
                        },
                        'calibration': {
                            'is_calibrating': False,
                            'progress_percent': 0.0,
                            'status': 'Waiting for analyzer...'
                        }
                    }
                
                time.sleep(0.5)  # Read every 500ms
                
            except Exception as e:
                print(f"Data reader error: {e}")
                time.sleep(1.0)
    
    def run(self, host='0.0.0.0', port=5002, debug=False):
        """Run the web UI server."""
        print(f"Starting Minimal Web UI on http://{host}:{port}")
        print("Data file path:", os.path.abspath(self.data_file))
        self.app.run(host=host, port=port, debug=debug, threaded=True)
    
    def stop(self):
        """Stop the web UI."""
        self.data_thread_active = False
        if self.data_thread:
            self.data_thread.join(timeout=1.0)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Minimal Behavioral Analyzer Web UI')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5002, help='Port to bind to')
    parser.add_argument('--data-file', default='video/analyzer_data.json', help='Data file to read from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("Minimal Behavioral Analyzer Web UI")
    print("="*50)
    
    web_ui = MinimalWebUI(data_file=args.data_file)
    
    try:
        web_ui.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        web_ui.stop()
        print("Web UI stopped.")


if __name__ == "__main__":
    main()
