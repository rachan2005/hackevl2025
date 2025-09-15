#!/usr/bin/env python3
"""
Behavioral Analyzer Web UI Launcher

This script launches the web-based dashboard for the behavioral analyzer.
"""

import argparse
import sys
import os

# Add the behavioral_analyzer package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from behavioral_analyzer import Config, UnifiedBehavioralAnalyzer
from behavioral_analyzer.web_ui import BehavioralWebUI


def main():
    """Main function to launch the web UI."""
    parser = argparse.ArgumentParser(description='Behavioral Analyzer Web UI')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--no-analyzer', action='store_true', help='Run web UI without starting analyzer')
    parser.add_argument('--camera', type=int, help='Camera index (overrides config)')
    parser.add_argument('--no-video', action='store_true', help='Disable video analysis')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio analysis')
    parser.add_argument('--no-objects', action='store_true', help='Disable object detection')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Behavioral Analyzer Web UI")
    print("=" * 60)
    
    # Load configuration
    config = Config()
    if args.config:
        try:
            config = Config.load_from_file(args.config)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
            config = Config()
    
    # Apply command line overrides
    if args.camera is not None:
        config.video.camera_id = args.camera
    if args.no_video:
        config.video.enable_emotion = False
        config.video.enable_blink_detection = False
        config.video.enable_attention_analysis = False
        config.video.enable_posture_analysis = False
        config.video.enable_movement_analysis = False
        config.video.enable_fatigue_detection = False
    if args.no_audio:
        config.audio.enable_transcription = False
        config.audio.enable_emotion_detection = False
    if args.no_objects:
        config.video.enable_object_detection = False
    
    # Create analyzer if not disabled
    analyzer = None
    if not args.no_analyzer:
        print("Initializing Behavioral Analyzer...")
        analyzer = UnifiedBehavioralAnalyzer(config)
    
    # Create web UI
    print("Initializing Web UI...")
    web_ui = BehavioralWebUI(config, analyzer)
    
    # Start analyzer if available
    if analyzer:
        print("Starting Behavioral Analyzer...")
        if not web_ui.start_analyzer():
            print("Failed to start analyzer. Continuing with web UI only.")
    
    print(f"Web UI will be available at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        # Run the web UI
        web_ui.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if analyzer:
            web_ui.stop_analyzer()
        print("Web UI stopped.")


if __name__ == "__main__":
    main()
