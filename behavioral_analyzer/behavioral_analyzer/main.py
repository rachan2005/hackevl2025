"""
Main entry point for the Behavioral Analyzer.

This module provides the command-line interface and main execution logic
for the unified behavioral analysis system.
"""

import argparse
import sys
import os
import cv2
import time
import platform
from typing import Optional

from behavioral_analyzer import UnifiedBehavioralAnalyzer, Config
from behavioral_analyzer.config import VideoConfig, AudioConfig, OutputConfig


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified Behavioral Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  behavioral-analyzer                    # Run with default settings
  behavioral-analyzer --no-audio        # Video analysis only
  behavioral-analyzer --no-video        # Audio analysis only
  behavioral-analyzer --camera 1        # Use camera index 1
  behavioral-analyzer --config my.json  # Use custom configuration
  behavioral-analyzer --debug           # Enable debug mode
        """
    )
    
    # Analysis mode options
    parser.add_argument('--no-video', action='store_true', 
                       help='Disable video analysis')
    parser.add_argument('--no-audio', action='store_true', 
                       help='Disable audio analysis')
    
    # Camera options
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--resolution', type=str, default='640x480', 
                       help='Camera resolution (default: 640x480)')
    
    # Video analysis options
    parser.add_argument('--no-emotion', action='store_true', 
                       help='Disable emotion detection')
    parser.add_argument('--no-blink', action='store_true', 
                       help='Disable blink detection')
    parser.add_argument('--no-attention', action='store_true', 
                       help='Disable attention analysis')
    parser.add_argument('--no-posture', action='store_true', 
                       help='Disable posture analysis')
    parser.add_argument('--no-movement', action='store_true', 
                       help='Disable movement analysis')
    parser.add_argument('--no-fatigue', action='store_true', 
                       help='Disable fatigue detection')
    
    # Audio analysis options
    parser.add_argument('--model', type=str, default='tiny.en', 
                       choices=['tiny.en', 'small.en', 'base.en', 'medium.en'],
                       help='Whisper model size (default: tiny.en)')
    parser.add_argument('--device', type=str, default='cpu', 
                       choices=['cpu', 'cuda'],
                       help='Processing device (default: cpu)')
    parser.add_argument('--no-transcription', action='store_true', 
                       help='Disable speech transcription')
    parser.add_argument('--no-sentiment', action='store_true', 
                       help='Disable sentiment analysis')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='video', 
                       help='Output directory (default: video)')
    parser.add_argument('--save-video', action='store_true', 
                       help='Save video recording')
    parser.add_argument('--session-name', type=str, 
                       help='Custom session name')
    
    # Configuration options
    parser.add_argument('--config', type=str, 
                       help='Load configuration from JSON file')
    parser.add_argument('--save-config', type=str, 
                       help='Save current configuration to JSON file')
    
    # Debug and display options
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode')
    parser.add_argument('--show-landmarks', action='store_true', 
                       help='Show MediaPipe landmarks')
    parser.add_argument('--no-display', action='store_true', 
                       help='Run without display (headless mode)')
    
    # Performance options
    parser.add_argument('--low-latency', action='store_true', 
                       help='Optimize for low latency')
    parser.add_argument('--high-quality', action='store_true', 
                       help='Optimize for high quality')
    
    # API server options
    parser.add_argument('--enable-api', action='store_true', 
                       help='Enable HTTP API server for external integration')
    parser.add_argument('--api-port', type=int, default=8083, 
                       help='API server port (default: 8083)')
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> Config:
    """Create configuration from command line arguments."""
    # Load from file if specified
    if args.config:
        try:
            config = Config.load_from_file(args.config)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
            config = Config()
    else:
        config = Config()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        config.video.resolution = (width, height)
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}. Using default 640x480.")
        config.video.resolution = (640, 480)
    
    # Video analysis settings
    config.video.camera_id = args.camera
    config.video.enable_emotion = not args.no_emotion and not args.no_video
    config.video.enable_blink_detection = not args.no_blink and not args.no_video
    config.video.enable_attention_analysis = not args.no_attention and not args.no_video
    config.video.enable_posture_analysis = not args.no_posture and not args.no_video
    config.video.enable_movement_analysis = not args.no_movement and not args.no_video
    config.video.enable_fatigue_detection = not args.no_fatigue and not args.no_video
    config.video.show_landmarks = args.show_landmarks
    config.video.debug_mode = args.debug
    
    # Audio analysis settings
    config.audio.model = args.model
    config.audio.device = args.device
    config.audio.enable_transcription = not args.no_transcription and not args.no_audio
    config.audio.enable_sentiment_analysis = not args.no_sentiment and not args.no_audio
    config.audio.enable_emotion_detection = not args.no_audio
    
    # Output settings
    config.output.output_dir = args.output_dir
    config.output.save_video = args.save_video
    if args.session_name:
        config.session_name = args.session_name
    
    # Performance optimizations
    if args.low_latency:
        config.video.emotion_cooldown = 0.1
        config.audio.chunk_duration = 1.0
        config.video.resolution = (320, 240)
    elif args.high_quality:
        config.video.emotion_cooldown = 1.0
        config.audio.chunk_duration = 3.0
        config.video.resolution = (1280, 720)
    
    return config


def print_system_info():
    """Print system information."""
    print("=" * 60)
    print("Behavioral Analyzer - Unified Analysis System")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"OpenCV: {cv2.__version__}")
    print("=" * 60)


def print_usage_instructions():
    """Print usage instructions."""
    print("\nControls:")
    print("  'q' - Quit")
    print("  'd' - Toggle debug mode")
    print("  'l' - Toggle landmark display")
    print("  'r' - Start/stop recording")
    print("  's' - Save session data")
    print("  'h' - Show this help")
    print("\nSession data will be automatically saved when you quit.")
    print("=" * 60)


def handle_keyboard_input(key: int, analyzer: UnifiedBehavioralAnalyzer, 
                         frame: cv2.typing.MatLike) -> bool:
    """Handle keyboard input and return True if should continue."""
    if key == ord('q'):
        return False
    elif key == ord('d'):
        # Toggle debug mode
        analyzer.config.video.debug_mode = not analyzer.config.video.debug_mode
        analyzer.config.audio.debug_mode = analyzer.config.video.debug_mode
        print(f"Debug mode {'enabled' if analyzer.config.video.debug_mode else 'disabled'}")
    elif key == ord('l'):
        # Toggle landmark display
        analyzer.config.video.show_landmarks = not analyzer.config.video.show_landmarks
        print(f"Landmarks {'enabled' if analyzer.config.video.show_landmarks else 'disabled'}")
    elif key == ord('r'):
        # Toggle recording
        analyzer.toggle_recording(frame)
    elif key == ord('s'):
        # Save session data
        saved_file = analyzer.save_session_data()
        if saved_file:
            print(f"Session data saved to: {saved_file}")
        else:
            print("Failed to save session data")
    elif key == ord('h'):
        # Show help
        print_usage_instructions()
    
    return True


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print system info
        print_system_info()
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Save configuration if requested
        if args.save_config:
            config.save_to_file(args.save_config)
            print(f"Configuration saved to {args.save_config}")
            return 0
        
        # Validate configuration
        if args.no_video and args.no_audio:
            print("Error: Cannot disable both video and audio analysis")
            return 1
        
        # Print configuration summary
        print(f"Session: {config.session_name}")
        print(f"Camera: {config.video.camera_id}")
        print(f"Resolution: {config.video.resolution[0]}x{config.video.resolution[1]}")
        print(f"Video Analysis: {'Enabled' if not args.no_video else 'Disabled'}")
        print(f"Audio Analysis: {'Enabled' if not args.no_audio else 'Disabled'}")
        print(f"Output Directory: {config.output.output_dir}")
        print(f"Debug Mode: {'Enabled' if config.video.debug_mode else 'Disabled'}")
        
        # Initialize analyzer
        analyzer = UnifiedBehavioralAnalyzer(
            config, 
            enable_api=args.enable_api, 
            api_port=args.api_port
        )
        
        # Start analysis
        if not analyzer.start_analysis():
            print("Failed to start analysis")
            return 1
        
        # Print usage instructions
        print_usage_instructions()
        
        # Main processing loop
        consecutive_failures = 0
        max_failures = 5
        
        try:
            while True:
                # Process frame
                success, frame = analyzer.process_frame()
                
                if not success:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("Too many consecutive failures. Exiting.")
                        break
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter
                consecutive_failures = 0
                
                # Display frame if not in headless mode
                if not args.no_display:
                    cv2.imshow('Behavioral Analysis', frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if not handle_keyboard_input(key, analyzer, frame):
                        break
                else:
                    # In headless mode, just sleep briefly
                    time.sleep(0.033)  # ~30 FPS
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # Cleanup
            analyzer.cleanup()
            if not args.no_display:
                cv2.destroyAllWindows()
            
            # Print session summary
            summary = analyzer.get_session_summary()
            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print("=" * 60)
            print(f"Duration: {summary['session_info']['duration_formatted']}")
            print(f"Final Emotion: {summary['unified_analysis']['final_emotion']}")
            print(f"Final Attention: {summary['unified_analysis']['final_attention']}")
            print(f"Final Fatigue: {summary['unified_analysis']['final_fatigue']}")
            print(f"Overall Sentiment: {summary['unified_analysis']['overall_sentiment']:.2f}")
            print(f"Average FPS: {summary['performance']['average_fps']:.1f}")
            print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

