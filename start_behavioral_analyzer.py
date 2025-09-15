#!/usr/bin/env python3
"""
Startup script for the Behavioral Analyzer with API server
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

def run_behavioral_analyzer():
    """Run the behavioral analyzer with API enabled"""
    print("🚀 Starting Behavioral Analyzer with API server...")
    
    # Change to behavioral analyzer directory
    analyzer_dir = Path(__file__).parent / "behavioral_analyzer"
    os.chdir(analyzer_dir)
    
    # Run the behavioral analyzer with API enabled
    cmd = [
        "uv", "run", "python", "behavioral_analyzer/main.py",
        "--enable-api",
        "--api-port", "8083"
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(f"[Behavioral Analyzer] {line.rstrip()}")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Behavioral Analyzer...")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"❌ Error running behavioral analyzer: {e}")

if __name__ == "__main__":
    run_behavioral_analyzer()
