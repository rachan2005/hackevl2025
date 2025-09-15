#!/usr/bin/env python3
"""
Startup script for the ADK Voice Agent
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

def run_adk_voice_agent():
    """Run the ADK voice agent"""
    print("🎤 Starting ADK Voice Agent...")
    
    # Change to server directory
    server_dir = Path(__file__).parent / "server"
    os.chdir(server_dir)
    
    # Run the ADK voice agent
    cmd = ["uv", "run", "python", "server.py"]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(f"[ADK Voice Agent] {line.rstrip()}")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping ADK Voice Agent...")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"❌ Error running ADK voice agent: {e}")

if __name__ == "__main__":
    run_adk_voice_agent()
