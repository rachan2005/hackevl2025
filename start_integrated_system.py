#!/usr/bin/env python3
"""
Startup script for the integrated ADK Voice Agent + Behavioral Analyzer system
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

class IntegratedSystem:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def run_behavioral_analyzer(self):
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
            self.processes.append(("Behavioral Analyzer", process))
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if self.running:
                    print(f"[Behavioral Analyzer] {line.rstrip()}")
                else:
                    break
            
        except Exception as e:
            print(f"❌ Error running behavioral analyzer: {e}")
    
    def run_adk_voice_agent(self):
        """Run the ADK voice agent"""
        print("🎤 Starting ADK Voice Agent...")
        
        # Change to server directory
        server_dir = Path(__file__).parent / "server"
        os.chdir(server_dir)
        
        # Run the ADK voice agent
        cmd = ["uv", "run", "python", "server.py"]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.processes.append(("ADK Voice Agent", process))
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if self.running:
                    print(f"[ADK Voice Agent] {line.rstrip()}")
                else:
                    break
            
        except Exception as e:
            print(f"❌ Error running ADK voice agent: {e}")
    
    def run_client_web_server(self):
        """Run the client web server"""
        print("🌐 Starting Client Web Server...")
        
        # Change to client directory
        client_dir = Path(__file__).parent / "client"
        os.chdir(client_dir)
        
        # Run the web server
        cmd = ["python3", "-m", "http.server", "8080"]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.processes.append(("Client Web Server", process))
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if self.running:
                    print(f"[Client Web Server] {line.rstrip()}")
                else:
                    break
            
        except Exception as e:
            print(f"❌ Error running client web server: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}. Shutting down integrated system...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up all processes"""
        print("🧹 Cleaning up processes...")
        for name, process in self.processes:
            try:
                print(f"Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"Error stopping {name}: {e}")
    
    def run(self):
        """Run the integrated system"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🎯 Starting Integrated ADK Voice Agent + Behavioral Analyzer System")
        print("=" * 70)
        print("This will start all three systems:")
        print("1. Behavioral Analyzer (API server on port 8083)")
        print("2. ADK Voice Agent (WebSocket on port 8081, HTTP API on port 8082)")
        print("3. Client Web UI (Web interface on port 8080)")
        print("=" * 70)
        print("Press Ctrl+C to stop both systems")
        print()
        
        try:
            # Start behavioral analyzer in a separate thread
            analyzer_thread = threading.Thread(target=self.run_behavioral_analyzer, daemon=True)
            analyzer_thread.start()
            
            # Wait a moment for the analyzer to start
            time.sleep(3)
            
            # Start client web server in a separate thread
            client_thread = threading.Thread(target=self.run_client_web_server, daemon=True)
            client_thread.start()
            
            # Wait a moment for the client server to start
            time.sleep(2)
            
            # Start ADK voice agent in the main thread
            self.run_adk_voice_agent()
            
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.cleanup()

if __name__ == "__main__":
    system = IntegratedSystem()
    system.run()
