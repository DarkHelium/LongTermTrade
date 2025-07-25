#!/usr/bin/env python3
import subprocess
import sys
import os

def start_server():
    """Start uvicorn server"""
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "api:app", 
        "--reload", 
        "--port", "3000",
        "--host", "0.0.0.0"
    ]
    
    print(f"Starting server with command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    start_server()