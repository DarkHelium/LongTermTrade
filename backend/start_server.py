#!/usr/bin/env python3
import subprocess
import sys
import os

def start_server():
    """Start uvicorn server with proper exclusions for virtual environment"""
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "api:app", 
        "--reload", 
        "--port", "3000",
        "--reload-include", "*.py",
        "--reload-exclude", ".myenv/**",
        "--reload-exclude", "__pycache__/**",
        "--reload-exclude", "*.pyc",
        "--reload-exclude", "Fin-R1/**"
    ]
    
    print(f"Starting server with command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    start_server()