#!/usr/bin/env python
"""Launch ARES Streamlit Visualizer with Public Tunnel on Kaggle."""

import os
import subprocess
import time
import sys
from pathlib import Path


def launch():
    print("=" * 65)
    print("  LAUNCHING ARES STREAMLIT INTERACTIVE VISUALIZER")
    print("=" * 65)

    # Ensure streamlit is installed
    subprocess.run(["pip", "install", "-q", "streamlit", "plotly", "pandas"], check=False)

    # Launch Streamlit in background
    app_path = Path(__file__).parent.parent / "app.py"
    streamlit_cmd = [
        "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ]
    
    print(f"\n[1/2] Starting Streamlit server on port 8501...")
    st_proc = subprocess.Popen(streamlit_cmd)
    time.sleep(3)

    # Check external IP for localtunnel password
    try:
        import urllib.request
        ip = urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode("utf8").strip()
        print(f"\n[2/2] Your Kaggle Public Tunnel Password is: \033[1;32m{ip}\033[0m")
    except Exception:
        pass

    print("\nStarting Localtunnel (npx localtunnel --port 8501)...")
    print("Click the generated URL below and enter the IP password above to access the dashboard:\n")
    
    # Run localtunnel
    os.system("npx -y localtunnel --port 8501")


if __name__ == "__main__":
    launch()
