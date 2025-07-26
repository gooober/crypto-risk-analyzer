#!/usr/bin/env python
"""
Auto-installer and runner for Crypto Risk Analyzer
Just run: python run_crypto_analyzer.py
"""

import subprocess
import sys
import os

def install_and_run():
    """Install dependencies and run the app"""
    
    print("🚀 Crypto Risk Analyzer - Auto Setup")
    print("=" * 40)
    
    # List of required packages
    required_packages = [
        'streamlit',
        'requests', 
        'pandas',
        'numpy',
        'plotly'
    ]
    
    # Install packages
    print("📦 Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except:
            print(f"❌ Failed to install {package}")
            return False
    
    print("\n✅ All packages installed!")
    print("🎉 Launching Crypto Risk Analyzer...\n")
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found in current directory!")
        print("📝 Please save the main application code as 'app.py' first.")
        return False
    
    # Run streamlit
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    return True

if __name__ == "__main__":
    install_and_run()
