#!/usr/bin/env python
"""
Setup script for URL Phishing Detection System
This script helps users set up the project quickly
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print("✓ Python version check passed")

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['models', 'plots', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("Error: Failed to install required packages")
        sys.exit(1)

def check_env_file():
    """Check if .env file exists, create template if it doesn't"""
    if not os.path.exists('.env'):
        print("Creating .env file template...")
        with open('.env', 'w') as f:
            f.write("""# OpenRouter API Key
OPENROUTER_API_KEY=your-api-key-here

# Optional: Site information for OpenRouter analytics
SITE_URL=http://localhost:8501
SITE_NAME=URL Phishing Detector

# Other API keys (uncomment if needed)
# OPENAI_API_KEY=your-openai-api-key
# ANTHROPIC_API_KEY=your-anthropic-api-key
# META_API_KEY=your-meta-api-key
""")
        print("✓ Created .env file template")
        print("⚠️ Please edit the .env file to add your API keys")
    else:
        print("✓ .env file already exists")

def main():
    """Main setup function"""
    print("Setting up URL Phishing Detection System...")
    
    check_python_version()
    create_directories()
    install_requirements()
    check_env_file()
    
    print("\nSetup complete! You can now:")
    print("1. Train the model:   python model_training.py")
    print("2. Run the app:       streamlit run app.py")
    print("\nMake sure to add your API keys to the .env file before running the app.")

if __name__ == "__main__":
    main() 