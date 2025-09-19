# Hugging Face Spaces entry point
# This file is required for Hugging Face Spaces deployment

import subprocess
import sys

# Install requirements if needed
try:
    import streamlit
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_demo.txt"])

# Import and run the demo app
from demo_app import main

if __name__ == "__main__":
    main()