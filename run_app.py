"""
Simple launcher script for the Financial Report Summarization UI.
Run this file to start the Gradio interface.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the UI
from ui.app import main

if __name__ == "__main__":
    main()
