"""
Simple launcher script for the driver drowsiness detection system
Run this from the project root directory
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Change to src directory for imports
original_dir = os.getcwd()
os.chdir(src_path)

from main import main

if __name__ == '__main__':
    try:
        main()
    finally:
        os.chdir(original_dir)

