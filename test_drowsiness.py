"""
Test script for drowsiness detection with head pose disabled
Run this to test drowsiness detection without head pose interference
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
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Drowsiness Detection (Head Pose Disabled)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--ear-threshold', type=float, default=0.20, 
                       help='EAR threshold (default: 0.20, lower = more sensitive)')
    
    args = parser.parse_args()
    
    # Override to disable head pose and use lower EAR threshold
    sys.argv = ['test_drowsiness.py', '--disable-head-pose', '--ear-threshold', str(args.ear_threshold), '--camera', str(args.camera)]
    
    try:
        main()
    finally:
        os.chdir(original_dir)

