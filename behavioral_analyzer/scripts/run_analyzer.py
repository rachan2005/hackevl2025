#!/usr/bin/env python3
"""
Convenience script to run the behavioral analyzer.

This script provides an easy way to run the analyzer with common configurations.
"""

import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behavioral_analyzer.main import main

if __name__ == "__main__":
    main()
