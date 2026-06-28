"""
conftest.py — Pytest configuration for AI Resume Screener
Internship Project

This file allows pytest to import modules from the project root
without needing sys.path.insert() hacks in every test file.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))