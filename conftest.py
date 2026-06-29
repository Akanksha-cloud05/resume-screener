"""
conftest.py — Pytest configuration for AI Resume Screener
Internship Project

This file allows pytest to import modules from the project root
without needing sys.path.insert() hacks in every test file.
"""

import sys
import os
import logging

# Add project root to path so tests can import engine, config, and analytics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure pytest logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)