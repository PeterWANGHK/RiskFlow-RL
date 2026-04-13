"""
DRIFT Configuration
===================
Re-exports configuration from root level for package organization.
"""

import sys
import os

# Add parent directory to path to import root-level config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import and re-export from root config
from config import Config

__all__ = ['Config']
