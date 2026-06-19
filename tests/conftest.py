"""Make the `app` package importable from the test suite.

predictor.py lives in app/, which isn't on sys.path when pytest runs from the
project root, so we add it explicitly.
"""

import os
import sys

APP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
