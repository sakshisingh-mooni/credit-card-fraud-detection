"""
tests/conftest.py
-----------------
Pytest configuration: sets MODEL_DIR to test fixtures before app.py is
imported. app.py loads model artifacts at import time using the MODEL_DIR
env var — without this, tests fail on a fresh clone with no model/ folder.

Reference: https://flask.palletsprojects.com/en/3.1.x/testing/
"""
import os

# Must be set BEFORE app.py is imported — pytest loads conftest.py first.
# Points to minimal valid fixture artifacts in tests/fixtures/.
# Production model in model/ is never loaded during tests.
os.environ.setdefault("MODEL_DIR", os.path.join(os.path.dirname(__file__), "fixtures"))
