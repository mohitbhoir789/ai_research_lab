"""
Pytest configuration file for AI Research Lab tests.
"""
import os
import sys
import pytest

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Configure pytest-asyncio for async tests
pytest_plugins = ["pytest_asyncio"]

# Optionally set the default event loop policy for asyncio tests
@pytest.fixture(scope="session")
def event_loop():
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()