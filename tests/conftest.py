import os
import sys
import asyncio
from pathlib import Path

# Ensure test mode before importing application module
os.environ.setdefault("ALEX_TEST_MODE", "1")

# Ensure repository root is on sys.path so `import main` works when pytest changes CWD
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import main  # noqa: E402

import pytest


@pytest.fixture
def app_module():
    return main


@pytest.fixture
def chunk_size(app_module):
    return app_module.porcupine.frame_length


@pytest.fixture
def make_silence_frame(chunk_size):
    def _make():
        # int16 zeros of length chunk_size
        return b"\x00\x00" * chunk_size
    return _make


@pytest.fixture
def event_loop():
    # Create a fresh event loop per test for isolation
    loop = asyncio.new_event_loop()
    yield loop
    loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop=loop), return_exceptions=True))
    loop.close()


