import os
import platform
import subprocess
import sys
import time
import wave
import contextlib
from pathlib import Path

import pytest


def _has_alsa_loopback() -> bool:
    try:
        out = subprocess.check_output(["arecord", "-l"], stderr=subprocess.STDOUT, text=True)
        return "Loopback" in out
    except Exception:
        return False


@pytest.mark.skipif(platform.system() != "Linux", reason="Layer 4 runs on Raspberry Pi/Linux only")
@pytest.mark.skipif(not _has_alsa_loopback(), reason="ALSA loopback not found. Run: sudo modprobe snd-aloop")
def test_layer4_two_cycles_with_alsa_loopback(tmp_path: Path):
    # Prepare a short 16kHz stereo WAV to feed via aplay to the loopback capture
    wav_path = tmp_path / "test_sequence.wav"
    sample_rate = 16000
    duration_seconds = 2.0
    num_frames = int(sample_rate * duration_seconds)

    with contextlib.closing(wave.open(str(wav_path), "wb")) as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write silence frames
        wf.writeframes(b"\x00\x00" * 2 * num_frames)

    # Environment for the app process
    env = os.environ.copy()
    # Default: Fake session and fake wakeword. Can be overridden by environment when invoking pytest
    env.setdefault("ALEX_TEST_MODE", "1")
    env.setdefault("ALEX_FAKE_SESSION", "1")
    env.setdefault("ALEX_FAKE_WAKEWORD", "1")
    env.setdefault("ALEX_DISABLE_LIBRESPOT", "1")
    env.setdefault("ALEX_DISABLE_LEDS", "1")
    # Ensure PyAudio input uses the ALSA loopback capture
    env.setdefault("ALEX_INPUT_DEVICE_NAME", "Loopback")

    # Start the app with debug enabled to emit debug_*.wav
    app_proc = subprocess.Popen(
        [sys.executable, "-u", "main.py", "--debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    # Helper to feed loopback with aplay
    def play_once():
        try:
            subprocess.run(["aplay", "-D", "plughw:Loopback,0", str(wav_path)], check=True)
        except Exception:
            # Fallback to default device; test still exercises capture path
            subprocess.run(["aplay", str(wav_path)])

    # Allow app to initialize PyAudio
    time.sleep(1.0)

    # Feed audio into loopback twice with a gap (to simulate two wakeword->conversation cycles)
    play_once()
    time.sleep(1.0)
    play_once()

    # Collect output for a short period and look for two cycles
    start = time.time()
    deadline = start + 12.0
    starts = 0
    goodbyes = 0
    lines = []
    assert app_proc.stdout is not None
    while time.time() < deadline:
        line = app_proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        lines.append(line)
        if "Starting conversation..." in line:
            starts += 1
        if "Goodbye!" in line:
            goodbyes += 1
        if starts >= 2 and goodbyes >= 2:
            break

    # Stop the app process
    try:
        app_proc.terminate()
        app_proc.wait(timeout=5)
    except Exception:
        app_proc.kill()

    # Basic assertions: we observed two cycles
    assert starts >= 2, f"Expected >=2 conversation starts, got {starts}. Output: {''.join(lines)}"
    assert goodbyes >= 2, f"Expected >=2 conversation completions, got {goodbyes}. Output: {''.join(lines)}"

    # Debug WAVs should have been written
    repo_root = Path(__file__).resolve().parents[1]
    unprocessed = repo_root / "debug_unprocessed.wav"
    processed = repo_root / "debug_processed.wav"
    assert unprocessed.exists(), "debug_unprocessed.wav not found"
    assert processed.exists(), "debug_processed.wav not found"
    assert unprocessed.stat().st_size > 0, "debug_unprocessed.wav is empty"
    assert processed.stat().st_size > 0, "debug_processed.wav is empty"


