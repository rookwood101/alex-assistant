import os
import platform
import select
import subprocess
import sys
import time
import threading
from pathlib import Path
from dotenv import dotenv_values

import pytest


def _has_alsa_loopback() -> bool:
    try:
        out = subprocess.check_output(["arecord", "-l"], stderr=subprocess.STDOUT, text=True)
        return "Loopback" in out
    except Exception:
        return False


@pytest.mark.skipif(platform.system() != "Linux", reason="Layer 4 runs on Raspberry Pi/Linux only")
@pytest.mark.skipif(not _has_alsa_loopback(), reason="ALSA loopback not found. Run: sudo modprobe snd-aloop")
def test_real_porcupine_real_llm_three_scenarios(tmp_path: Path):
    # Hard watchdog to avoid hangs: terminate app after 120s
    timed_out = {"flag": False}
    app_proc: subprocess.Popen | None = None
    repo_root = Path(__file__).resolve().parents[1]
    fixtures = [
        repo_root / "tests" / "layer4" / "wakeword.wav",
        repo_root / "tests" / "layer4" / "whats_the_capital_of_england.wav",
        repo_root / "tests" / "layer4" / "tell_me_my_local_weather.wav",
    ]
    for f in fixtures:
        assert f.exists(), f"Missing fixture: {f}"

    env = os.environ.copy()
    # Require keys to be present in .env, otherwise skip (not a failure of the test logic)
    config = dotenv_values(str((Path(__file__).resolve().parents[1] / ".env")))
    if not config.get("PICOVOICE_ACCESS_KEY") or not config.get("GEMINI_API_KEY"):
        pytest.skip("PICOVOICE_ACCESS_KEY and/or GEMINI_API_KEY missing in .env; skipping real end-to-end scenario")

    env.update(
        {
            "ALEX_TEST_MODE": "0",
            "ALEX_FAKE_SESSION": "0",
            "ALEX_FAKE_WAKEWORD": "0",
            "ALEX_DISABLE_LIBRESPOT": "1",
            "ALEX_DISABLE_LEDS": "1",
            "ALEX_INPUT_DEVICE_NAME": "Loopback",
            "ALEX_OUTPUT_DEVICE_NAME": "Loopback",
        }
    )

    # Try to resolve an exact input device index for the ALSA Loopback capture
    try:
        import pyaudio  # type: ignore
        pa = pyaudio.PyAudio()
        try:
            loopback_input_index = None
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                name = str(info.get("name", ""))
                if "loopback" in name.lower() and info.get("maxInputChannels", 0) > 0:
                    loopback_input_index = i
                    break
            if loopback_input_index is not None:
                env["ALEX_INPUT_DEVICE_INDEX"] = str(loopback_input_index)
        finally:
            pa.terminate()
    except Exception:
        pass

    # Start watchdog
    def _watchdog_timeout():
        timed_out["flag"] = True
        try:
            if app_proc is not None and app_proc.poll() is None:
                app_proc.terminate()
                # Give a brief moment, then force kill to avoid lingering
                try:
                    app_proc.wait(timeout=3)
                except Exception:
                    app_proc.kill()
        except Exception:
            pass

    watchdog = threading.Timer(120.0, _watchdog_timeout)
    watchdog.daemon = True
    watchdog.start()

    # Start the app
    app_proc = subprocess.Popen(
        [sys.executable, "-u", "main.py", "--debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(repo_root),
    )

    try:
        # Allow app to initialize (PyAudio, VAD, client)
        time.sleep(3.0)

        # Feed wakeword, then question 1, then wakeword, then question 2
        def play(path: Path):
            try:
                subprocess.run(["aplay", "-D", "plughw:Loopback,0", str(path)], check=True, timeout=10)
            except Exception:
                try:
                    subprocess.run(["aplay", str(path)], check=False, timeout=10)
                except Exception:
                    pass

        play(fixtures[0])  # wakeword
        time.sleep(1.2)
        play(fixtures[1])  # question 1
        time.sleep(2.5)
        play(fixtures[0])  # wakeword again
        time.sleep(1.2)
        play(fixtures[2])  # question 2

        # Collect output and look for completions
        starts = 0
        goodbyes = 0
        lines = []
        assert app_proc.stdout is not None
        deadline = time.time() + 30.0
        stdout = app_proc.stdout
        assert stdout is not None
        while time.time() < deadline and not timed_out["flag"]:
            ready, _, _ = select.select([stdout], [], [], 0.2)
            if not ready:
                continue
            line = stdout.readline()
            if not line:
                continue
            lines.append(line)
            if "Starting conversation..." in line:
                starts += 1
            if "Goodbye!" in line:
                goodbyes += 1
            if starts >= 2 and goodbyes >= 2:
                break

        if timed_out["flag"]:
            pytest.fail("Layer 4 scenario test timed out after 120 seconds")

        assert starts >= 2, f"Expected >=2 conversation starts, got {starts}. Output: {''.join(lines)}"
        assert goodbyes >= 2, f"Expected >=2 conversation completions, got {goodbyes}. Output: {''.join(lines)}"
    finally:
        try:
            if app_proc is not None and app_proc.poll() is None:
                app_proc.terminate()
                app_proc.wait(timeout=5)
        except Exception:
            if app_proc is not None:
                app_proc.kill()
        finally:
            try:
                watchdog.cancel()
            except Exception:
                pass


