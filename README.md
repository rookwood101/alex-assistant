An AI assistant like Alexa that runs on Raspberry Pi with Seeed Studio 2 Mic HAT and Google Gemini Live.

## Prereqs
- Install `uv` from `astral.sh` on both dev and the Raspberry Pi
- Raspberry Pi OS (Bookworm recommended). Python 3.11+ (project uses uv to manage env)

## Quick start (developer machine)
```bash
uv sync --dev
uv run pytest -q
```

## Raspberry Pi setup
```bash
ssh pi@raspberrypi.local
cd /home/pi/alex-assistant
uv sync --dev
```

### Enable ALSA loopback (for Layer 4 tests)
```bash
sudo apt-get update
sudo apt-get install -y alsa-utils
sudo modprobe snd-aloop
# Persist across reboots
echo snd-aloop | sudo tee -a /etc/modules
# Verify
arecord -l | cat
aplay -l | cat
```

You should see a device like `card X: Loopback` in the listings.

### Provide test audio fixtures
Place these WAV files (16kHz recommended) in:
- `tests/layer4/wakeword.wav` (contains the word "porcupine")
- `tests/layer4/whats_the_capital_of_england.wav`
- `tests/layer4/tell_me_my_local_weather.wav`

Tip: mono is fine; stereo is also fine. The system captures at 16kHz.

## Running the app (headful)
```bash
uv run python main.py --debug
```

Environment flags:
- `ALEX_DISABLE_LEDS=1` to disable LEDs
- `ALEX_DISABLE_LIBRESPOT=1` to skip librespot
- `ALEX_INPUT_DEVICE_NAME=Loopback` to capture from loopback
- `ALEX_OUTPUT_DEVICE_NAME=...` to select the speaker output device (optional)

## Layer 4 tests (on Raspberry Pi with real Porcupine and real LLM)

We support two modes:
- Self-contained (fake LLM + fake wakeword) for fast validation
- Real Porcupine + real Gemini for full end-to-end

### 1) Self-contained mode (sanity check)
```bash
ssh pi@raspberrypi.local 'cd /home/pi/alex-assistant && \
  ALEX_TEST_MODE=1 ALEX_FAKE_SESSION=1 ALEX_FAKE_WAKEWORD=1 ALEX_DISABLE_LEDS=1 ALEX_DISABLE_LIBRESPOT=1 \
  uv run pytest -q tests/test_layer4_pi.py -s -vv'
```

### 2) Real Porcupine + real Gemini
Requirements:
- `PICOVOICE_ACCESS_KEY` exported in the environment
- `GEMINI_API_KEY` exported in the environment
- Loopback enabled and visible
- WAV fixtures present under `tests/layer4/`

Run:
```bash
ssh pi@raspberrypi.local 'cd /home/pi/alex-assistant && \
  export PICOVOICE_ACCESS_KEY=... && export GEMINI_API_KEY=... && \
  ALEX_TEST_MODE=0 ALEX_FAKE_SESSION=0 ALEX_FAKE_WAKEWORD=0 ALEX_DISABLE_LEDS=1 ALEX_DISABLE_LIBRESPOT=1 ALEX_INPUT_DEVICE_NAME=Loopback \
  uv run pytest -q tests/test_layer4_scenarios.py -s -vv'
```

The tests will:
- Feed `wakeword.wav` followed by each prompt WAV via `aplay -D plughw:Loopback,0`
- Observe wakeword detection and capture LLM audio responses
- Ensure two consecutive conversations complete without leakage

Artifacts:
- `debug_unprocessed.wav` and `debug_processed.wav` will be saved in repo root when `--debug` is used.

## Troubleshooting
- If `arecord -l` does not list Loopback, re-run `sudo modprobe snd-aloop` and ensure `/etc/modules` contains `snd-aloop`.
- For device selection, set `ALEX_INPUT_DEVICE_INDEX` or `ALEX_OUTPUT_DEVICE_INDEX` if name matching is unreliable.
- If VLC/Spotify errors appear in tests, set `ALEX_TEST_MODE=1` or `ALEX_DISABLE_SPOTIFY=1`.