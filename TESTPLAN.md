## Test Plan: Alex Assistant

This plan defines automated testing for the assistant with Raspberry Pi constraints, microphone input dependencies, and a focus on reproducing second-request failures.

### Goals and constraints
- **Must run on Raspberry Pi OS**; use `uv` for all test execution and dependencies
- **Simulate microphone input** deterministically (no human voice required)
- **Reliably reproduce issues** when making a second request/conversation

### Layer 1 — Unit tests (fast, deterministic)
- [ ] `LEDController.set_vad_active` with `HAS_LEDS=False` → no hardware calls, no exceptions
- [ ] `DebugAudioRecorder.save_recordings`/`clear_recordings` with in-memory frames
- [ ] `apply_dual_channel_noise_suppression` synthetic cases:
  - [ ] High correlation → beamforming path
  - [ ] Low correlation → spectral subtraction path
- [ ] `apply_echo_cancellation` with short mic/reference signals (length checks, non-identity output)
- [ ] `apply_vad_silencing` with monkeypatched `vad_model`/`vad_iterator` return values (silencing only when enabled)
- [ ] `reset_vad_iterator` called at conversation start (verify `reset_states()` on mock)

Commands

```
uv run pytest -q
```

### Layer 2 — Service-level integration tests (async tasks, no hardware)
Run core asyncio tasks with fake audio I/O, fake Porcupine, and a fake Gemini session.

- [x] Replace PyAudio input stream with a fake that returns deterministic 16kHz/512-sample frames, or push bytes into `audio_input_queue`
- [x] Stub `porcupine.process` to return `-1` then `0` to simulate wakeword detection
- [x] Fake `AsyncSession` with:
  - [x] `receive()` yielding scripted chunks: input transcription, small PCM blob, then `turn_complete=True`
  - [x] `send_realtime_input()` capturing audio/text
  - [x] `send_tool_response()` no-op or capture
- [x] Single conversation flow completes without leakage (queues drained, task cancelled)
- [x] Second-request flow (primary target): trigger two wakewords in sequence; assert:
  - [x] `reset_vad_iterator` called per conversation
  - [x] First conversation's `gemini_task` is cancelled and awaited before starting the second
  - [x] `audio_input_queue` not clogged (bounded size)
  - [x] `audio_output_queue` drained on `turn_complete`
  - [ ] `sessions` list stable (no stale references)
  - [x] `echo_reference_buffer` length <= `echo_buffer_size`
- [x] Repeat N cycles (e.g., 10) to catch intermittent second-request failures

Commands

```
uv run pytest -q -k integration --asyncio-mode=auto
```

### Layer 3 — End-to-end (headless) on dev machine (no cloud)
Use a test mode to avoid real hardware/cloud while exercising the full event loop.

- [ ] Env flags:
  - [ ] `ALEX_TEST_MODE=1` → use fake Porcupine and fake `AsyncSession`; disable LEDs if needed
  - [ ] `ALEX_DISABLE_LIBRESPOT=1` → skip spawning librespot in tests
  - [ ] `ALEX_FAKE_AUDIO=path/to/pcm16le_16k_mono.wav` → harness feeds frames into `audio_input_queue`
- [ ] Frame feeder sends 512-sample chunks at 16kHz cadence (`sleep(512/16000)`) with speech/silence patterns
- [ ] Drive two consecutive wakeword→conversation cycles; ensure clean completion
- [ ] Assertions: no task leakage (`asyncio.all_tasks()` stable), queues drained, `echo_reference_buffer` bounded, no exceptions

### Layer 4 — End-to-end on Raspberry Pi (ALSA loopback)
Run near-real with PyAudio and ALSA loopback as the microphone.

- [ ] Enable ALSA loopback
  - [ ] `sudo modprobe snd-aloop`
- [ ] Identify devices
  - [ ] `arecord -l`
  - [ ] `aplay -l`
- [ ] Configure app to capture from loopback device; keep normal speaker output
- [ ] Prepare 16kHz PCM/WAV fixtures including ambient noise, wakeword, and short utterances
- [ ] Feed audio into loopback input
  - [ ] `aplay -D plughw:Loopback,0 test_sequence.wav`
- [ ] Run app with test mode (mock LLM if desired)
- [ ] Execute two cycles with a gap; verify the second request succeeds
- [ ] Use `--debug` to save and compare `debug_unprocessed.wav` and `debug_processed.wav`

### Soak and robustness tests
- [ ] Repeat two-conversation cycle 50–100 times on Pi overnight
- [ ] Inject transient network errors in fake session mid-`receive()`
- [ ] Inject empty audio frames and brief overflows into input to simulate device hiccups
- [ ] Preserve logs and debug WAVs on failure

### Hooks/refactors to enable testing (minimal, incremental)
- [ ] Porcupine wrapper injectable via env (real vs fake)
- [ ] Session factory to provide real or fake `AsyncSession`
- [ ] Audio I/O abstraction to swap PyAudio with in-memory feeders/sinks
- [ ] Env flags: `ALEX_TEST_MODE`, `ALEX_DISABLE_LIBRESPOT`, `ALEX_FAKE_AUDIO`, `ALEX_DISABLE_LEDS`
- [ ] Lightweight metrics/logging:
  - [ ] Start/end of each conversation
  - [ ] Sizes of `audio_input_queue`, `audio_output_queue`
  - [ ] Length of `echo_reference_buffer`
  - [ ] Count of running tasks

### CI on Raspberry Pi
- [ ] Self-hosted GitHub Actions runner on a Raspberry Pi 4+
- [ ] `uv sync --dev` for dependencies
- [ ] Enable `snd-aloop` at job start
- [ ] Run headless end-to-end tests using loopback fixtures
- [ ] Upload artifacts (logs, `debug_*.wav`) on failure

### Acceptance criteria for the “second request” regression
- [ ] After two consecutive conversations, there are no unhandled exceptions
- [ ] First conversation's `gemini_task` is cancelled and awaited
- [ ] `audio_output_queue` is empty after `turn_complete`
- [ ] `audio_input_queue` does not grow unbounded
- [ ] `vad_iterator` is reset between conversations
- [ ] `echo_reference_buffer` length <= `echo_buffer_size`
- [ ] LEDs return to idle/off when conversation ends

### Quick commands reference

```
uv run pytest -q
uv run pytest -q -k integration --asyncio-mode=auto
```

```
sudo modprobe snd-aloop
arecord -l | cat
aplay -l | cat
aplay -D plughw:Loopback,0 test_sequence.wav
```


