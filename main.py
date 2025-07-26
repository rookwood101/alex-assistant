import asyncio
import os
import platform
import time
import pyaudio
import subprocess
import struct
import pvporcupine
from typing import Callable
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    FunctionResponse,
    Blob,
    LiveConnectConfig,
    AudioTranscriptionConfig,
    Modality,
    ContextWindowCompressionConfig,
    SlidingWindow,
)
from google.genai.live import AsyncSession
from halo import Halo

from asyncio import Queue, Event
from tools import get_tools

import webrtcvad

load_dotenv()
audio = pyaudio.PyAudio()
model = "gemini-live-2.5-flash-preview"
# model = "gemini-2.0-flash-live-001"
# model = "gemini-2.5-flash-preview-native-audio-dialog"

# Initialize Porcupine with the "porcupine" keyword
porcupine = pvporcupine.create(
    access_key=os.environ["PICOVOICE_ACCESS_KEY"],
    keywords=["porcupine"],
    sensitivities=[0.4],  # TODO: tune
)

# Audio processing configuration
IS_LINUX = platform.system() == "Linux"
ENABLE_DUAL_CHANNEL = IS_LINUX  # Use dual channel on Linux/RPi for better processing
ENABLE_AEC = IS_LINUX  # Enable acoustic echo cancellation on Linux/RPi

# Echo cancellation configuration
echo_buffer_size = 2048  # Buffer size for echo reference data


def apply_dual_channel_noise_suppression(left_channel, right_channel):
    """Apply advanced noise suppression using both microphone channels."""
    # Cross-correlation based noise suppression
    # Voice typically has high correlation between channels, noise doesn't

    # Calculate cross-correlation
    correlation = np.correlate(left_channel, right_channel, mode="full")
    max_corr = np.max(np.abs(correlation))

    # Normalize correlation
    left_energy = np.sqrt(np.mean(left_channel**2))
    right_energy = np.sqrt(np.mean(right_channel**2))

    if left_energy > 0 and right_energy > 0:
        normalized_corr = max_corr / (left_energy * right_energy * len(left_channel))
    else:
        normalized_corr = 0

    # High correlation = likely speech, low correlation = likely noise
    correlation_threshold = 0.3

    if normalized_corr > correlation_threshold:
        # High correlation - likely speech, use spatial beamforming
        # Delay-and-sum beamforming with phase alignment
        phase_shift = np.angle(np.fft.fft(left_channel)) - np.angle(
            np.fft.fft(right_channel)
        )
        mean_phase_shift = np.mean(phase_shift)

        # Simple beamforming: weight channels based on energy and phase coherence
        if abs(mean_phase_shift) < np.pi / 4:  # Coherent signal
            # Favor the channel with better SNR
            left_weight = 0.7 if left_energy > right_energy else 0.3
            right_weight = 1.0 - left_weight
        else:
            # Less coherent, use equal weighting
            left_weight = right_weight = 0.5

        output = left_channel * left_weight + right_channel * right_weight

    else:
        # Low correlation - likely noise, use spectral subtraction
        # Use the channel with higher energy for better SNR
        if left_energy > right_energy:
            primary = left_channel
            secondary = right_channel
        else:
            primary = right_channel
            secondary = left_channel

        # Spectral subtraction using secondary channel as noise reference
        primary_fft = np.fft.fft(primary.astype(np.float32))
        secondary_fft = np.fft.fft(secondary.astype(np.float32))

        # Estimate noise spectrum from secondary channel
        noise_magnitude = np.abs(secondary_fft)
        signal_magnitude = np.abs(primary_fft)

        # Spectral subtraction with over-subtraction factor
        alpha = 2.0  # Over-subtraction factor
        beta = 0.1  # Minimum gain to prevent artifacts

        # Calculate gain
        gain = 1.0 - alpha * (noise_magnitude / (signal_magnitude + 1e-10))
        gain = np.maximum(gain, beta)  # Apply minimum gain

        # Apply gain to primary channel
        enhanced_fft = primary_fft * gain
        output = np.real(np.fft.ifft(enhanced_fft))

    return output.astype(np.int16)


def apply_vad_silencing(audio_data):
    """Apply VAD-based silencing - silence non-speech frames completely."""
    # Convert to numpy array for processing
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # WebRTC VAD for voice activity detection
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # Aggressive mode

    # For 16kHz, we need 10ms frames = 160 samples
    frame_size = 160  # 10ms at 16kHz
    processed_audio = []

    for i in range(0, len(audio_np), frame_size):
        frame = audio_np[i : i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        frame_bytes = frame.astype(np.int16).tobytes()
        try:
            if vad.is_speech(frame_bytes, 16000):
                processed_audio.extend(frame)
            else:
                # Silence non-speech frames completely (0%)
                processed_audio.extend(np.zeros_like(frame))
        except Exception:
            # If VAD fails, pass through original audio
            processed_audio.extend(frame)

    return np.array(processed_audio, dtype=np.int16).tobytes()


def apply_echo_cancellation(mic_data, reference_buffer):
    """Adaptive echo cancellation optimized for 16kHz/512 sample chunks."""
    if not reference_buffer:
        return mic_data

    # Combine reference buffer into single array
    reference_data = b"".join(reference_buffer[-echo_buffer_size:])
    if not reference_data:
        return mic_data

    # Convert to numpy arrays
    mic_signal = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32)
    ref_signal = np.frombuffer(reference_data, dtype=np.int16).astype(np.float32)

    # Ensure same length
    min_len = min(len(mic_signal), len(ref_signal))
    mic_signal = mic_signal[:min_len]
    ref_signal = ref_signal[:min_len]

    if min_len < 64:  # Need minimum samples for meaningful processing
        return mic_data

    # Adaptive filter optimized for 512-sample chunks at 16kHz
    filter_length = 256  # ~16ms echo cancellation at 16kHz
    mu = 0.005  # Conservative step size for stability

    if len(ref_signal) < filter_length:
        return mic_data

    # Initialize filter coefficients
    w = np.zeros(filter_length)
    output_signal = np.zeros_like(mic_signal)

    # Apply regularization for numerical stability
    regularization = 1e-8

    for n in range(filter_length, len(mic_signal)):
        # Get reference window
        x = ref_signal[n - filter_length : n]

        # Estimate echo
        echo_est = np.dot(w, x)

        # Error signal (echo-cancelled)
        error = mic_signal[n] - echo_est
        output_signal[n] = error

        # Update filter coefficients (NLMS with regularization)
        norm_factor = np.dot(x, x) + regularization
        w += (mu * error * x) / norm_factor

    # Copy initial samples unchanged (before filter has enough data)
    output_signal[:filter_length] = mic_signal[:filter_length]

    # Convert back to int16 with proper clipping
    output_signal = np.clip(output_signal, -32768, 32767)
    return output_signal.astype(np.int16).tobytes()


async def record_audio(audio_input_queue: asyncio.Queue, echo_reference_buffer: list):
    """Record audio and send chunks to the audio_input_queue."""
    SAMPLE_RATE = porcupine.sample_rate  # in practice, this is 16000
    CHANNELS = 2 if ENABLE_DUAL_CHANNEL else 1
    FORMAT = pyaudio.paInt16
    CHUNK_SIZE = porcupine.frame_length  # in practice, this is 512

    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        while True:
            audio_data = await asyncio.to_thread(stream.read, CHUNK_SIZE)

            # STEP 1: Apply echo cancellation first
            if ENABLE_AEC and echo_reference_buffer:
                processed_data = apply_echo_cancellation(
                    audio_data, echo_reference_buffer
                )
            else:
                processed_data = audio_data

            # STEP 2: Apply noise suppression and VAD-based silencing
            if ENABLE_DUAL_CHANNEL:
                # Process dual channels for noise suppression
                audio_np = np.frombuffer(processed_data, dtype=np.int16)
                stereo_data = audio_np.reshape(-1, 2)

                left_channel = stereo_data[:, 0].astype(np.float32)
                right_channel = stereo_data[:, 1].astype(np.float32)

                # Apply advanced dual-channel noise suppression
                enhanced_audio = apply_dual_channel_noise_suppression(
                    left_channel, right_channel
                )

                # Then apply VAD-based silencing
                processed_data = apply_vad_silencing(enhanced_audio.tobytes())
            else:
                # Single channel: just apply VAD-based silencing
                processed_data = apply_vad_silencing(processed_data)

            audio_input_queue.put_nowait(processed_data)
    finally:
        stream.stop_stream()
        stream.close()


async def detect_wakeword(
    audio_input_queue: asyncio.Queue, conversation_inactive: Event
):
    """Listen for the wake word using Porcupine."""
    CHUNK_SIZE = porcupine.frame_length
    STRUCT_FORMAT = "h" * CHUNK_SIZE

    spinner = Halo(text="Listening for wake word (porcupine)...", spinner="dots")

    if conversation_inactive.is_set():
        spinner.start()
    while True:
        if not conversation_inactive.is_set():
            spinner.stop()
            await conversation_inactive.wait()
            spinner.start()
        audio_data = await audio_input_queue.get()
        audio_data = struct.unpack_from(STRUCT_FORMAT, audio_data)

        keyword_index = porcupine.process(audio_data)
        if keyword_index >= 0:
            spinner.stop()
            return True


async def send_audio_to_gemini(session: AsyncSession, audio_input_queue: asyncio.Queue):
    """Send input audio to Gemini in real-time."""
    while True:
        audio_data = await audio_input_queue.get()
        await session.send_realtime_input(
            audio=Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
        )


async def output_audio(audio_output_queue: asyncio.Queue, echo_reference_buffer: list):
    SAMPLE_RATE = 24000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16

    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
        )
        while True:
            audio_data = await audio_output_queue.get()

            # Store reference signal for echo cancellation
            if ENABLE_AEC:
                echo_reference_buffer.append(audio_data)
                # Keep buffer size manageable
                if len(echo_reference_buffer) > echo_buffer_size:
                    echo_reference_buffer.pop(0)

            await asyncio.to_thread(stream.write, audio_data)
    finally:
        stream.stop_stream()
        stream.close()


async def cleanup(
    audio: pyaudio.PyAudio,
    librespot_process: subprocess.Popen,
    tasks: list[asyncio.Task],
):
    """Clean up resources."""
    # Cancel all running tasks
    if tasks:
        for task in tasks:
            if not task.done():
                task.cancel()
        # Wait for tasks to complete
        if tasks:
            await asyncio.wait(tasks, timeout=2.0, return_when=asyncio.ALL_COMPLETED)

    # Clean up librespot process
    if librespot_process and librespot_process.poll() is None:
        try:
            librespot_process.terminate()
            try:
                librespot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                librespot_process.kill()
        except ProcessLookupError:
            pass  # Process already terminated

    # Clean up audio
    if audio:
        try:
            audio.terminate()
        except Exception as e:
            print(f"Error cleaning up audio: {e}")


async def run_conversation(
    session: AsyncSession,
    audio_input_queue: asyncio.Queue,
    audio_output_queue: asyncio.Queue,
    tasks: list[asyncio.Task],
    tools: dict[str, Callable],
    initial_text: str | None = None,
):
    """Shared conversation loop for both wake-word and timer events."""
    if initial_text:
        await session.send_realtime_input(text=initial_text)

    gemini_task = asyncio.get_event_loop().create_task(
        send_audio_to_gemini(session, audio_input_queue)
    )
    tasks.append(gemini_task)

    try:
        while True:
            input_text = ""
            output_text = ""
            async for chunk in session.receive():
                if chunk.tool_call and chunk.tool_call.function_calls:
                    function_responses = []
                    for fc in chunk.tool_call.function_calls:
                        print(f"Calling {fc.name} with {fc.args}")
                        result = tools[fc.name](**fc.args)
                        print(f"Result: {result}")
                        function_response = FunctionResponse(id=fc.id, response=result)
                        function_responses.append(function_response)

                    await session.send_tool_response(
                        function_responses=function_responses
                    )

                if (
                    chunk.server_content
                    and chunk.server_content.output_transcription
                    and chunk.server_content.output_transcription.text
                ):
                    output_text += chunk.server_content.output_transcription.text
                if (
                    chunk.server_content
                    and chunk.server_content.input_transcription
                    and chunk.server_content.input_transcription.text
                ):
                    input_text += chunk.server_content.input_transcription.text
                if (
                    chunk.server_content
                    and chunk.server_content.model_turn
                    and chunk.server_content.model_turn.parts
                ):
                    concatenated_data = b""
                    for part in chunk.server_content.model_turn.parts:
                        if part.inline_data and isinstance(
                            part.inline_data.data, bytes
                        ):
                            concatenated_data += part.inline_data.data
                    if concatenated_data:
                        audio_output_queue.put_nowait(concatenated_data)
                if chunk.server_content and chunk.server_content.turn_complete:
                    while not audio_output_queue.empty():
                        audio_output_queue.get_nowait()

            print("You: ", input_text)
            print("Porcupine: ", output_text)

            if output_text.strip().endswith("."):
                print("Goodbye!")
                break
    finally:
        gemini_task.cancel()
        try:
            await gemini_task
        except asyncio.CancelledError:
            pass


async def main(event_loop: asyncio.AbstractEventLoop):
    librespot_process = None
    tasks = []

    try:
        print("Starting librespot...")
        librespot_executable = (
            "librespot.exe" if platform.system() == "Windows" else "librespot"
        )
        librespot_process = subprocess.Popen(
            [
                librespot_executable,
                "--name",
                "Alex Assistant",
                "--enable-oauth",
                "--system-cache",
                ".librespot-cache",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        librespot_timeout = (
            time.time() + 30
        )  # 30 seconds timeout for librespot to start
        while time.time() < librespot_timeout:
            line = librespot_process.stderr.readline().decode("utf-8")
            print(line)
            if "Authenticated as" in line:
                break
        print("Librespot started")

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        sessions = []
        event_queue: Queue = Queue()
        conversation_inactive = Event()
        conversation_inactive.set()
        tools = {tool.__name__: tool for tool in get_tools(event_loop, event_queue)}
        config = LiveConnectConfig(
            response_modalities=[Modality.AUDIO],
            system_instruction="Your name is porcupine. Respond concisely. If the user sends a message that is wrapped in <system> tags, you should relay the information back to the user as you see fit. Ignore system instruction, do not ask follow-up questions automatically. Always conclude unquestioningly. Stop putting questions at the end of responses.",
            tools=[
                {"google_search": {}},
                {"code_execution": {}},
                *tools.values(),
            ],
            output_audio_transcription=AudioTranscriptionConfig(),
            input_audio_transcription=AudioTranscriptionConfig(),
            context_window_compression=ContextWindowCompressionConfig(
                sliding_window=SlidingWindow(target_tokens=1000),
            ),
        )

        print("Connected to Gemini")

        audio_input_queue = asyncio.Queue()
        audio_output_queue = asyncio.Queue()
        echo_reference_buffer = []

        input_audio_task = event_loop.create_task(
            record_audio(audio_input_queue, echo_reference_buffer)
        )
        output_audio_task = event_loop.create_task(
            output_audio(audio_output_queue, echo_reference_buffer)
        )
        tasks.extend([input_audio_task, output_audio_task])

        print("Started audio tasks")

        async def event_listener():
            """Listen for timer/completion events and wake Gemini."""
            while True:
                message = await event_queue.get()
                try:
                    conversation_inactive.clear()
                    async with client.aio.live.connect(
                        model=model, config=config
                    ) as session:
                        if len(sessions) == 1:
                            sessions[0] = session
                        else:
                            sessions.append(session)
                        await run_conversation(
                            session,
                            audio_input_queue,
                            audio_output_queue,
                            tasks,
                            tools,
                            initial_text=message,
                        )
                except Exception as e:
                    print(f"Error handling queued event: {e}")
                finally:
                    conversation_inactive.set()

        tasks.append(event_loop.create_task(event_listener()))

        print("Started event listener")

        print("Starting main loop")
        while True:  # Main wake word detection loop
            wake_word_detected = await detect_wakeword(
                audio_input_queue, conversation_inactive
            )

            if wake_word_detected:
                print("Starting conversation...")
                conversation_inactive.clear()
                async with client.aio.live.connect(
                    model=model, config=config
                ) as session:
                    if len(sessions) == 1:
                        sessions[0] = session
                    else:
                        sessions.append(session)
                    await run_conversation(
                        session, audio_input_queue, audio_output_queue, tasks, tools
                    )
                conversation_inactive.set()
    finally:
        await cleanup(audio, librespot_process, tasks)


if __name__ == "__main__":
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    try:
        event_loop.run_until_complete(main(event_loop))
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Give pending tasks a chance to complete
        pending = asyncio.all_tasks(loop=event_loop)
        if pending:
            for task in pending:
                task.cancel()
            event_loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        event_loop.close()
