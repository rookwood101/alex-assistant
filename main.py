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
import argparse
import wave

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
import torch

# LED support for Raspberry Pi
try:
    import apa102
    HAS_LEDS = True
except ImportError:
    HAS_LEDS = False


class DebugAudioRecorder:
    """Records debug audio to WAV files when debug mode is enabled"""
    
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.unprocessed_frames = []
        self.processed_frames = []
        self.sample_rate = 16000
        
    def record_unprocessed(self, audio_data: bytes):
        """Record unprocessed audio data"""
        if self.enabled:
            self.unprocessed_frames.append(audio_data)
    
    def record_processed(self, audio_data: bytes):
        """Record processed audio data"""
        if self.enabled:
            self.processed_frames.append(audio_data)
    
    def save_recordings(self):
        """Save recorded audio to disk"""
        if not self.enabled:
            return
            
        # Save unprocessed audio
        if self.unprocessed_frames:
            with wave.open('debug_unprocessed.wav', 'wb') as wf:
                wf.setnchannels(2 if ENABLE_DUAL_CHANNEL else 1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.unprocessed_frames))
            print(f"Saved {len(self.unprocessed_frames)} unprocessed audio frames to debug_unprocessed.wav")
        
        # Save processed audio
        if self.processed_frames:
            with wave.open('debug_processed.wav', 'wb') as wf:
                wf.setnchannels(1)  # Processed audio is always mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.processed_frames))
            print(f"Saved {len(self.processed_frames)} processed audio frames to debug_processed.wav")
    
    def clear_recordings(self):
        """Clear recorded frames to free memory"""
        if self.enabled:
            self.unprocessed_frames.clear()
            self.processed_frames.clear()


class LEDController:
    """Elegant LED controller for VAD feedback"""
    
    def __init__(self, max_brightness=128):
        self.enabled = HAS_LEDS
        self.max_brightness = max_brightness
        if self.enabled:
            self.dev = apa102.APA102(num_led=3)
            self.off()
    
    def set_vad_active(self, is_active: bool, confidence: float = 1.0):
        """Set LED state based on VAD activity and confidence level"""
        if not self.enabled:
            return
            
        if is_active:
            if confidence >= 0.7:
                # High confidence: Full purple pulse
                intensity = int(self.max_brightness + (self.max_brightness - 1) * np.sin(time.time() * 8))
                for i in range(3):
                    self.dev.set_pixel(i, intensity, 0, intensity)
            else:
                # Medium confidence: Dimmer purple (static, no pulse)
                dim_intensity = int(self.max_brightness * 0.5)  # 40% brightness for medium confidence
                for i in range(3):
                    self.dev.set_pixel(i, dim_intensity, 0, dim_intensity)
        else:
            # Dim blue when listening but no speech
            dim_intensity = int(self.max_brightness * 0.1)  # Scale down dim blue relative to max brightness
            for i in range(3):
                self.dev.set_pixel(i, 0, 0, dim_intensity)
        
        self.dev.show()
    
    def off(self):
        """Turn off all LEDs"""
        if not self.enabled:
            return
        for i in range(3):
            self.dev.set_pixel(i, 0, 0, 0)
        self.dev.show()


load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Alex Assistant')
parser.add_argument('--debug', action='store_true', help='Enable debug audio recording')
parser.add_argument('--enable-vad-silencing', action='store_true', help='Enable VAD-based audio silencing')
args = parser.parse_args()

audio = pyaudio.PyAudio()
model = "gemini-live-2.5-flash-preview"
# model = "gemini-2.0-flash-live-001"
# model = "gemini-2.5-flash-preview-native-audio-dialog"

# Initialize LED controller
led_controller = LEDController(max_brightness=10)

# Initialize debug audio recorder
debug_recorder = DebugAudioRecorder(enabled=args.debug)

# Initialize Porcupine with the "porcupine" keyword
porcupine = pvporcupine.create(
    access_key=os.environ["PICOVOICE_ACCESS_KEY"],
    keywords=["porcupine"],
    sensitivities=[0.3],  # TODO: tune
)

# Audio processing configuration
IS_LINUX = platform.system() == "Linux"
ENABLE_DUAL_CHANNEL = IS_LINUX  # Use dual channel on Linux/RPi for better processing
ENABLE_AEC = IS_LINUX  # Enable acoustic echo cancellation on Linux/RPi

# Echo cancellation configuration
echo_buffer_size = 2048  # Buffer size for echo reference data

def run_vad_inference(audio_float):
    """Run VAD inference on audio data using VADIterator properly."""
    audio_tensor = torch.from_numpy(audio_float)
    
    # VADIterator maintains state between calls and returns speech segments
    # when they are detected (handles thresholding internally at 0.4)
    speech_dict = vad_iterator(audio_tensor, return_seconds=True)
    
    if speech_dict:
        # Speech segment detected - return high confidence
        print(f"Speech segment detected: {speech_dict}")
        return 0.8  # High confidence when VADIterator detects speech
    else:
        # No complete speech segment in this chunk
        # Get raw model confidence for partial segments
        raw_confidence = vad_model(audio_tensor, 16000).item()
        return raw_confidence


# Silero VAD model and utils
try:
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False,
                                          onnx=False)
    get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = vad_utils
    vad_iterator = VADIterator(vad_model, threshold=0.1, sampling_rate=16000, 
                               min_silence_duration_ms=100, speech_pad_ms=200)
except Exception as e:
    print(f"Warning: Failed to initialize Silero VAD: {e}")
    vad_model = None
    vad_iterator = None

def reset_vad_iterator():
    """Reset VAD iterator state (call at start of new conversation)"""
    global vad_iterator
    if vad_iterator is not None:
        vad_iterator.reset_states()  # Use built-in reset method

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
        primary_fft = np.fft.fft(primary)
        secondary_fft = np.fft.fft(secondary)

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


async def apply_vad_silencing(audio_np):
    """Apply Silero VAD-based LED feedback and optional audio silencing.
    
    Args:
        audio_np: numpy array of int16 audio data
    
    Returns:
        numpy array of int16 processed audio data
    """
    
    if vad_model is None:
        # Fallback: pass through original audio if VAD not available
        led_controller.set_vad_active(True, 1.0)
        return audio_np

    # Convert to float32 for processing
    audio_float = audio_np.astype(np.float32) / 32768.0
    
    try:
        # Always run VAD for LED feedback
        speech_prob = await asyncio.to_thread(run_vad_inference, audio_float)
        
        # Threshold for speech detection
        speech_threshold = 0.1
        
        if speech_prob > speech_threshold:
            # Speech detected
            speech_detected = True
            confidence = min(speech_prob, 1.0)  # Use actual probability as confidence
            
            # Only apply volume scaling if VAD silencing is enabled
            if args.enable_vad_silencing:
                volume_factor = 1.0
            else:
                volume_factor = 1.0  # No silencing, keep original volume
        else:
            # No speech detected
            speech_detected = False
            confidence = speech_prob
            
            # Only apply volume scaling if VAD silencing is enabled
            if args.enable_vad_silencing:
                volume_factor = 0.1  # 10% volume for non-speech
            else:
                volume_factor = 1.0  # No silencing, keep original volume
        
        # Apply volume scaling to the current chunk
        processed_audio = (audio_np * volume_factor).astype(np.int16)
        
    except Exception as e:
        print(f"Silero VAD error: {e}")
        # Fallback: pass through original audio
        processed_audio = audio_np
        speech_detected = True
        confidence = 1.0
    
    # Always update LED based on speech detection and confidence
    led_controller.set_vad_active(speech_detected, confidence)
    
    return processed_audio


def apply_echo_cancellation(mic_data, reference_buffer):
    """Adaptive echo cancellation optimized for 16kHz/512 sample chunks."""
    if not reference_buffer:
        return mic_data

    # Combine reference buffer into single array
    reference_data = b"".join(reference_buffer[-echo_buffer_size:])
    if not reference_data:
        return mic_data

    # Convert to numpy arrays as float32 for processing
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
            audio_data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)

            # Record unprocessed audio for debug
            debug_recorder.record_unprocessed(audio_data)

            # STEP 1: Apply echo cancellation first
            if ENABLE_AEC and echo_reference_buffer:
                processed_data = apply_echo_cancellation(
                    audio_data, echo_reference_buffer
                )
            else:
                processed_data = audio_data

            # STEP 2: Apply noise suppression and VAD-based silencing
            if ENABLE_DUAL_CHANNEL:
                # Convert to numpy array once
                audio_np = np.frombuffer(processed_data, dtype=np.int16)
                stereo_data = audio_np.reshape(-1, 2)

                left_channel = stereo_data[:, 0].astype(np.float32)
                right_channel = stereo_data[:, 1].astype(np.float32)

                # Apply advanced dual-channel noise suppression
                enhanced_audio = apply_dual_channel_noise_suppression(
                    left_channel, right_channel
                )

                # Then apply VAD-based silencing (pass numpy array directly)
                final_audio = await apply_vad_silencing(enhanced_audio)
                processed_data = final_audio.tobytes()
            else:
                # Single channel: convert once and apply VAD-based silencing
                audio_np = np.frombuffer(processed_data, dtype=np.int16)
                final_audio = await apply_vad_silencing(audio_np)
                processed_data = final_audio.tobytes()

            # Record processed audio for debug
            debug_recorder.record_processed(processed_data)

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

            await asyncio.to_thread(stream.write, audio_data, exception_on_underflow=False)
    finally:
        stream.stop_stream()
        stream.close()


async def cleanup(
    audio: pyaudio.PyAudio,
    librespot_process: subprocess.Popen,
    tasks: list[asyncio.Task],
):
    """Clean up resources."""
    # Save debug recordings if enabled
    debug_recorder.save_recordings()
    
    # Turn off LEDs
    led_controller.off()
    
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
    # Reset VAD iterator state for new conversation
    reset_vad_iterator()
    
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
