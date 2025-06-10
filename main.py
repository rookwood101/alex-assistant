import asyncio
import os
import time
import pyaudio
import subprocess
import struct
import pvporcupine

from dotenv import load_dotenv
from google import genai
from google.genai.types import FunctionResponse, Blob, LiveConnectConfig, AudioTranscriptionConfig, Modality, ContextWindowCompressionConfig, SlidingWindow
from google.genai.live import AsyncSession

from tools import get_tools

load_dotenv()
audio = pyaudio.PyAudio()
model = "gemini-2.0-flash-live-001"
# model = "gemini-2.5-flash-preview-native-audio-dialog"

# Initialize Porcupine with the "bumblebee" keyword
porcupine = pvporcupine.create(
    access_key=os.environ["PICOVOICE_ACCESS_KEY"],
    keywords=["porcupine"],
    sensitivities=[0.7], # TODO: tune
)


async def record_audio(audio_input_queue: asyncio.Queue):
    """Record audio and send chunks to the audio_input_queue."""
    SAMPLE_RATE = porcupine.sample_rate
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    CHUNK_SIZE = porcupine.frame_length

    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        while True:
            audio_data = await asyncio.to_thread(stream.read, CHUNK_SIZE)
            audio_input_queue.put_nowait(audio_data)
    finally:
        stream.stop_stream()
        stream.close()


async def detect_wakeword(audio_input_queue: asyncio.Queue):
    """Listen for the wake word using Porcupine."""
    CHUNK_SIZE = porcupine.frame_length
    STRUCT_FORMAT = "h" * CHUNK_SIZE

    while True:
        audio_data = await audio_input_queue.get()
        audio_data = struct.unpack_from(STRUCT_FORMAT, audio_data)
        
        keyword_index = porcupine.process(audio_data)
        if keyword_index >= 0:
            print("Wake word detected!")
            return True


async def send_audio_to_gemini(session: AsyncSession, audio_input_queue: asyncio.Queue):
    """Send input audio to Gemini in real-time."""
    while True:
        audio_data = await audio_input_queue.get()
        await session.send_realtime_input(
            audio=Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
        )


async def output_audio(audio_output_queue: asyncio.Queue):
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
            await asyncio.to_thread(stream.write, audio_data)
    finally:
        stream.stop_stream()
        stream.close()


async def cleanup(audio: pyaudio.PyAudio, librespot_process: subprocess.Popen, tasks: list[asyncio.Task]):
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


async def main(event_loop: asyncio.AbstractEventLoop):
    librespot_process = None
    tasks = []
    
    try:
        print("Starting librespot...")
        librespot_process = subprocess.Popen(
            ["librespot.exe", "--name", "Alex Assistant", "--enable-oauth", "--system-cache", ".librespot-cache"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        librespot_timeout = time.time() + 30 # 30 seconds timeout for librespot to start
        while time.time() < librespot_timeout:
                line = librespot_process.stderr.readline().decode('utf-8')
                if "Authenticated as" in line:
                    break

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        sessions = []
        tools = { tool.__name__: tool for tool in get_tools(event_loop, sessions) }
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

        audio_input_queue = asyncio.Queue()
        audio_output_queue = asyncio.Queue()
        input_audio_task = event_loop.create_task(record_audio(audio_input_queue))
        output_audio_task = event_loop.create_task(output_audio(audio_output_queue))
        tasks.extend([input_audio_task, output_audio_task])

        while True:  # Main wake word detection loop
            print("Listening for wake word (porcupine)...")
            wake_word_detected = await detect_wakeword(audio_input_queue)
            
            if wake_word_detected:
                print("Starting conversation...")
                async with client.aio.live.connect(model=model, config=config) as session:
                    if len(sessions) == 1:
                        sessions[0] = session
                    else:
                        sessions.append(session)
                    print("Porcupine is now listening to your microphone...")
                    print()
                    gemini_task = event_loop.create_task(send_audio_to_gemini(session, audio_input_queue))
                    tasks.append(gemini_task)
                    
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
                                    function_response = FunctionResponse(
                                        id=fc.id,
                                        response=result
                                    )
                                    function_responses.append(function_response)

                                await session.send_tool_response(function_responses=function_responses)
                            if chunk.server_content and chunk.server_content.output_transcription and chunk.server_content.output_transcription.text:
                                output_text = output_text + chunk.server_content.output_transcription.text
                            if chunk.server_content and chunk.server_content.input_transcription and chunk.server_content.input_transcription.text:
                                input_text = input_text + chunk.server_content.input_transcription.text
                            if chunk.server_content and chunk.server_content.model_turn and chunk.server_content.model_turn.parts:
                                concatenated_data = b''
                                for part in chunk.server_content.model_turn.parts:
                                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                                        concatenated_data += part.inline_data.data
                                if len(concatenated_data) > 0:
                                    audio_output_queue.put_nowait(concatenated_data)
                            if chunk.server_content and chunk.server_content.turn_complete:
                                while not audio_output_queue.empty():
                                    audio_output_queue.get_nowait()

                        print("You: ", input_text)
                        print("Porcupine: ", output_text)

                        if output_text.strip().endswith("."):
                            print("Goodbye!")
                            gemini_task.cancel()
                            try:
                                await gemini_task
                            except asyncio.CancelledError:
                                pass
                            break  # Return to wake word detection
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
            event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        event_loop.close()
