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

# Initialize Porcupine with the "bumblebee" keyword
porcupine = pvporcupine.create(
    access_key=os.environ["PICOVOICE_ACCESS_KEY"],
    keywords=["porcupine"]
)


async def record_and_send_audio_to_gemini(session: AsyncSession):
    """Record audio and send it to Gemini in real-time."""
    SAMPLE_RATE = 16000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    CHUNK_SIZE = 1024

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
            await session.send_realtime_input(
                audio=Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
            )
    finally:
        stream.stop_stream()
        stream.close()


async def play_gemini_audio(audio_output_queue: asyncio.Queue):
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
        


async def main(event_loop: asyncio.AbstractEventLoop):
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
        )
    )

    while True:  # Main wake word detection loop
        print("Listening for wake word (porcupine)...")
        wake_word_detected = await asyncio.to_thread(detect_wakeword)
        
        if wake_word_detected:
            print("Starting conversation...")
            async with client.aio.live.connect(model=model, config=config) as session:
                if len(sessions) == 1:
                    sessions[0] = session
                else:
                    sessions.append(session)
                print("Porcupine is now listening to your microphone...")
                print()
                audio_output_queue = asyncio.Queue()
                record_task = event_loop.create_task(record_and_send_audio_to_gemini(session))
                play_task = event_loop.create_task(play_gemini_audio(audio_output_queue))
                
                while True:
                    input_text = ""
                    output_text = ""
                    async for chunk in session.receive():
                        if chunk.tool_call and chunk.tool_call.function_calls:
                            function_responses = []
                            for fc in chunk.tool_call.function_calls:
                                if fc.name and fc.args:
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
                        if chunk.data is not None:
                            audio_output_queue.put_nowait(chunk.data)
                        if chunk.server_content and chunk.server_content.turn_complete:
                            while not audio_output_queue.empty():
                                audio_output_queue.get_nowait()

                    print("You: ", input_text)
                    print("Porcupine: ", output_text)

                    if output_text.strip().endswith("."):
                        print("Goodbye!")
                        record_task.cancel()
                        play_task.cancel()
                        try:
                            await record_task
                            await play_task
                        except asyncio.CancelledError:
                            pass
                        break  # Return to wake word detection


def detect_wakeword():
    """Listen for the wake word using Porcupine."""
    CHANNELS = 1
    SAMPLE_RATE = porcupine.sample_rate
    FORMAT = pyaudio.paInt16
    CHUNK_SIZE = porcupine.frame_length
    stream = None

    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        while True:
            pcm = stream.read(CHUNK_SIZE)
            pcm = struct.unpack_from("h" * CHUNK_SIZE, pcm)
            
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
                return True

    except Exception as e:
        print(f"Error in wake word detection: {e}")
        return False
    finally:
        if stream:
            stream.stop_stream()
            stream.close()

if __name__ == "__main__":
    event_loop = asyncio.new_event_loop()
    event_loop.run_until_complete(main(event_loop))
