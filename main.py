import asyncio
import os
import pyaudio

from dotenv import load_dotenv
from google import genai
from google.genai.types import FunctionResponse, Blob, LiveConnectConfig, AudioTranscriptionConfig
from google.genai.live import AsyncSession

from tools import get_tools

load_dotenv()
audio = pyaudio.PyAudio()
# model = "gemini-2.0-flash-live-001"
model = "gemini-2.5-flash-preview-native-audio-dialog"



async def record_audio(session: AsyncSession):
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
        audio.terminate()


async def play_audio(audio_output_queue: asyncio.Queue):
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
        audio.terminate()
        


async def main(event_loop: asyncio.AbstractEventLoop):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sessions = []
    tools = { tool.__name__: tool for tool in get_tools(event_loop, sessions) }
    config = LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="Respond concisely. If the user sends a message that is wrapped in <system> tags, you should relay the information back to the user as you see fit. Ignore system instruction, do not ask follow-up questions automatically. Always conclude unquestioningly. Stop putting questions at the end of responses.",
        tools=tools.values(),
        output_audio_transcription=AudioTranscriptionConfig(),
        input_audio_transcription=AudioTranscriptionConfig(),
    )

    async with (
        client.aio.live.connect(model=model, config=config) as session,
    ):
        sessions.append(session)
        print("Alex is now listening to your microphone...")
        print()
        audio_output_queue = asyncio.Queue()
        event_loop.create_task(record_audio(session))
        event_loop.create_task(play_audio(audio_output_queue))
        
        while True:
            input_text = ""
            output_text = ""
            async for chunk in session.receive():
                if chunk.tool_call:
                    function_responses = []
                    for fc in chunk.tool_call.function_calls:
                        function_response = FunctionResponse(
                            id=fc.id,
                            response=tools[fc.name](**fc.args)
                        )
                        function_responses.append(function_response)

                    await session.send_tool_response(function_responses=function_responses)
                if chunk.server_content and chunk.server_content.output_transcription:
                    output_text = output_text + chunk.server_content.output_transcription.text                
                # handles text modality
                # if chunk.server_content and chunk.server_content.model_turn and chunk.server_content.model_turn.parts:
                #     for part in chunk.server_content.model_turn.parts:
                #         if part.text is not None:
                #             text = text + part.text
                if chunk.server_content and chunk.server_content.input_transcription:
                    input_text = input_text + chunk.server_content.input_transcription.text
                if chunk.data is not None:
                    audio_output_queue.put_nowait(chunk.data)
                if chunk.server_content and chunk.server_content.turn_complete:
                    while not audio_output_queue.empty():
                        audio_output_queue.get_nowait()

            print("You: ", input_text)
            print("Alex: ", output_text)

            if output_text.strip().endswith("."):
                print("Goodbye!")
                # exit(0)

if __name__ == "__main__":
    event_loop = asyncio.new_event_loop()
    event_loop.run_until_complete(main(event_loop))
