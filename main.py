import asyncio
import os
import pyaudio

from dotenv import load_dotenv
from google import genai
from google.genai.types import FunctionResponse, Blob, LiveConnectConfig
from google.genai.live import AsyncSession

from tools import get_current_temperature

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
model = "gemini-2.0-flash-live-001"

tools = { tool.__name__: tool for tool in [
    get_current_temperature
] }

config = LiveConnectConfig(
    response_modalities=["TEXT"],
    system_instruction="Respond concisely. The user's current location is Cambridge, UK. Finish your response with DONE if the user's request is complete.",
    tools=tools.values(),
)

async def record_audio(session: AsyncSession):
    """Record audio and send it to Gemini in real-time."""
    SAMPLE_RATE = 16000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    CHUNK_SIZE = 1024

    audio = pyaudio.PyAudio()
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

async def main():
    async with (
        client.aio.live.connect(model=model, config=config) as session,
        asyncio.TaskGroup() as tg
    ):
        print("Alex is now listening to your microphone...")
        print()
        # Start audio recording task
        audio_task = tg.create_task(record_audio(session))
        
        while True:
            text = ""
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

                elif chunk.server_content and chunk.server_content.model_turn and chunk.server_content.model_turn.parts:
                    for part in chunk.server_content.model_turn.parts:
                        if part.text is not None:
                            text = text + part.text
            print(text)
            if text.endswith("DONE"):
                print("Alex is done with your request.")
                return

if __name__ == "__main__":
    asyncio.run(main())
