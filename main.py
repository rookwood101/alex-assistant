import asyncio
import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import FunctionResponse

from tools import get_current_temperature

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
model = "gemini-2.0-flash-live-001"

tools = { tool.__name__: tool for tool in [
    get_current_temperature
] }

config = {
    "response_modalities": ["TEXT"],
    "system_instruction": "Respond concisely",
    "tools": tools.values()
}

async def main():
    print("Hello from alex-assistant!")
    async with client.aio.live.connect(model=model, config=config) as session:
        message = input("Enter a message: ")
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": message}]}, turn_complete=True
        )

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

            elif chunk.server_content:
                if chunk.server_content.executable_code:
                    # there's a bug where gemini is returning executable code as well as a tool response
                    pass
                elif chunk.text is not None:
                    print(chunk.text)
            else:
                print("Unknown chunk type")
                breakpoint()

if __name__ == "__main__":
    asyncio.run(main())
