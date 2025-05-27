import asyncio
from random import randint
from typing import Awaitable, Callable
from functools import wraps
from google.genai.live import AsyncSession


def get_tools(event_loop: asyncio.AbstractEventLoop, sessions: list[AsyncSession]) -> list[Callable]:
    """There's always only one session, it's just it's a list because the session won't be available immediately"""
    def async_tool(func: Callable[..., Awaitable]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            event_loop.create_task(func(*args, **kwargs))
            return None
        return wrapper

    @async_tool
    async def set_timer(name: str, hours: int, minutes: int, seconds: int, context: str = ""):
        """Sets a timer for a given hours, minutes, and seconds. If you can't think of a name use the empty string.

        You will receive a system message when the timer is finished.

        <context> is optional and must be used to pass in any additional information you need to remind yourself when the timer is up.
        For example if the user asked to be told something when the timer is up, you should pass the full request in as the <context> parameter.

        Remember to always tell the user which timer has finished and how long it was.
        """
        duration_seconds = hours * 3600 + minutes * 60 + seconds
        await asyncio.sleep(duration_seconds)
        await sessions[0].send_realtime_input(
            text=f"<system>Tell the user that their {hours} hour {minutes} minute {seconds} second {name} timer finished! and <context>{context}</context></system>"
        )
    
    def get_current_temperature(city: str, country: str = "United Kingdom") -> dict:
        """Gets the current temperature for a given location.

        Args:
            city: The city name
            country: The country name if known (optional - defaults to "United Kingdom")

        Returns:
            A dictionary containing the temperature in celsius, the city, and the country
        """
        return {"celsius": randint(10, 20), "city": city, "country": country}
    
    return [set_timer, get_current_temperature]
