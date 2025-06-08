import asyncio
from dataclasses import dataclass
from random import randint
from typing import Any, Awaitable, Callable, Literal
from functools import wraps
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
from google.genai.live import AsyncSession
import time
import vlc

@dataclass
class NowPlayingInfo:
    state: Literal["playing", "paused", "stopped"]
    platform: str | None
    capabilities: list[Literal["play", "pause", "stop", "skip_next", "skip_previous"]]
    track: str | None = None
    artist: str | None = None


def get_tools(event_loop: asyncio.AbstractEventLoop, sessions: list[AsyncSession]) -> list[Callable]:
    """There's always only one session, it's just it's a list because the session won't be available immediately"""
    
    # Initialize Spotify client
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        redirect_uri="http://127.0.0.1:8888/callback",
        scope="user-read-playback-state user-modify-playback-state",
        cache_handler=CacheFileHandler(cache_path=".spotipy-cache"),
    ))
    
    # Find the librespot device
    device_id = None
    devices = sp.devices()
    for device in devices["devices"]:
        if device["name"] == "Alex Assistant":
            device_id = device["id"]
            break

    def async_tool(func: Callable[..., Awaitable]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            event_loop.create_task(func(*args, **kwargs))
            return None
        return wrapper
    
    now_playing = NowPlayingInfo(
        state="stopped",
        platform=None,
        capabilities=[]
    )

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
        return {"status": "success", "message": f"Timer '{name}' set for {hours}h {minutes}m {seconds}s."}

    def get_current_temperature(city: str, country: str = "United Kingdom") -> dict:
        """Gets the current temperature for a given location.

        Args:
            city: The city name
            country: The country name if known (optional - defaults to "United Kingdom")

        Returns:
            A dictionary containing the temperature in celsius, the city, and the country
        """
        temp = randint(10, 20)
        return {"status": "success", "message": f"Current temperature in {city}, {country} is {temp}°C.", "celsius": temp, "city": city, "country": country}
    
    def play_spotify(query: str, type: str) -> dict:
        """Play music or podcasts("shows", which have "episodes") on Spotify

        You only need to call this tool once
        
        Args:
            query: The search query for a song, album, or artist.
                You should narrow down your search using field filters. The available filters are album, artist, track, year, upc, tag:hipster, tag:new, isrc, and genre. Each field filter only applies to certain result types.
                The artist and year filters can be used while searching albums, artists and tracks. You can filter on a single year or a range (e.g. 1955-1960).
                The album filter can be used while searching albums and tracks.
                The genre filter can be used while searching artists and tracks.
                The isrc and track filters can be used while searching tracks.
                The upc, tag:new and tag:hipster filters can only be used while searching albums. The tag:new filter will return albums released in the past two weeks and tag:hipster can be used to return only albums with the lowest 10% popularity.
                Example: "Money Money Money artist:ABBA"
            type: The type of search to perform. Can be "track", "album", "artist", "show", "episode", or "all"
            
        Returns:
            A dictionary containing the status and what's playing
        """
        if device_id is None:
            return {"status": "error", "message": "Playback device not found."}
            
        # Search for the track
        if type == "all":
            type = "track,album,artist,show,episode"
        results: Any = sp.search(q=query, type=type, limit=1)
        
        if "tracks" in results and results["tracks"]["items"]:
            type = "track"
            item = results["tracks"]["items"][0]
            now_playing_response = f"Now playing \"{item['name']}\" by {', '.join(artist['name'] for artist in item['artists'])} from the album {item['album']['name']}"
            uri = item["uri"]
        elif "albums" in results and results["albums"]["items"]:
            type = "album"
            item = results["albums"]["items"][0]
            now_playing_response = f"Now playing album \"{item['name']}\" by {', '.join(artist['name'] for artist in item['artists'])}"
            uri = item["uri"]
        elif "artists" in results and results["artists"]["items"]:
            type = "artist"
            item = results["artists"]["items"][0]
            now_playing_response = f"Now playing music by artist \"{item['name']}\""
            uri = item["uri"]
        elif "shows" in results and results["shows"]["items"]:
            type = "show"
            item = results["shows"]["items"][0]
            now_playing_response = f"Now playing podcast \"{item['name']}\" by {item['publisher']}"
            uri = item["uri"]
        elif "episodes" in results and results["episodes"]["items"]:
            type = "episode"
            item = results["episodes"]["items"][0]
            now_playing_response = f"Now playing episode \"{item['name']}\" from a podcast"
            uri = item["uri"]
        else:
            return {"status": "error", "message": "No results found for that spotify query"}
        
        # Start playback
        try:
            if type == "track" or type == "episode":
                sp.start_playback(device_id=device_id, uris=[uri])
            else:
                sp.start_playback(device_id=device_id, context_uri=uri)
            return {"status": "success", "message": now_playing_response}
        except Exception as e:
            return {"status": "error", "message": f"Unknown error. Cannot play spotify. Please try again. Details: {str(e)}"}

    def stop() -> dict:
        """Stop any current audio output or event. If the user asks to "Stop" or "Pause" you should use this tool. For music or anything.
        If the user asks to stop, then after using this tool just respond with "Ok"
        
        Returns:
            A dictionary containing the status of the operation
        """
        if device_id is None:
            return {"status": "error", "message": "Error: Playback device not found."}
            
        try:
            sp.pause_playback(device_id=device_id)
            return {"status": "success", "message": "Playback stopped"}
        except Exception as e:
            return {"status": "error", "message": f"Error: Unknown error. Cannot play spotify. Please try again. Details: {str(e)}"}

    def play() -> dict:
        """Resume playback of the currently loaded track/episode/etc.
        If the user asks to resume or continue playback, use this tool.
        
        Returns:
            A dictionary containing the status of the operation
        """
        if device_id is None:
            return {"status": "error", "message": "Error: Playback device not found."}
            
        try:
            sp.start_playback(device_id=device_id)
            return {"status": "success", "message": "Playback resumed"}
        except Exception as e:
            return {"status": "error", "message": f"Error: Unknown error. Cannot resume playback. Please try again. Details: {str(e)}"}

    def skip_track(direction: str = "next") -> dict:
        """Skip to the next or previous track in the current context.
        
        Args:
            direction: Either "next" or "previous" to indicate which direction to skip
            
        Returns:
            A human-readable string describing the result of the operation
        """
        if device_id is None:
            return {"status": "error", "message": "Error: Playback device not found."}
            
        try:
            if direction == "next":
                sp.next_track(device_id=device_id)
            elif direction == "previous":
                sp.previous_track(device_id=device_id)
            else:
                return {"status": "error", "message": "Error: Invalid skip direction. Must be 'next' or 'previous'."}
            time.sleep(0.5)
            current = sp.current_playback()
            if not current or not current["item"]:
                return {"status": "success", "message": f"Skipped to {direction} track."}
            item = current["item"]
            if item["type"] == "track":
                track_name = item["name"]
                artists = ", ".join(artist["name"] for artist in item["artists"])
                album = item["album"]["name"]
                return {"status": "success", "message": f"Now playing \"{track_name}\" by {artists} from the album \"{album}\"."}
            elif item["type"] == "episode":
                episode_name = item["name"]
                show = item.get("show", {}).get("name", "a podcast")
                return {"status": "success", "message": f"Now playing episode \"{episode_name}\" from \"{show}\"."}
            else:
                return {"status": "success", "message": f"Skipped to {direction} track."}
        except Exception as e:
            return {"status": "error", "message": f"Error: Unknown error. Cannot skip track. Please try again. Details: {str(e)}"}

    vlc_instance = vlc.Instance('--no-xlib')  # Use --no-xlib to avoid X11 dependency on headless systems
    vlc_media_player = vlc_instance.media_player_new()

    def play_radio_station(station: str) -> dict:
        """Play a UK radio station (BBC Radio 1, 2, 3, 4, 1 Dance, 1 Anthems, Classic FM) in the background using VLC.
        Args:
            station: The name of the radio station (e.g. 'bbc radio 1', 'bbc radio 2', 'classic fm', etc.)
        Returns:
            A string describing the result.
        """
        # Map station names to 320k HLS (m3u8) stream URLs from radiofeeds.co.uk/hifi.asp (using lstn.lv and thisisdax.com)
        streams = {
            'bbc radio 1': 'https://lstn.lv/bbcradio.m3u8?station=bbc_radio_one&bitrate=320000',
            'bbc radio 2': 'https://lstn.lv/bbcradio.m3u8?station=bbc_radio_two&bitrate=320000',
            'bbc radio 3': 'https://lstn.lv/bbcradio.m3u8?station=bbc_radio_three&bitrate=320000',
            'bbc radio 4': 'https://lstn.lv/bbcradio.m3u8?station=bbc_radio_fourfm&bitrate=320000',
            'bbc radio 1 dance': 'https://lstn.lv/bbcradio.m3u8?station=bbc_radio_one_dance&bitrate=320000',
            'bbc radio 1 anthems': 'https://lstn.lv/bbcradio.m3u8?station=bbc_radio_one_anthem&bitrate=320000',
            'classic fm': 'http://icecast.thisisdax.com/ClassicFMMP3.m3u',
        }
        key = station.strip().lower()
        if key not in streams:
            return {"status": "error", "message": f"Error: Station '{station}' not recognized. Available: {', '.join(streams.keys())}"}
        url = streams[key]
        vlc_media_player.stop()
        media = vlc_instance.media_new(url)
        vlc_media_player.set_media(media)
        vlc_media_player.play()
        return {"status": "success", "message": f"Now playing {station.title()} radio."}

    return [set_timer, get_current_temperature, play_spotify, stop, play, skip_track, play_radio_station]
