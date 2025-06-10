import asyncio
from dataclasses import dataclass, asdict
from random import randint
from typing import Any, Awaitable, Callable, Literal, Optional
from functools import wraps
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
from google.genai.live import AsyncSession
import time
import vlc
from asyncio import Queue

@dataclass
class NowPlayingInfo:
    state: Literal["playing", "paused", "stopped"]
    platform: str | None
    capabilities: list[Literal["play", "pause", "stop", "skip_next", "skip_previous"]]
    track: str | None = None
    artist: str | None = None
    # Generic control functions for the active platform
    stop: Optional[Callable[[], None]] = None
    resume: Optional[Callable[[], None]] = None
    skip_next: Optional[Callable[[], None]] = None
    skip_prev: Optional[Callable[[], None]] = None


def get_tools(event_loop: asyncio.AbstractEventLoop, event_queue: Queue) -> list[Callable]:
    
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
        capabilities=[],
    )

    def _clear_controls():
        now_playing.stop = None
        now_playing.resume = None
        now_playing.skip_next = None
        now_playing.skip_prev = None

    def _set_spotify_controls():
        def _stop():
            sp.pause_playback(device_id=device_id)
            now_playing.state = "paused"
            now_playing.capabilities = ["play", "stop", "skip_next", "skip_previous"]
        def _resume():
            sp.start_playback(device_id=device_id)
            now_playing.state = "playing"
            now_playing.capabilities = ["pause", "stop", "skip_next", "skip_previous"]
        def _next():
            sp.next_track(device_id=device_id)
        def _prev():
            sp.previous_track(device_id=device_id)
        _clear_controls()
        now_playing.stop = _stop
        now_playing.resume = _resume
        now_playing.skip_next = _next
        now_playing.skip_prev = _prev

    def _set_radio_controls():
        def _stop():
            vlc_media_player.pause()
            now_playing.state = "paused"
            now_playing.capabilities = ["play", "stop"]
        def _resume():
            vlc_media_player.play()
            now_playing.state = "playing"
            now_playing.capabilities = ["pause", "stop"]
        _clear_controls()
        now_playing.stop = _stop
        now_playing.resume = _resume
        # Radio has no skip
        now_playing.skip_next = None
        now_playing.skip_prev = None

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
        message = f"<system>Tell the user that their {hours} hour {minutes} minute {seconds} second {name} timer finished! and {context}</system>"
        await event_queue.put(message)

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
        # Stop other platform if running
        if now_playing.platform and now_playing.stop:
            try:
                now_playing.stop()
            except Exception as e:
                return {"status": "error", "message": f"Failed to stop existing playback: {str(e)}"}

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
            # Update now_playing information
            now_playing.state = "playing"
            now_playing.platform = "spotify"
            now_playing.capabilities = ["pause", "stop", "skip_next", "skip_previous"]
            if type == "track":
                now_playing.track = item["name"]
                now_playing.artist = ", ".join(a["name"] for a in item["artists"])
            else:
                now_playing.track = item.get("name")
                now_playing.artist = None
            # Set callbacks for spotify
            _set_spotify_controls()
            return {"status": "success", "message": now_playing_response}
        except Exception as e:
            return {"status": "error", "message": f"Unknown error. Cannot play spotify. Please try again. Details: {str(e)}"}

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
        # Stop other platform if running
        if now_playing.platform and now_playing.stop:
            try:
                now_playing.stop()
            except Exception as e:
                return {"status": "error", "message": f"Failed to stop existing playback: {str(e)}"}
        vlc_media_player.stop()
        media = vlc_instance.media_new(url)
        vlc_media_player.set_media(media)
        vlc_media_player.play()
        # Update now_playing for radio
        now_playing.state = "playing"
        now_playing.platform = "radio"
        now_playing.capabilities = ["pause", "stop"]
        now_playing.track = station.title()
        now_playing.artist = None
        # Set radio callbacks
        _set_radio_controls()
        return {"status": "success", "message": f"Now playing {station.title()} radio."}

    def stop() -> dict:
        """Stop the current playback, whether Spotify or radio.
        If the user asks for you to "stop" or "pause" or "shut up" etc., use this tool.
        If nothing is playing, an error is returned.
        """
        if now_playing.stop is None:
            return {"status": "error", "message": "Nothing is currently playing."}

        try:
            now_playing.stop()
        except Exception as e:
            return {"status": "error", "message": f"Failed to stop playback: {str(e)}"}

        return {"status": "success", "message": "Playback stopped"}

    def resume_music() -> dict:
        """Resume playback based on what was previously playing (Spotify or radio)."""
        if now_playing.resume is None:
            return {"status": "error", "message": "Cannot resume current platform."}

        try:
            now_playing.resume()
        except Exception as e:
            return {"status": "error", "message": f"Failed to resume playback: {str(e)}"}

        return {"status": "success", "message": "Playback resumed"}

    def skip_track(direction: str = "next") -> dict:
        """Skip to the next or previous track in the current context.
        
        Args:
            direction: Either "next" or "previous" to indicate which direction to skip
            
        Returns:
            A human-readable string describing the result of the operation
        """
        if now_playing.platform != "spotify" or now_playing.skip_next is None:
            return {"status": "error", "message": "Cannot skip track when radio is playing."}

        if direction == "next":
            now_playing.skip_next()
        elif direction == "previous":
            if now_playing.skip_prev is None:
                return {"status": "error", "message": "Previous track not supported."}
            now_playing.skip_prev()
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
            now_playing.track = track_name
            now_playing.artist = artists
            return {"status": "success", "message": f"Now playing \"{track_name}\" by {artists} from the album \"{album}\"."}
        elif item["type"] == "episode":
            episode_name = item["name"]
            show = item.get("show", {}).get("name", "a podcast")
            now_playing.track = episode_name
            now_playing.artist = show
            return {"status": "success", "message": f"Now playing episode \"{episode_name}\" from \"{show}\"."}
        else:
            return {"status": "success", "message": f"Skipped to {direction} track."}

    def get_now_playing() -> dict:
        """Return a human-readable description of what is currently playing."""
        if now_playing.platform is None or now_playing.state == "stopped":
            return {"status": "success", "message": "Nothing is currently playing."}

        # Build description based on platform and state
        state_word = "Playing" if now_playing.state == "playing" else "Paused"

        if now_playing.platform == "spotify":
            if now_playing.track and now_playing.artist:
                description = f"{state_word} \"{now_playing.track}\" by {now_playing.artist} on Spotify."
            elif now_playing.track:
                description = f"{state_word} {now_playing.track} on Spotify."
            else:
                description = f"{state_word} on Spotify."
        elif now_playing.platform == "radio":
            station = now_playing.track or "radio"
            description = f"{state_word} {station}."
        else:
            description = "Unknown playback platform."

        return {"status": "success", "message": description}

    vlc_instance = vlc.Instance('--no-xlib')  # Use --no-xlib to avoid X11 dependency on headless systems
    vlc_media_player = vlc_instance.media_player_new()

    return [set_timer, get_current_temperature, play_spotify, stop, resume_music, skip_track, play_radio_station, get_now_playing]
