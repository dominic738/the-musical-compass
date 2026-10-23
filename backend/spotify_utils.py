import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import re
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

_sp = None


def get_client():
    """Create the Spotify client on first use so the app can start without Spotify credentials."""
    global _sp
    if _sp is None:
        if not CLIENT_ID or not CLIENT_SECRET:
            raise RuntimeError("SPOTIFY_CLIENT_ID and/or SPOTIFY_CLIENT_SECRET not found. Make sure .env is loaded properly.")
        auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        _sp = spotipy.Spotify(auth_manager=auth_manager)
    return _sp




# Only suffixes that name a different edition of the same recording, so a title
# like "Live Forever" or "Bonus Track Blues" is left alone
EDITION_SUFFIX_RE = re.compile(
    r'\s*(?:[-–—]\s+[^-–—]*|[\(\[][^\)\]]*[\)\]])$'
)
EDITION_WORDS_RE = re.compile(
    r'\b(?:remaster|single version|album version|radio edit|\d{4} mix|mono\b|stereo\b|deluxe|bonus track)',
    re.IGNORECASE
)

# Every new song costs an embedding call, so cap how much one import can add
MAX_PLAYLIST_TRACKS = 100


def clean_track_name(name):
    """Strip edition suffixes like " - 2011 Remaster" or "(Mono)", repeatedly."""
    while True:
        match = EDITION_SUFFIX_RE.search(name)
        if not match or not EDITION_WORDS_RE.search(match.group(0)):
            return name.strip()
        name = name[:match.start()]


def get_playlist(playlist_url):
    """
    Return ([{"title", "artist"}, ...], total_in_playlist) for a public playlist.
    Duplicates, local files and podcast episodes are skipped; at most
    MAX_PLAYLIST_TRACKS tracks are returned, in playlist order.
    """
    sp = get_client()
    results = sp.playlist_items(playlist_url, additional_types=("track",))
    total = results.get("total", 0)
    tracks = []
    seen = set()

    while results:
        for item in results["items"]:
            # Newer API responses name the field "item"; older ones "track"
            track = item.get("track") or item.get("item")
            if not track or track.get("type") != "track" or item.get("is_local"):
                continue

            title = clean_track_name(track["name"])
            # First artist only: LRCLIB matching already accepts "Artist & Someone" credits
            artist = track["artists"][0]["name"] if track["artists"] else ""
            key = (title.lower(), artist.lower())
            if not title or not artist or key in seen:
                continue
            seen.add(key)
            tracks.append({"title": title, "artist": artist})
            if len(tracks) == MAX_PLAYLIST_TRACKS:
                return tracks, total

        results = sp.next(results) if results["next"] else None

    return tracks, total
