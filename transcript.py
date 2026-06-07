import re
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


def get_transcript(video_url: str) -> str:
    """Fetch the transcript of a YouTube video and return as plain text.
    Tries multiple languages: English, Hindi, Telugu, Tamil, Kannada, and auto-generated ones.
    """
    video_id = extract_video_id(video_url)
    api = YouTubeTranscriptApi()

    # Try fetching with a broad language priority list
    try:
        transcript = api.fetch(video_id, languages=['en', 'hi', 'te', 'ta', 'kn', 'ml', 'mr', 'bn', 'gu', 'pa'])
    except Exception:
        # If none of the listed languages match, fetch the first available transcript
        transcript_list = api.list(video_id)
        first_transcript = next(iter(transcript_list))
        transcript = first_transcript.fetch()

    full_text = " ".join(snippet.text for snippet in transcript)
    return full_text


def get_transcript_with_timestamps(video_url: str) -> list[dict]:
    """Fetch the transcript with timestamps."""
    video_id = extract_video_id(video_url)
    api = YouTubeTranscriptApi()

    try:
        transcript = api.fetch(video_id, languages=['en', 'hi', 'te', 'ta', 'kn', 'ml', 'mr', 'bn', 'gu', 'pa'])
    except Exception:
        transcript_list = api.list(video_id)
        first_transcript = next(iter(transcript_list))
        transcript = first_transcript.fetch()

    return [{"text": s.text, "start": s.start, "duration": s.duration} for s in transcript]
