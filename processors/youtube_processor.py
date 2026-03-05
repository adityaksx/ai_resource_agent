import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


def get_video_id(url):

    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]


def download_video(url, path):

    ydl_opts = {
        "format": "best[height<=1080]",
        "outtmpl": f"{path}/%(title)s.%(ext)s",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def get_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)

    transcript = transcript_list.find_transcript(
        ["en", "hi", "en-US", "en-GB"]
    )

    fetched = transcript.fetch()

    text = " ".join([item.text for item in fetched])

    return text