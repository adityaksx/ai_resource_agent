from utils.source_detector import detect_source
from processors.youtube_processor import get_video_id, get_transcript
from processors.web_processor import extract_article
from processors.github_processor import clone_repo, read_readme
from llm.summarizer import summarize


def process_link(url):

    source = detect_source(url)

    if source == "youtube":

        vid = get_video_id(url)

        transcript = get_transcript(vid)

        summary = summarize(transcript)

        print(summary)


    elif source == "github":

        repo = clone_repo(url)

        text = read_readme(repo)

        summary = summarize(text)

        print(summary)


    elif source == "web":

        text = extract_article(url)

        summary = summarize(text)

        print(summary)


if __name__ == "__main__":

    url = input("Enter resource link: ")

    process_link(url)