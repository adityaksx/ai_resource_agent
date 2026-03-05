import trafilatura

def extract_article(url):

    downloaded = trafilatura.fetch_url(url)

    text = trafilatura.extract(downloaded)

    return text