from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader

text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=2000)

def compact_text(soup: BeautifulSoup) -> str:
    raw = soup.get_text(separator=" ")
    return " ".join(raw.split())

sitemap_loader = SitemapLoader(
    web_path="https://dspy.ai/sitemap.xml",
    parsing_function=compact_text
)

docs = sitemap_loader.load()

texts = [text_splitter.split_text(doc.page_content) for doc in docs]

print(texts[0][:100])