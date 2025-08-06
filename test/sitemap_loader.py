import requests
from bs4 import BeautifulSoup

def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    response = requests.get(sitemap_url)
    response.raise_for_status()
    # Parse as XML
    soup = BeautifulSoup(response.text, "xml")
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]

# Example usage:
urls = get_urls_from_sitemap("https://dspy.ai/sitemap.xml")
for u in urls:
    print(u)
