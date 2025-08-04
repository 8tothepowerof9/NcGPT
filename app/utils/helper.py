from typing import List
from bs4 import BeautifulSoup
from qdrant_client.models import QueryResponse

class Helper:
    
    @staticmethod
    def extract_points(points: List[QueryResponse]) -> List[str]:
        context = [point.payload.get("text") for point in points]
        return context
    
    @staticmethod
    def compact_text(soup: BeautifulSoup) -> str:
        raw = soup.get_text(separator=" ")
        return " ".join(raw.split())
    
helper = Helper()