from typing import List
from qdrant_client.models import QueryResponse

class Helper:
    
    @staticmethod
    def extract_points(points: List[QueryResponse]) -> List[str]:
        context = [point.payload.get("text") for point in points]
        return context
    
helper = Helper()