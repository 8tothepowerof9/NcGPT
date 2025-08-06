import heapq
import traceback
from typing import List
from bs4 import BeautifulSoup
from qdrant_client.models import QueryResponse

class Helper:
    
    @staticmethod
    def reciprocal_rank_fusion(
        points: List[QueryResponse] = None,
        n_points: int = None,
        payload: List[str] = None,
        k: int = 50
    ) -> List[str]:
        
        try:
            dense_results = points[0].points
            sparse_results = points[1].points
            
            dense_scores = {str(r.id): r.score for r in dense_results}
            sparse_scores = {str(r.id): r.score for r in sparse_results}
            
            all_doc_ids = dense_scores.keys() | sparse_scores.keys()
            doc_lookup = {str(result.id): result for result in dense_results + sparse_results}

            dense_ranked = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
            sparse_ranked = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)

            dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_ranked)}
            sparse_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sparse_ranked)}

            rrf_scores = {
                doc_id: (1 / (k + dense_ranks.get(doc_id, len(dense_results) + 1))) +
                        (1 / (k + sparse_ranks.get(doc_id, len(sparse_results) + 1)))
                for doc_id in all_doc_ids
            }

            if n_points and n_points < len(rrf_scores):
                top_doc_ids = [doc_id for doc_id, _ in heapq.nlargest(n_points, rrf_scores.items(), key=lambda x: x[1])]
            else:
                top_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:n_points]

            context_list = []
            for idx, doc_id in enumerate(top_doc_ids):
                doc = doc_lookup[doc_id]
                payload_content = [f"Context {idx}: {doc.payload.get(key, '')}" for key in payload]
                context_list.append("\n".join(payload_content))

            return context_list

        except Exception as e:
            traceback.print_exc()
            return [f"RRF Error: {e}"]
    
    @staticmethod
    def compact_text(soup: BeautifulSoup) -> str:
        raw = soup.get_text(separator=" ")
        return " ".join(raw.split())
    
helper = Helper()