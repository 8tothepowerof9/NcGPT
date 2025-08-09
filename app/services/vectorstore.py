import uuid
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Sequence, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import QueryResponse
from ..core.config import QDRANT_API_KEY, QDRANT_URL


@dataclass(frozen=True)
class QdrantCollectionConfig:
    name: str
    dense_name: str = "dense-embedding"
    sparse_name: str = "sparse-embedding"
    dense_dim: int = 1024
    distance: models.Distance = models.Distance.COSINE
    # Optional: tune HNSW, quantization, etc., per named vector
    hnsw_config: Optional[models.HnswConfigDiff] = None
    on_disk_payload: bool = False  # set True for huge payloads


class VectorStore:
    """
    Manages multiple Qdrant collections with named dense+sparse vectors.
    Provides lifecycle ops, batch upserts, and server-side hybrid retrieval.
    """

    def __init__(self, url: str = QDRANT_URL, api_key: str = QDRANT_API_KEY):
        """
        Initialize the VectorStore with a Qdrant client.

        Args:
            url (str, optional): Url to Qdrant storage. If using local, set to "http://localhost:6333". Defaults to QDRANT_URL.
            api_key (str, optional): api key of Qdrant storage. If using local, this can be None. Defaults to QDRANT_API_KEY.
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.__configs: Dict[str, QdrantCollectionConfig] = {}

    def get_client(self) -> QdrantClient:
        """Get the underlying Qdrant client instance."""
        return self.client

    # ---------- Registry & lifecycle ----------

    def register_collection(
        self, config: QdrantCollectionConfig, ensure: bool = True
    ) -> None:
        """
        Register config locally and optionally create/ensure it exists in Qdrant.

        Args:
            config: Collection configuration to register.
            ensure: If True, create the collection if it does not exist.
        """
        self.__configs[config.name] = config
        if ensure:
            self.create_collection(config)

    def get_config(self, collection_name: str) -> QdrantCollectionConfig:
        """
        Get the configuration for a registered collection.

        Args:
            collection_name (str): Name of the collection to retrieve config for.

        Raises:
            KeyError: If the collection is not registered.

        Returns:
            QdrantCollectionConfig: Configuration for the specified collection.
        """
        if collection_name not in self.__configs:
            raise KeyError(
                f"Collection '{collection_name}' is not registered. "
                f"Call register_collection(QdrantCollectionConfig(...))."
            )
        return self.__configs[collection_name]

    def create_collection(self, config: QdrantCollectionConfig) -> None:
        """Create collection with named dense + sparse vectors if missing.

        Args:
            config (QdrantCollectionConfig): Configuration for the collection to create.
        """
        if not self.client.collection_exists(collection_name=config.name):
            self.client.create_collection(
                collection_name=config.name,
                vectors_config={
                    config.dense_name: models.VectorParams(
                        size=config.dense_dim,
                        distance=config.distance,
                        hnsw_config=config.hnsw_config,
                    )
                },
                sparse_vectors_config={
                    config.sparse_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
                on_disk_payload=config.on_disk_payload,
            )

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and its configuration.

        Args:
            collection_name (str): Name of the collection to delete.
        """
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
        self.__configs.pop(collection_name, None)

    def list_collections(self) -> List[str]:
        """
        List collections on the server (not just registered ones).
        """
        res = self.client.get_collections()
        return [c.name for c in res.collections]

    def list_registered(self) -> List[str]:
        """
        List collections registered in this VectorStore.
        """
        return list(self.__configs.keys())

    def count(self, collection_name: str) -> int:
        """
        Count the number of points in a collection.

        Args:
            collection_name (str): Name of the collection to count points in.

        Returns:
            int: Number of points in the collection.
        """
        return self.client.count(collection_name, exact=True).count

    # ---------- Indexes / payload ----------

    def create_payload_index_text(
        self,
        collection_name: str,
        field_name: str,
        tokenizer: models.TokenizerType = models.TokenizerType.WORD,
        lowercase: bool = True,
        min_token_len: int = 2,
        max_token_len: int = 50,
        wait: bool = True,
    ) -> None:
        """
        Create a text index on a payload field in the collection.

        Args:
            collection_name (str): Name of the collection to create the index in.
            field_name (str): Name of the payload field to index.
            tokenizer (TokenizerType): Tokenizer to use for indexing.
            lowercase (bool): Whether to lowercase tokens.
            min_token_len (int): Minimum token length to index.
            max_token_len (int): Maximum token length to index.
            wait (bool): Whether to wait for the index creation to complete.
        """
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=tokenizer,
                lowercase=lowercase,
                min_token_len=min_token_len,
                max_token_len=max_token_len,
            ),
            wait=wait,
        )

    # ---------- Mutations ----------

    def upsert_points(
        self,
        collection_name: str,
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[Tuple[Sequence[int], Sequence[float]]],
        payloads: Sequence[Dict[str, Any]],
        ids: Optional[Sequence[str]] = None,
        wait: bool = False,
    ) -> None:
        """
        Upsert points with per-point sparse vectors.

        Args:
            collection_name (str): Name of the collection to upsert into.
            dense_vectors (Sequence[Sequence[float]]): List of dense vectors.
            sparse_vectors (Sequence[Tuple[Sequence[int], Sequence[float]]]): List of sparse vectors as (indices, values) tuples.
            payloads (Sequence[Dict[str, Any]]): List of payloads for each point.
            ids (Optional[Sequence[str]]): Optional list of point IDs. If None, generates random UUIDs.
            wait (bool): Whether to wait for the upsert operation to complete.
        """
        cfg = self.get_config(collection_name)
        n = len(dense_vectors)
        if not (
            len(sparse_vectors) == len(payloads) == n and (ids is None or len(ids) == n)
        ):
            raise ValueError(
                "Lengths of dense_vectors, sparse_vectors, payloads, and ids (if provided) must match"
            )

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]

        points = []
        for pid, dv, (idx, val), pl in zip(
            ids, dense_vectors, sparse_vectors, payloads
        ):
            points.append(
                models.PointStruct(
                    id=pid,
                    vector={
                        cfg.dense_name: list(dv),
                        cfg.sparse_name: models.SparseVector(
                            indices=list(idx), values=list(val)
                        ),
                    },
                    payload=pl,
                )
            )

        self.client.upsert(collection_name=collection_name, points=points, wait=wait)

    # ---------- Queries ----------

    def hybrid_search(
        self,
        collection_name: str,
        dense_query: Sequence[float],
        sparse_idx: Sequence[int],
        sparse_val: Sequence[float],
        k: int = 20,
        query_filter: Optional[models.Filter] = None,
        with_vectors: bool = False,
        fusion: models.Fusion = models.Fusion.RRF,
    ) -> QueryResponse:
        """
        Server-side hybrid retrieval with prefetch (dense + sparse) and RRF fusion.

        Args:
            collection_name (str): Name of the collection to search.
            dense_query (Sequence[float]): Dense vector query.
            sparse_idx (Sequence[int]): Sparse vector indices.
            sparse_val (Sequence[float]): Sparse vector values.
            k (int): Number of nearest neighbors to return.
            query_filter (Optional[models.Filter]): Optional filter to apply to the query.
            with_vectors (bool): Whether to return vectors in the response.
            fusion (models.Fusion): Fusion method to use for combining results.

        Returns:
            QueryResponse: The search results.
        """
        cfg = self.get_config(collection_name)
        return self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=list(dense_query),
                    using=cfg.dense_name,
                    limit=k,
                    query_filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse_idx), values=list(sparse_val)
                    ),
                    using=cfg.sparse_name,
                    limit=k,
                    query_filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=fusion),
            with_payload=True,
            with_vectors=with_vectors,
            limit=k,
        )

    def search_dense(
        self,
        collection_name: str,
        dense_query: Sequence[float],
        k: int = 20,
        query_filter: Optional[models.Filter] = None,
        with_vectors: bool = False,
    ) -> QueryResponse:
        """
        Dense-only search using the named vector.

        Args:
            collection_name (str): Name of the collection to search.
            dense_query (Sequence[float]): Dense vector query.
            k (int): Number of nearest neighbors to return.
            query_filter (Optional[models.Filter]): Optional filter to apply to the query.
            with_vectors (bool): Whether to return vectors in the response.

        Returns:
            QueryResponse: The search results.
        """
        cfg = self.get_config(collection_name)
        return self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=list(dense_query),
                    using=cfg.dense_name,
                    limit=k,
                    query_filter=query_filter,
                )
            ],
            with_payload=True,
            with_vectors=with_vectors,
            limit=k,
        )

    def search_sparse(
        self,
        collection_name: str,
        sparse_idx: Sequence[int],
        sparse_val: Sequence[float],
        k: int = 20,
        query_filter: Optional[models.Filter] = None,
        with_vectors: bool = False,
    ) -> QueryResponse:
        """
        Sparse-only search using the named sparse vector.

        Args:
            collection_name (str): Name of the collection to search.
            sparse_idx (Sequence[int]): Sparse vector indices.
            sparse_val (Sequence[float]): Sparse vector values.
            k (int): Number of nearest neighbors to return.
            query_filter (Optional[models.Filter]): Optional filter to apply to the query.
            with_vectors (bool): Whether to return vectors in the response.

        Returns:
            QueryResponse: The search results.
        """
        cfg = self.get_config(collection_name)
        return self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse_idx), values=list(sparse_val)
                    ),
                    using=cfg.sparse_name,
                    limit=k,
                    query_filter=query_filter,
                )
            ],
            with_payload=True,
            with_vectors=with_vectors,
            limit=k,
        )

    # ---------- Deletions ----------

    def delete_points_by_filter(
        self, collection_name: str, flt: models.Filter, wait: bool = True
    ) -> None:
        """
        Delete points matching a filter from the collection.

        Args:
            collection_name (str): Name of the collection to delete points from.
            flt (models.Filter): Filter to match points for deletion.
            wait (bool, optional): Whether to wait for the operation to complete. Defaults to True.
        """
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=flt),
            wait=wait,
        )

    def delete_points_by_ids(
        self, collection_name: str, ids: Sequence[str], wait: bool = True
    ) -> None:
        """
        Delete points by their IDs from the collection.

        Args:
            collection_name (str): Name of the collection to delete points from.
            ids (Sequence[str]): List of point IDs to delete.
            wait (bool, optional): Whether to wait for the operation to complete. Defaults to True.
        """
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=list(ids)),
            wait=wait,
        )
