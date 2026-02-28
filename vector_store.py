"""
Vector store
    - Embedding
    - Faiss index (FlatL2, IVFFlat, IVFPQ)
    - add, search, delete
"""

import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from doc_processor import Chunk
from typing import List, Dict, Tuple, Optional, Any
import json

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    score: float

class Embedding:
    """
    Embedding using SentenceTransformer
    """
    # Model choices:
    # English: all-MiniLM-L6-v2 (384 dimensions)
    # Multilingual: paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
    # Chinese: shibing624/text2vec-base-chinese (768 dimensions)

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 64):
        logger.info(f"Using model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size # chunk group size
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model dimension: {self.dimension}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts
        """
        logger.info(f"Encoding {len(texts)} texts")
        if not texts: return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            # SentenceTransformer returns Pytorch Tensors, convert to numpy for Faiss
            convert_to_numpy=True,
        )
        return embeddings.astype("float32") # Faiss requires float32
    

class AutoFaissIndex:
    """
    Faiss just for testing demo, replacing ChromaDB in the future
    This class automatically selects the type of FAISS index based on the number of vectors
    """
    FLAT_THRESHOOLD = 10_000
    IVF_THRESHOOLD = 100_000

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.index_type = None
        self._nlist = None
        self._nprobe = None

    def build(self, vectors: np.ndarray) -> str:
        """
        Build the FAISS index
        
        Args:
            vectors: numpy array of shape (num_vectors, dimension), dtype float32
        
        Returns:
            index_type: str: "FlatL2", "IVFFlat", "IVFPQ"
        """

        n = len(vectors)
        if n == 0:
            raise ValueError("No vectors provided")
        
        index_type = self._select_index_type(n)
        if index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)

        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self._nlist)
            # Determined the positions of nlist centroids.
            self.index.train(vectors)           # K-Means clustering
            self.index.nprobe = self._nprobe

        elif index_type == "IVFPQ":
            m = self._calc_m()
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self._nlist, m, 8)
            self.index.train(vectors)
            self.index.nprobe = self._nprobe

        self.index.add(vectors)
        self.index_type = index_type

        logger.info(
            f"FAISS index built: type={index_type}, n={n}, "
            f"nlist={self._nlist}, nprobe={self._nprobe}"
        )
        return index_type
    
    def add(self, vectors: np.ndarray):
        """
        Add vectors to the existing index
        """
        if self.index is None:
            raise ValueError("Index not built yet")
        self.index.add(vectors)
        logger.info(f"FAISS index added: n={n}. Total vectors: {self.index.ntotal}")

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k nearest neighbors.

        Args:
            query: shape (query cnt, dimension)
            k:     top k nearest neighbors
        Returns:
            distances: shape (N, k)
            indices:   shape (N, k)  ← FAISS internal integer indices
        """
        if self.index is None or self.index.ntotal == 0:
            return np.empty((1, 0)), np.empty((1, 0), dtype=int)

        actual_k = min(k, self.index.ntotal)
        return self.index.search(query, actual_k)


    def _select_index_type(self, n: int) -> str:
        if n < self.FLAT_THRESHOOLD:
            return "FlatL2"
        if n <= self.IVF_THRESHOOLD:
            suggested_nlist = int(np.sqrt(n))
            max_nlist = n // 10
            if max_nlist < 4:
                return "FlatL2"
            self._nlist = min(suggested_nlist, max_nlist, 256)
            self._nprobe = max(1, self._nlist // 10)
            return "IVFFlat"
        
        # For very large datasets, use IVFPQ
        suggested_nlist = int(np.sqrt(n))
        self._nlist = min(suggested_nlist, 1024)
        self._nprobe = max(1, self._nlist // 20)
        return "IVFPQ"
    
    def _calc_m(self) -> int:
        for m in [16, 8, 4, 2, 1]:
            if self._nlist % m == 0:
                return m
        return 1
    
    @property
    def ntotal(self) -> int:
        return self.index.ntotal if self.index else 0
    
    def get_info(self) -> dict:
        return {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "ntotal": self.ntotal,
            "nlist": self._nlist,
            "nprobe": self._nprobe,
        }
    

class VectorStore:
    """
    Main interface for vector store and search.
        - add_documents
        - embed_query
        - build_index, update_index
        - search, return top k results(SearchResult)
    """

    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            batch_size: int = 64,
    ):
        self.embedder = Embedding(model_name=model_name, batch_size=batch_size)
        self.faiss_index = AutoFaissIndex(self.embedder.dimension)

        self._store: Dict[str, Dict] = {} # {chunk_id: chunk_info{content, metadata, faiss_position}}
        self._id_order: List[str] = [] 

    def add(self, chunks: List[Chunk]) -> int:
        """
        Embed and add chunks to the vector store.

        Args:
            chunks: List of Chunk objects to add

        Returns:
            The number of chunks added
        """

        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        # 1. Filter out already-indexed chunks (deduplication by chunk_id)
        final_chunks = [c for c in chunks if c.chunk_id not in self._store]
        if not final_chunks:
            logger.warning("All chunks already exist in the store")
            return 0
        logger.info(f"Adding {len(final_chunks)} chunks to the store")

        # 2. Encode chunks
        texts = [c.content for c in final_chunks]
        vectors = self.embedder.encode(texts)

        # 3. Update and mapping id before adding to FAISS  
        start = len(self._id_order)
        for i, chunk in enumerate(final_chunks):
            self._id_order.append(chunk.chunk_id)

            faiss_pos = start + i
            self._store[chunk.chunk_id] = {
                "content":   chunk.content,
                "metadata":  chunk.to_dict(),
                "faiss_pos": faiss_pos,
            }

        # 4. Build or update the FAISS index
        if self.faiss_index.ntotal == 0:
            self.faiss_index.build(vectors)
        else:
            self.faiss_index.add(vectors)

        logger.info(f"Added {len(chunks)} chunks to the FAISS index")
        return len(final_chunks)
    
    def search(self, query: str, k: int =10) -> List[SearchResult]:
        """
        Search for relevant chunks given a query.

        Args:
            query: The search query string
            k:     The number of top results to return

        Returns:
            A list of SearchResult objects containing chunk_id, content, metadata, and score
        """
        if self.faiss_index.ntotal == 0:
            logger.warning("FAISS index is empty. No results to return.")
            return []

        query_vector = self.embedder.encode([query])
        distances, indices = self.faiss_index.search(query_vector, k)

        results = []
        for dist, faiss_index in zip(distances[0], indices[0]):
            if faiss_index >= len(self._id_order):
                logger.warning(f"FAISS returned index {faiss_index} which is out of bounds")
                continue

            chunk_id = self._id_order[faiss_index]
            chunk_info = self._store.get(chunk_id)
            if chunk_info is None:
                logger.warning(f"Chunk {chunk_id} not found in store")
                continue
            results.append(SearchResult(
                chunk_id = chunk_id,
                chunk_content = chunk_info["content"],
                metadata = chunk_info["metadata"],
                score = float(dist),
            ))
        
        return results
    
    def to_json(self):
        """
        Convert the Chunk object to a JSON string.
        """
        return json.dumps(self.to_dict())
    
    def clean(self):
        self._store.clear()
        self._id_order.clear()
        self.faiss_index = AutoFaissIndex(self.embedder.dimension)
        logger.info(f"Cleared {self.doc_id}")

    def get_info(self):
        return {
            "total_chunks": len(self._id_order),
            "index_info": self.faiss_index.get_info(),
        }




