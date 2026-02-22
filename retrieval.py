import config
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, IndexIVFFlat
import numpy as np
import logging
import requests
import jieba
from rank_bm25 import BM25Okapi


# Embedding model: default
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# Embedding model: chinese language
# EMBEDDING_MODEL = SentenceTransformer('shibing624/text2vec-base-chinese')


# Encapsulation class for automatically selecting the type of FAISS index
class AutoFaissIndex:
    def __init__(self, dimension=384):
        """Auto-indexing for FAISS, optimizing the indexing process"""
        self.dimension = dimension
        self.index = None
        self.index_type = None
        self.nlist = None   # cluster centers - IVF
        self.m = None       # sub-vector segments for PQ
        self.nprobe = None  # number of clusters to search - IVF

        # threshold for dataset size
        self.small_dataset_threshold = 10_000
        self.medium_dataset_threshold = 100_000
        self.large_dataset_threshold = 1_000_000

    def select_index_type(self, num_vectors): 
        """Select the indexing type according to the number of vectors"""
        if num_vectors <= self.small_dataset_threshold:
            self.index_type = 'FlatL2'
            self.index = IndexFlatL2(self.dimension)
            self.nprobe = 1
        
        elif num_vectors <= self.medium_dataset_threshold:
            self.index_type = 'IVFFlat'
            self.nlist = min(100, int(np.sqrt(num_vectors)))
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.nprobe = min(10, max(1, int(self.nplist / 10)))

        elif num_vectors <= self.large_dataset_threshold:
            self.index_type = 'IVFPQ'
            self.nlist = min(256, int(np.sqrt(num_vectors)))
            self.m = min(8, self.dimension // 4)
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.nprobe = min(20, max(1, int(self.nlist / 20)))
        
        return self.index_type
    
    def train(self, vectors):
        """Train the index if necessary (for IVF)"""
        if self.index_type in ['IVFFlat', 'IVFPQ']:
            self.index.train(vectors)
        
    def add(self, vectors):
        """Add vectors to the index"""
        if self.index_type in ['IVFFlat', 'IVFPQ'] and  not self.index.is_trained:
            self.train(vectors)
        self.index.add(vectors)

    def search(self, query_vectors, k=5):
        """
        Search for the most similar vectors
            
            Parameters:
            - query_vectors: the vectors to query(np.array)
            - k: the number of nearest neighbors to return

            Returns:
            - distances: the distances between the query vectors and the nearest neighbors
            - indices: the indices of the nearest neighbors
        """
        if self.index_type in ['IVFFlat', 'IVFPQ']:
            self.index.nprobe = self.nprobe
        
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices
    
    def get_index_info(self):
        """Get the indexing information"""
        return {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "nlist": self.nlist,
            "m": self.m,
            "nprobe": self.nprobe,
            "index_size": self.ntotal if self.index else None
        }
    

# Multi-turn retrieval: 1. FAISS + SerpAPI -> 2. BM25
def multi_turn_retrieval(initial_query, max_turns=3, enable_web_search=False, client=None):
    """
    Four stages retrieval process:
    - 1. FAISS(semantic), BM25(keyword), SerpAPI (web search)
    - 2. Hybrid merge, Rerank(cross-encoder)
    - 3. LLM self-assesment
    - 4. LLM rephrase, recursive retrieval

        Parameters:
        - initial_query: the initial query from the user
        - max_turns: the maximum number of retrieval turns
        - enable_web_search: whether to enable web search (SerpAPI)
        - client: LLM model (ollama, openai)
    
        Returns:
        - final_contents: the final retrieved results after multi-turn
        - contents_ids: the ids of the retrieved results
        - list_metadata: the metadata list of the retrieved results
    """
    query = initial_query
    final_contents = []
    contexts_ids = []
    list_metadata = []

    global faiss_index, faiss_contents_map, faiss_metadata_map, map_index_to_doc_id

    for turn in range(max_turns):
        logging.info(f"Turn {turn+1} of {max_turns}: {query}")

        # 1. collect contexts from web search
        web_search_contexts = []
        if enable_web_search and check_web_search_api_key():
            try:
                web_search_raw_results = web_search(query)
                logging.info(f"Web search returned {len(web_search_contexts)} results")
                # filter and only keep the title and content of the results
                for result in web_search_raw_results:
                    filter_text = f"Title: {result.get('title', '')} Content: {result.get('content', '')}"
                    web_search_contexts.append(filter_text)
            except Exception as e:
                logging.error("Failed to execute web search: " + str(e))

        # 2. semantic retrieval
        # query embedding -> np.array
        query_embedding = EMBEDDING_MODEL([query]) 
        query_embedding_np = np.array(query_embedding).astype(np.float32)
        semantic_retrieval_results = []
        semantic_retrieval_metadata = []
        semantic_retrieval_ids = []

        if faiss_index is not None and hasattr(faiss_index, 'ntotal') and faiss_index.ntotal > 0:
            try:
                distances, indices = faiss_index.search(query_embedding_np, k=10)
                for faiss_index_id in indices[0]:
                    if faiss_index_id != -1 and faiss_index_id < len(map_index_to_doc_id):
                        doc_id = map_index_to_doc_id[faiss_index_id]
                        if doc_id in faiss_contents_map:
                            semantic_retrieval_results.append(faiss_contents_map.get(doc_id, ""))
                            semantic_retrieval_metadata.append(faiss_metadata_map.get(doc_id, {}))
                            semantic_retrieval_ids.append(doc_id)
                        else:
                            logging.warning(f"Doc ID {doc_id} not found in contents map")
            except Exception as e:
                logging.error("Failed to execute semantic retrieval: " + str(e))
        else:
            logging.warning("FAISS index is empty or not initialized")

        
        # 3. BM25 retrieval
        bm25_retrieval_results = BM25SearchEngine.search(query, top_k=10) if BM25SearchEngine.bm25_index else []

        
                



def check_web_search_api_key():
    """Check if the web search API key is set"""
    return config.SERPAPI_KEY is not None and config.SERPAPI_KEY != ""

def web_search(query):
    """
    Web search using SerpAPI, 
    but the results doesn't embedded for avoiding contaimination
    """
    results = serpapi_search(query, result_num=5)
    if not results: 
        logging.warning("No results found for query: " + query)
        return []
    return results

def serpapi_search(query, result_num):
    """Search using SerpAPI"""
    if not config.SERPAPI_KEY: 
        logging.error("No SERPAPI_KEY provided")
    try:
        params = {
            "engine": config.SEARCH_ENGINE,
            "q": query,
            "api_key": config.SERPAPI_KEY,
            "num": result_num,
            "hl": "en",
            "gl": "us",
            "google_domain": "google.com"
        }
        response = requests.get(config.SERPAPI_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return _parse_serpapi_results(data)
    except Exception as e:
        logging.error("Failed to execute SerpAPI search: " + str(e))
        return []
    
def _parse_serpapi_results(data):
    results = []
    if "organic_results" in data:
        for item in data["organic_results"]:
            result = {
                "title": item.get("title", ""),
                "content": item.get("snippet", ""),
                "url": item.get("link", ""),
                "timestamp": item.get("timestamp", "")
            }
            results.append(result)
            logging.info(f"Added result: {result}")
    return results


STOPWORDS = {
    '的', '了', '是', '在', '和', '与', '对', '为', '以', '及',
    '也', '都', '但', '而', '或', '被', '把', '将', '从', '到',
    '之', '其', '等', '中', '该', '此', '那', '这', '有', '无',
}
class BM25SearchEngine:
    def __init__(self, stopwords: set = None):
        self.bm25_index = None
        self.doc_mapping: dict[int, str] = {}
        self.tokenized_corpus: list[list[str]] = []
        self.raw_corpus: list[str] = []
        self.stopwords = stopwords if stopwords is not None else STOPWORDS
    
    def _tokenize(self, text) -> list[str]:
        """Tokenize the given text, filtering out stopwords"""
        return [token for token in jieba.cut(text) if token not in self.stopwords and len(token) > 1]
    
    def build_index(self, documents: list[str], doc_ids: list[str]) -> bool:
        """
        Build the BM25 index from the given documents
        
            Parameters:
            - documents: a list of document texts to be indexed
            - doc_ids: a list of document IDs corresponding to the documents
        """
        # 1. validate inputs
        if not documents:
            raise ValueError("No documents provided")
        if len(documents) != len(doc_ids): 
            raise ValueError("The number of documents and document IDs do not match")

        self.raw_corpus = list(documents)
        self.doc_mapping = {idx: doc_id for idx, doc_id in enumerate(doc_ids)}

        # 2. tokenize documents
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]

        # 3. build BM25 index
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        return True
        
    
    def search(self, query: str, top_k: int=10) -> list[dict]:
        """
        Search for the most relevant documents using BM25
        
            Parameters:
            - query: the search query string
            - top_k: the number of top results to return
            
            Returns:
            - a list of dictionaries containing the document ID, 
                title, and content of the top matching documents
        """

        # 1. validate inputs
        if not self.bm25_index:
            logging.warning("BM25 index is not built yet")
            return []
        if not query: 
            logging.warning("No query provided")
            return []
        
        # 2. 
        top_k = max(1, min(top_k, len(self.raw_corpus)))
        tokenized_query = self._tokenize(query)
        if not tokenized_query: 
            logging.warning("No query tokens found")
            return []
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc_id = self.doc_mapping.get(idx, "")
            if doc_id is None:
                logging.warning(f"Document ID for index {idx} not found")
                continue
            if scores[idx] < 0.01: 
                logging.info(f"Skipping document {doc_id} due to low score: {scores[idx]}")
                continue
            results.append({
                "doc_id": doc_id,
                "title": "",
                "content": "",
                "score": scores[idx]
            })
            logging.info(f"Added result: {results[-1]}")
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def clear(self) -> None:
        """Clear the BM25 index and all associated data"""
        self.bm25_index = None
        self.doc_mapping.clear()
        self.tokenized_corpus.clear()
        self.raw_corpus.clear()