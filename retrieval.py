import config
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, IndexIVFFlat
import numpy as np
import logging
import requests


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
    

# Multi-turn retrieval
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

    global faiss_index, faiss_contents_map, faiss_metadata_map, faiss_id_map

    for turn in range(max_turns):
        logging.info(f"Turn {turn+1} of {max_turns}: {query}")

        # collect contexts from web search
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


