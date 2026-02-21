import logging
# cross-encoder for reranking
from sentence_transformers import CrossEncoder
# threading
import threading


# Cross-encoder for reranking
cross_encoder = None
cross_encoder_lock = threading.Lock()
def get_cross_encoder():
    """
    Lazy loading cross-encoder model to avoid blocking during the first run
    """
    global cross_encoder
    with cross_encoder_lock:
        if cross_encoder is None:
            try:
                # for multilingual language support
                cross_encoder = CrossEncoder('sentence-transformers/distiluse-base-multilingual-cased-v2')
                # a more lighter model
                # cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
                logging.info("Loaded cross-encoder model" + cross_encoder.model_name)
            except Exception as e:
                logging.error("Failed to load cross-encoder model: " + str(e))
                cross_encoder = None
    return cross_encoder