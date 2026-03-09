import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent



@dataclass
class OllamaConfig:
    model:    str = "llama3.2"
    base_url: str = "http://localhost:11434"
    timeout:  int = 180


@dataclass
class OpenAIConfig:
    model:   str = "gpt-4o-mini"
    api_url: str = "https://api.openai.com/v1/chat/completions"
    timeout: int = 120


@dataclass
class LLMConfig:
    backend:     str          = "ollama"
    temperature: float        = 0.7
    max_tokens:  int          = 1536
    ollama:      OllamaConfig = field(default_factory=OllamaConfig)
    openai:      OpenAIConfig = field(default_factory=OpenAIConfig)

    @property
    def active(self) -> OllamaConfig | OpenAIConfig:
        """Return the config block for the currently selected backend."""
        if self.backend == "ollama":
            return self.ollama
        return self.openai


@dataclass
class RetrievalConfig:
    chunk_size:     int   = 512
    chunk_overlap:  int   = 128
    alpha:          float = 0.7
    retrieve_k:     int   = 10
    rerank_top_k:   int   = 5
    max_iterations: int   = 3
    use_reranker:   bool  = True
    reranker_model: str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64


@dataclass
class WebSearchConfig:
    enabled:     bool = False
    engine:      str  = "google"
    num_results: int  = 5
    timeout:     int  = 15
    hl:          str  = "en"
    gl:          str  = "us"


@dataclass
class ServerConfig:
    host:         str       = "0.0.0.0"
    port:         int       = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:5173"])


@dataclass
class Secrets:
    serpapi_key:     Optional[str] = None
    openai_api_key:  Optional[str] = None


@dataclass
class AppConfig:
    llm:        LLMConfig       = field(default_factory=LLMConfig)
    retrieval:  RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding:  EmbeddingConfig = field(default_factory=EmbeddingConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    server:     ServerConfig    = field(default_factory=ServerConfig)
    secrets:    Secrets         = field(default_factory=Secrets)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        logger.warning(f"config.yaml not found at {path}, using defaults")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _populate(instance, data: dict) -> None:
    """
    Recursively populate an instance with values from a dict.
    """
    nested_types = (
        LLMConfig, OllamaConfig, OpenAIConfig,
        RetrievalConfig, EmbeddingConfig,
        WebSearchConfig, ServerConfig,
    )
    for key, value in data.items():
        if not hasattr(instance, key):
            logger.warning(f"Unknown config key '{key}', skipping")
            continue
        current = getattr(instance, key)
        if isinstance(current, nested_types) and isinstance(value, dict):
            _populate(current, value)
        else:
            setattr(instance, key, value)


def load_config(
    yaml_path: Optional[Path] = None,
    env_path:  Optional[Path] = None,
) -> AppConfig:
    cfg = AppConfig()

    # Step 1: YAML overrides defaults
    yaml_path = yaml_path or _PROJECT_ROOT / "config.yaml"
    raw = _load_yaml(yaml_path)
    _populate(cfg, raw)

    # Step 2: .env → Secrets only
    env_file = env_path or _PROJECT_ROOT / raw.get("env_file", ".env")
    if env_file.exists():
        load_dotenv(env_file, override=False)
    else:
        logger.warning(f".env not found at {env_file}")

    cfg.secrets = Secrets(
        serpapi_key     = os.getenv("SERPAPI_KEY"),
        openai_api_key  = os.getenv("OPENAI_API_KEY"),
        siliconflow_key = os.getenv("SILICONFLOW_API_KEY"),
    )

    logger.info(
        f"Config loaded | backend={cfg.llm.backend} "
        f"model={cfg.llm.active.model} | "
        f"port={cfg.server.port} | "
        f"serpapi={'✓' if cfg.secrets.serpapi_key else '✗'}"
    )
    return cfg


cfg: AppConfig = load_config()