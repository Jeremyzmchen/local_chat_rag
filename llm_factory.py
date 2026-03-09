"""

"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Generator, Iterator, Optional

import requests
from dotenv import load_dotenv
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

dotenv_path = Path(__file__).parent / "dev.env"
load_dotenv(dotenv_path)


def _build_session(total_retries: int = 3, backoff: float = 0.3) -> requests.Session:
    """Build a requests.Session with automatic retry on server errors."""
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class BaseLLM(ABC):
    """Interface for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Return the full response as a single string (blocking).

        Args:
            prompt:   the complete prompt string
            **kwargs: backend-specific overrides (temperature, max_tokens, …)

        Returns:
            The model's response
        """

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Yield response tokens / chunks incrementally.

        Args:
            prompt:   the complete prompt string
            **kwargs: backend-specific overrides (temperature, max_tokens, …)

        Yields:
            Successive text chunks.
        """

    def __call__(self, prompt: str, **kwargs) -> str:
        """For internal callers, like when refine_query() calls the LLM."""
        return self.generate(prompt, **kwargs)
    


class OllamaLLM(BaseLLM):
    """ Local model """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
        temperature: float = 0.7,
        max_tokens: int = 1536,
    ):
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._session = _build_session()

    def generate(self, prompt: str, **kwargs) -> str:
        payload = self._build_payload(prompt, stream=False, **kwargs)
        try:
            resp = self._session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.RequestException as e:
            logger.error(f"OllamaLLM.generate failed: {e}")
            raise

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        payload = self._build_payload(prompt, stream=True, **kwargs)
        try:
            with self._session.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        # line.decode() -> bytes to str
                        # json.loads(line.decode()) -> str to dict
                        chunk = json.loads(line.decode()).get("response", "")
                        if chunk:
                            yield chunk
        except requests.RequestException as e:
            logger.error(f"OllamaLLM.stream failed: {e}")
            raise

    def is_available(self) -> bool:
        """Quick health check — returns True if Ollama is reachable."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _build_payload(self, prompt: str, stream: bool, **kwargs) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", self._default_temperature),
                "num_predict": kwargs.get("max_tokens", self._default_max_tokens),
            },
        }
    

class OpenAILLM(BaseLLM):
    """
    Calls any OpenAI-compatible chat-completions endpoint.
    """

    _DEFAULT_URL   = "https://api.openai.com/v1/chat/completions"
    _DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_url: str = _DEFAULT_URL,
        timeout: int = 120,
        temperature: float = 0.7,
        max_tokens: int = 1536,
    ):
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._api_url = api_url
        self._timeout = timeout
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._session = _build_session()

        if not self._api_key:
            logger.warning("OpenAILLM: OPENAI_API_KEY is not set")

    def generate(self, prompt: str, **kwargs) -> str:
        payload = self._build_payload(prompt, stream=False, **kwargs)
        try:
            resp = self._session.post(
                self._api_url,
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())
        except requests.RequestException as e:
            logger.error(f"OpenAILLM.generate failed: {e}")
            raise

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Standard OpenAI SSE streaming.
        Each line: data: {"choices": [{"delta": {"content": "..."}}]}
        Stream ends with: data: [DONE]
        """
        payload = self._build_payload(prompt, stream=True, **kwargs)
        try:
            with self._session.post(
                self._api_url,
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded == "data: [DONE]":
                        break
                    if decoded.startswith("data: "):
                        decoded = decoded[len("data: "):]
                    try:
                        data = json.loads(decoded)
                        chunk = (
                            data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                        )
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
        except requests.RequestException as e:
            logger.error(f"OpenAILLM.stream failed: {e}")
            raise


    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key.strip()}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, prompt: str, stream: bool, **kwargs) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "max_tokens": kwargs.get("max_tokens", self._default_max_tokens),
            "temperature": kwargs.get("temperature", self._default_temperature),
        }

    @staticmethod
    def _parse_response(data: dict) -> str:
        choices = data.get("choices", [])
        if not choices:
            logger.warning("OpenAILLM: empty choices in response")
            return ""
        return choices[0].get("message", {}).get("content", "")



def get_llm(backend: str = "ollama", **kwargs) -> BaseLLM:
    """
    Factory function — returns a configured LLM instance.

    Args:
        backend: "openai" | "ollama"
        **kwargs: forwarded to the LLM constructor

    Usage:
        llm = get_llm("openai")
        llm = get_llm("ollama", model="llama3.2")
    """
    backend = backend.lower().strip()

    if backend == "openai":
        return OpenAILLM(**kwargs)
    if backend == "ollama":
        return OllamaLLM(**kwargs)

    raise ValueError(
        f"Unknown backend '{backend}'. "
        f"Available options: 'openai', 'ollama'"
    )