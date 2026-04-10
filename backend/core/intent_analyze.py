"""
Intent Analyze
    - search scope
    - intent analyzer
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchScope:
    """
    Defines the scope of search for intent analysis.
    """
    target_doc_id: Optional[str] = None
    target_doc_type: Optional[str] = None
    target_doc_source: Optional[str] = None
    is_global: bool = True

    def is_constrained(self) -> bool:
        """
        Returns true if the search scope is constrained.
        """
        return any([
            self.target_doc_id,
            self.target_doc_type,
            self.target_doc_source,
        ])
    

class IntentAnalyzer:
    """
    Analyzes user input to determine search intent and scope.
        - phase 1: file_name match
        - phase 2: LLM structured understanding of user input
    """
    
    # some examples
    _LLM_TRIGGER_KEYWORDS = [
        "file", "folder", "latest", "recent", 
    ]

    _LLM_PROMPT_TEMPLATE = """\
You are a file intent analysis assistant. \
Given the user's question and the file list below, \
determine which files the user wants to query.

[File List]
{file_list}

[User Question]
{question}

[Output Format]
Return ONLY a single JSON object with the following fields \
(use null for fields that do not apply):
{{
  "target_file": "relative path of the specific file, e.g. finance/report.pdf, or null",
  "directory":   "directory name, e.g. finance, or null",
  "file_type":   "one of: pdf / docx / md / xlsx / txt / png, or null"
}}

Do not include any explanation. Output JSON only."""

    def __init__(self, all_files: List[str], llm_fn=None,):
        self._all_files = all_files
        self._llm_fn    = llm_fn

    def _should_call_llm(self, question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in self._LLM_TRIGGER_KEYWORDS)

    def analyze(self, question: str) -> SearchScope:
        """
        Analyze the question and return a SearchScope.
        Phases are tried in order; the first hit short-circuits the rest.
        """
        q = question.strip()

        # Phase 0: direct filename match (fastest)
        scope = self._filename_match(q)
        if scope.is_constrained():
            logger.info(f"Intent resolved at Phase 0: {scope}")
            return scope

        # Phase 1: LLM analysis
        if self._llm_fn and self._should_call_llm(q):
            scope = self._call_llm(q)
            if scope.is_constrained():
                logger.info(f"Intent resolved at Phase 1: {scope}")
                return scope

        logger.info("No constraints detected — falling back to global search")
        return SearchScope()
    
    # phase 1
    def _filename_match(self, question: str) -> SearchScope:
        """Return a scope if the question directly contains a filename or path."""
        q = question.lower()
        for file_path in self._all_files:
            filename = Path(file_path).name.lower()
            if file_path.lower() in q or filename in q:
                return SearchScope(
                    target_doc_id = file_path,
                    is_global     = False,
                )
        return SearchScope()
    
    # phase 2
    def _call_llm(self, question: str) -> SearchScope:
        """Call the LLM to analyze the question."""
        # only extract the first 50 files to keep the prompt short
        file_list = "\n".join(f"- {f}" for f in self._all_files[:50])
        prompt = self._LLM_PROMPT_TEMPLATE.format(
            file_list=file_list,
            question=question,
        )

        try:
            raw   = self._llm_fn(prompt).strip()
            result = self._parse_llm_json(raw)
            return self._build_scope_from_llm(result)
        except Exception as e:
            logger.warning(f"Phase 2 LLM call failed, falling back to global search: {e}")
            return SearchScope()
        
    def _parse_llm_json(self, raw: str) -> dict:
        """
        Parse the JSON returned by the LLM.
        Handles the common case where the model wraps output in a markdown
        code block:  ```json ... ```
        """
        if "```" in raw:
            parts = raw.split("```")
            raw   = parts[1].removeprefix("json").strip() if len(parts) > 1 else raw
        return json.loads(raw)
    
    def _build_scope_from_llm(self, result: dict) -> SearchScope:
        """Convert the LLM's parsed dict into a SearchScope."""
        target_file = result.get("target_file")
        directory   = result.get("directory")
        file_type   = result.get("file_type")

        # fuzzy match
        if target_file and target_file not in self._all_files:
            matched = None
            for f in self._all_files:
                if target_file.lower() in f.lower():
                    matched = f
                    break
            target_file = matched

        if any([target_file, directory, file_type]):
            return SearchScope(
                target_doc_id = target_file,
                target_doc_source = directory,
                target_doc_type = file_type,
                is_global = False,
            )
        return SearchScope()
    




