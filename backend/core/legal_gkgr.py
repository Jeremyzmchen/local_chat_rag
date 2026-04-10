"""
Legal GKGR
    Generative Knowledge-Guided Retrieval framework for legal documents.

    Three core modules:
        LegalKeyInfoExtractor     -- three-level legal key information extraction
        LegalKnowledgeScorer      -- knowledge-level score computation (Phi_K)
        LegalReviewQueryGenerator -- two-phase review query generation
"""

import re
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class KeyInfo:
    """Three-level key information extraction result."""
    max_terms: List[str]   # highest priority: core legal relationship / dispute focus
    mid_terms: List[str]   # medium priority:  limiting conditions / applicability premise
    lit_terms: List[str]   # literal priority: specific values / referenced provisions

    # Term weights as defined in the paper
    WEIGHTS = {"max": 0.5, "mid": 0.3, "lit": 0.2}


@dataclass
class ReviewQuery:
    """A single generated review question."""
    query:     str
    dimension: str   # "legality" | "equality" | "completeness" | "risk"
    priority:  int   # 1 (highest) ~ 5 (lowest)


# ─────────────────────────────────────────────
#  LegalKeyInfoExtractor
# ─────────────────────────────────────────────

_KEY_INFO_PROMPT = """\
You are a legal document analysis expert. Extract three priority levels of key information from the legal query below.

Highest priority (max): core legal relationships or dispute focus
    e.g. breach of contract, contract termination, confidentiality obligation, damages

Medium priority (mid): limiting conditions or applicability premises
    e.g. under force majeure, written notice 30 days in advance, by mutual agreement

Literal priority (lit): specific values, deadlines, or referenced provisions
    e.g. penalty not exceeding 20%, within three business days, pursuant to Article X

[Query]
{query}

Output strictly in the following format, one level per line, keywords separated by commas, no explanation:
max: keyword1, keyword2
mid: keyword1, keyword2
lit: keyword1, keyword2"""


class LegalKeyInfoExtractor:
    """
    Extracts three-level key information from a legal query.

    Args:
        llm_fn: callable that accepts a prompt string and returns a response string.
                Pass the RAGPipeline LLM instance directly (BaseLLM implements __call__).
    """

    def __init__(self, llm_fn):
        self._llm = llm_fn

    def extract(self, query: str) -> KeyInfo:
        prompt = _KEY_INFO_PROMPT.format(query=query)
        try:
            raw = self._llm(prompt).strip()
            return self._parse(raw)
        except Exception as e:
            logger.warning(f"LegalKeyInfoExtractor: LLM call failed ({e}), returning empty KeyInfo")
            return KeyInfo(max_terms=[], mid_terms=[], lit_terms=[])

    @staticmethod
    def _parse(raw: str) -> KeyInfo:
        """Parse the three-line format returned by the LLM."""
        result = {"max": [], "mid": [], "lit": []}
        for line in raw.splitlines():
            for level in ("max", "mid", "lit"):
                if line.lower().startswith(f"{level}:"):
                    terms_str = line[len(level) + 1:].strip()
                    terms = [t.strip() for t in re.split(r"[,，]", terms_str) if t.strip()]
                    result[level] = terms
        return KeyInfo(
            max_terms=result["max"],
            mid_terms=result["mid"],
            lit_terms=result["lit"],
        )


# ─────────────────────────────────────────────
#  LegalKnowledgeScorer
#  Implements Phi_K from the paper
# ─────────────────────────────────────────────

class LegalKnowledgeScorer:
    """
    Computes knowledge-level score Phi_K for each chunk in the knowledge base.

    Formula:
        Phi_K(chunk) = sum( Sign(t, chunk) * Rarity(t) * (1 + CI(t, chunk)) * weight(level) )

    Where:
        Sign(t, chunk) = 2 * f(t, chunk) * Lambda_DL / (f(t, chunk) + 1)   # term significance
        Rarity(t)      = log((D - df(t) + 0.5) / (df(t) + 0.5) + 1)        # term rarity
        CI(t, chunk)   = max_window( count(t, w) * |w| / |chunk| )           # coherence index
        Lambda_DL      = (avg_len + chunk_len) / (2 * avg_len)               # length adaptive factor
    """

    def __init__(self, window_size: int = 50):
        self._window_size = window_size

    def score_chunks(
        self,
        key_info: KeyInfo,
        chunks:   List[str],
    ) -> List[float]:
        """
        Compute Phi_K scores for a list of chunks.

        Args:
            key_info: KeyInfo extracted by LegalKeyInfoExtractor
            chunks:   list of text chunks to score

        Returns:
            List of floats, same length as chunks, normalized to [0, 1]
        """
        if not chunks or not any([key_info.max_terms, key_info.mid_terms, key_info.lit_terms]):
            return [0.0] * len(chunks)

        # precompute token lists and average length for Lambda_DL
        tokenized = [self._tokenize(c) for c in chunks]
        avg_len   = max(1, sum(len(t) for t in tokenized) / len(tokenized))

        # precompute document frequency per term for Rarity
        all_terms = (
            [(t, "max") for t in key_info.max_terms] +
            [(t, "mid") for t in key_info.mid_terms] +
            [(t, "lit") for t in key_info.lit_terms]
        )
        df_map = {
            term: sum(1 for tokens in tokenized if term in tokens)
            for term, _ in all_terms
        }

        scores = []
        for tokens, chunk in zip(tokenized, chunks):
            chunk_len   = max(1, len(tokens))
            lambda_dl   = (avg_len + chunk_len) / (2 * avg_len)
            chunk_score = 0.0

            for term, level in all_terms:
                if term not in tokens:
                    continue

                weight = KeyInfo.WEIGHTS[level]
                tf     = tokens.count(term)
                sign   = (2 * tf * lambda_dl) / (tf + 1)
                rarity = self._rarity(df_map[term], len(chunks))
                ci     = self._coherence_index(term, tokens)

                chunk_score += sign * rarity * (1 + ci) * weight

            scores.append(chunk_score)

        # normalize to [0, 1]
        max_score = max(scores) if scores else 1.0
        if max_score > 0:
            scores = [s / max_score for s in scores]

        return scores

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Character-level tokenization for Chinese, word-level for English.
        Avoids a jieba dependency to keep things lightweight.
        """
        tokens = [ch for ch in text if '\u4e00' <= ch <= '\u9fff']
        tokens += re.findall(r'[a-zA-Z0-9]+', text.lower())
        return tokens

    @staticmethod
    def _rarity(doc_freq: int, total_docs: int) -> float:
        """Rarity(t) = log((D - df + 0.5) / (df + 0.5) + 1)"""
        return float(np.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1))

    def _coherence_index(self, term: str, tokens: List[str]) -> float:
        """
        CI(t, chunk) = max over sliding windows of ( count(t, w) * |w| / |chunk| )
        Sliding window step is fixed at 10.
        """
        chunk_len = len(tokens)
        if chunk_len == 0:
            return 0.0

        ws     = min(self._window_size, chunk_len)
        max_ci = 0.0
        for i in range(0, chunk_len - ws + 1, 10):
            window = tokens[i: i + ws]
            cnt    = window.count(term)
            if cnt > 0:
                ci     = (cnt * ws) / chunk_len
                max_ci = max(max_ci, ci)
        return max_ci


# ─────────────────────────────────────────────
#  LegalReviewQueryGenerator
#  Two-phase review query generation
# ─────────────────────────────────────────────

_CORE_COMPONENTS_PROMPT = """\
You are a senior legal review expert. Perform a structured analysis of the following legal document fragment and extract its core elements.

Elements to extract (not limited to):
- Core legal relationships (contracting parties, rights and obligations)
- Key clauses (payment terms, breach liability, confidentiality, etc.)
- Important limiting conditions (deadlines, amounts, applicability premises)
- Potential risk points (vague language, missing clauses, unequal terms)

[Document fragment under review]
{document_chunk}

Output the core elements in a structured format, one category per line:"""

_REVIEW_QUERIES_PROMPT = """\
You are a senior legal review expert. Based on the document and its core elements below, generate review questions across four dimensions.

Review dimensions:
  Legality     -- whether clauses comply with mandatory legal provisions; whether any clause is void
  Equality     -- whether rights and obligations are balanced; whether any clause is grossly unfair
  Completeness -- whether necessary clauses are missing (e.g. dispute resolution, force majeure, confidentiality)
  Risk         -- whether there is vague language, ambiguous clauses, or latent risks

[Document under review]
{document_chunk}

[Core elements]
{core_components}

Generate 5 review questions. Output strictly in the following format (one per line, dimension in parentheses):
'question text'(dimension)

Examples:
'Does the penalty rate exceed the statutory cap of 30%'(Legality)
'Are the conditions for Party A to unilaterally terminate too broad'(Equality)"""


class LegalReviewQueryGenerator:
    """
    Two-phase review query generator.

    Phase 1: extract core elements from the document (targeting step)
    Phase 2: generate review questions across Legality / Equality /
             Completeness / Risk dimensions based on those elements

    Args:
        llm_fn: callable that accepts a prompt string and returns a response string.
    """

    _DIMENSION_MAP = {
        "Legality":     "legality",
        "Equality":     "equality",
        "Completeness": "completeness",
        "Risk":         "risk",
        # Chinese fallbacks for mixed-language LLM output
        "合法性": "legality",
        "对等性": "equality",
        "完整性": "completeness",
        "风险性": "risk",
    }

    def __init__(self, llm_fn):
        self._llm = llm_fn

    def generate(self, document_chunk: str, max_queries: int = 5) -> List[ReviewQuery]:
        """
        Generate a list of review questions for the given document fragment.

        Args:
            document_chunk: text of the document to review
            max_queries:    maximum number of questions to return

        Returns:
            List[ReviewQuery] sorted by priority
        """
        core_components = self._extract_core_components(document_chunk)
        logger.info("LegalReviewQueryGenerator: core components extracted")

        queries = self._generate_queries(document_chunk, core_components, max_queries)
        logger.info(f"LegalReviewQueryGenerator: {len(queries)} queries generated")

        return queries

    def get_query_strings(self, document_chunk: str, max_queries: int = 5) -> List[str]:
        """Convenience method: return plain query strings for use in RAGPipeline.query()."""
        return [q.query for q in self.generate(document_chunk, max_queries)]

    def _extract_core_components(self, document_chunk: str) -> str:
        prompt = _CORE_COMPONENTS_PROMPT.format(document_chunk=document_chunk)
        try:
            return self._llm(prompt).strip()
        except Exception as e:
            logger.warning(f"Phase 1 failed: {e}")
            return ""

    def _generate_queries(
        self,
        document_chunk:  str,
        core_components: str,
        max_queries:     int,
    ) -> List[ReviewQuery]:
        prompt = _REVIEW_QUERIES_PROMPT.format(
            document_chunk  = document_chunk,
            core_components = core_components,
        )
        try:
            raw = self._llm(prompt).strip()
            return self._parse_queries(raw, max_queries)
        except Exception as e:
            logger.warning(f"Phase 2 failed: {e}")
            return []

    def _parse_queries(self, raw: str, max_queries: int) -> List[ReviewQuery]:
        """
        Parse LLM output in the format: 'question text'(dimension)
        Priority order: Legality > Equality > Completeness > Risk
        """
        priority_order = ["Legality", "Equality", "Completeness", "Risk"]
        pattern        = re.compile(r"'([^']+)'\s*[（(]([^）)]+)[）)]")
        queries        = []

        for m in pattern.finditer(raw):
            query_text  = m.group(1).strip()
            raw_dim     = m.group(2).strip()
            matched_dim = self._match_dimension(raw_dim)
            # map to canonical English name for priority lookup
            canonical   = next(
                (k for k, v in self._DIMENSION_MAP.items() if v == matched_dim and k in priority_order),
                None
            )
            priority = priority_order.index(canonical) + 1 if canonical in priority_order else 5

            queries.append(ReviewQuery(
                query     = query_text,
                dimension = matched_dim or raw_dim,
                priority  = priority,
            ))

        queries.sort(key=lambda q: q.priority)
        return queries[:max_queries]

    def _match_dimension(self, raw_dim: str) -> str:
        """Fuzzy-match a raw dimension string to a canonical value."""
        for key, val in self._DIMENSION_MAP.items():
            if key.lower() in raw_dim.lower() or raw_dim.lower() in key.lower():
                return val
        return raw_dim


# ─────────────────────────────────────────────
#  Quick verification
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class MockLLM:
        def __call__(self, prompt: str) -> str:
            if "core elements" in prompt.lower() or "structured analysis" in prompt.lower():
                return (
                    "Core legal relationship: agency\n"
                    "Key clauses: agency fee CNY 30,000, confidentiality obligation\n"
                    "Risk points: penalty rate has no explicit upper limit"
                )
            if "review questions" in prompt.lower() or "generate 5" in prompt.lower():
                return (
                    "'Does the penalty rate exceed the statutory cap of 30%'(Legality)\n"
                    "'Are the conditions for Party A to unilaterally terminate too broad'(Equality)\n"
                    "'Is a force majeure clause missing'(Completeness)\n"
                    "'Is the confidentiality period description ambiguous'(Risk)\n"
                    "'Is there a cap on travel expense reimbursement'(Equality)"
                )
            # KeyInfo extraction fallback
            return "max: breach of contract, contract termination\nmid: advance notice, written form\nlit: 30 days, 20%"

    mock_llm = MockLLM()

    sample = """
Article 5  Breach Liability
If either party breaches this agreement, it shall pay the non-breaching party
a penalty equal to 20% of the total contract value.
If the penalty is insufficient to cover actual losses, the breaching party
shall continue to compensate for the remaining loss.
"""

    # verify LegalKeyInfoExtractor
    extractor = LegalKeyInfoExtractor(mock_llm)
    key_info  = extractor.extract("Does the penalty clause comply with legal requirements?")
    print("=== KeyInfo ===")
    print(f"  max: {key_info.max_terms}")
    print(f"  mid: {key_info.mid_terms}")
    print(f"  lit: {key_info.lit_terms}")

    # verify LegalKnowledgeScorer
    scorer = LegalKnowledgeScorer()
    chunks = [sample, "Article 1  Scope of Engagement\nParty A engages Party B to provide legal consulting services."]
    scores = scorer.score_chunks(key_info, chunks)
    print("\n=== KnowledgeScorer ===")
    for i, s in enumerate(scores):
        print(f"  chunk-{i+1}: {s:.4f}")

    # verify LegalReviewQueryGenerator
    generator = LegalReviewQueryGenerator(mock_llm)
    queries   = generator.generate(sample)
    print("\n=== ReviewQueries ===")
    for q in queries:
        print(f"  [{q.dimension}] P{q.priority}: {q.query}")