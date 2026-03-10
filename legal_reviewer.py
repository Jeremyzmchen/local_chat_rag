"""
Legal Reviewer
    Two-stage document review pipeline.

    ErrorAnalyzer     -- locates deviations between the document and reference knowledge
    RevisionGenerator -- produces revision suggestions based on the deviation analysis,
                         along with a confidence score
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Output of ErrorAnalyzer.analyze().

    Fields:
        query:              the review question that triggered this analysis
        dimension:          review dimension (legality / equality / completeness / risk)
        analysis:           free-text deviation analysis produced by the LLM
        error_regions:      list of problematic text spans identified in the document
        reference_support:  list of reference chunk contents used during analysis
    """
    query:             str
    dimension:         str
    analysis:          str
    error_regions:     List[str]
    reference_support: List[str]


@dataclass
class RevisionResult:
    """
    Output of RevisionGenerator.generate().

    Fields:
        original_text:        the document fragment that was reviewed
        revision_suggestions: free-text revision advice produced by the LLM
        modified_regions:     the error regions carried over from AnalysisResult
        confidence:           score in [0, 1] estimating reliability of this result
        analysis_summary:     condensed version of the deviation analysis
    """
    original_text:        str
    revision_suggestions: str
    modified_regions:     List[str]
    confidence:           float
    analysis_summary:     str


# ─────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
You are a senior legal review expert. Based on the reference provisions below, \
identify any deviations in the document under review.

[Reference provisions]
{references}

[Document under review]
{document}

[Review question]
{query}

Requirements:
- Base your analysis strictly on the reference provisions; do not use outside knowledge.
- For each deviation found, quote the exact problematic text from the document.
- If the document is consistent with the reference provisions, state "No deviation found."
- Structure your output as follows:

Deviation analysis:
<your analysis here>

Problematic text regions (one per line, quote directly from the document):
<region 1>
<region 2>"""

_REVISION_PROMPT = """\
You are a senior legal review expert. Based on the deviation analysis and reference \
provisions below, provide specific revision suggestions for the document.

[Document under review]
{document}

[Deviation analysis]
{analysis}

[Reference provisions]
{references}

Requirements:
- Adhere strictly to the reference provisions; do not introduce outside knowledge.
- For each problematic region, provide a concrete revised version.
- If no revision is needed, state "This clause complies with the provisions."
- Structure your output as follows:

Revision suggestions:
<your suggestions here>"""


# ─────────────────────────────────────────────
#  ErrorAnalyzer
# ─────────────────────────────────────────────

class ErrorAnalyzer:
    """
    Locates deviations between a document fragment and retrieved reference knowledge.

    Args:
        llm_fn: callable that accepts a prompt string and returns a response string.
    """

    def __init__(self, llm_fn):
        self._llm = llm_fn

    def analyze(
        self,
        document_chunk:    str,
        query:             str,
        dimension:         str,
        retrieved_chunks:  List[str],
    ) -> AnalysisResult:
        """
        Run deviation analysis for one review question.

        Args:
            document_chunk:   text of the document fragment under review
            query:            the review question
            dimension:        review dimension label
            retrieved_chunks: reference content retrieved from the knowledge base

        Returns:
            AnalysisResult with analysis text and extracted error regions
        """
        references = self._format_references(retrieved_chunks)
        prompt     = _ANALYSIS_PROMPT.format(
            references = references,
            document   = document_chunk,
            query      = query,
        )

        try:
            raw = self._llm(prompt).strip()
        except Exception as e:
            logger.error(f"ErrorAnalyzer: LLM call failed ({e})")
            return AnalysisResult(
                query             = query,
                dimension         = dimension,
                analysis          = "",
                error_regions     = [],
                reference_support = retrieved_chunks,
            )

        analysis, error_regions = self._parse(raw)

        logger.info(
            f"ErrorAnalyzer: [{dimension}] {len(error_regions)} region(s) identified"
        )
        return AnalysisResult(
            query             = query,
            dimension         = dimension,
            analysis          = analysis,
            error_regions     = error_regions,
            reference_support = retrieved_chunks,
        )

    @staticmethod
    def _format_references(chunks: List[str]) -> str:
        if not chunks:
            return "(no reference content)"
        return "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))

    @staticmethod
    def _parse(raw: str) -> tuple:
        """
        Split LLM output into (analysis_text, error_regions).
        Looks for the 'Problematic text regions' section as the separator.
        """
        marker = "Problematic text regions"
        if marker in raw:
            parts          = raw.split(marker, 1)
            analysis_text  = parts[0].replace("Deviation analysis:", "").strip()
            regions_block  = parts[1].strip().lstrip(":")
            error_regions  = [
                line.strip()
                for line in regions_block.splitlines()
                if line.strip() and line.strip() != "No deviation found."
            ]
        else:
            analysis_text = raw.replace("Deviation analysis:", "").strip()
            error_regions = []

        return analysis_text, error_regions


# ─────────────────────────────────────────────
#  RevisionGenerator
# ─────────────────────────────────────────────

class RevisionGenerator:
    """
    Produces revision suggestions from an AnalysisResult, together with
    a confidence score that reflects how well-supported the result is.

    Args:
        llm_fn: callable that accepts a prompt string and returns a response string.
    """

    def __init__(self, llm_fn):
        self._llm = llm_fn

    def generate(
        self,
        document_chunk:  str,
        analysis_result: AnalysisResult,
    ) -> RevisionResult:
        """
        Generate revision suggestions for a document fragment.

        Args:
            document_chunk:  text of the document fragment under review
            analysis_result: AnalysisResult produced by ErrorAnalyzer.analyze()

        Returns:
            RevisionResult with suggestions, confidence score, and analysis summary
        """
        references = ErrorAnalyzer._format_references(analysis_result.reference_support)
        prompt     = _REVISION_PROMPT.format(
            document   = document_chunk,
            analysis   = analysis_result.analysis,
            references = references,
        )

        try:
            raw = self._llm(prompt).strip()
            suggestions = raw.replace("Revision suggestions:", "").strip()
        except Exception as e:
            logger.error(f"RevisionGenerator: LLM call failed ({e})")
            suggestions = ""

        confidence = self._calculate_confidence(analysis_result)

        logger.info(
            f"RevisionGenerator: [{analysis_result.dimension}] "
            f"confidence={confidence:.2f}"
        )
        return RevisionResult(
            original_text        = document_chunk,
            revision_suggestions = suggestions,
            modified_regions     = analysis_result.error_regions,
            confidence           = confidence,
            analysis_summary     = analysis_result.analysis[:200].strip(),
        )

    @staticmethod
    def _calculate_confidence(analysis: AnalysisResult) -> float:
        """
        Confidence score in [0, 1].

        Scoring logic:
            base score      = 0.5
            +0.1 per reference chunk  (capped contribution: up to +0.3)
            +0.05 per error region    (capped contribution: up to +0.2)
            hard ceiling at 0.9       (full certainty is never claimed)

        A result with no references and no identified errors scores 0.5,
        signalling that human review is advisable.
        """
        ref_count   = len(analysis.reference_support)
        error_count = len(analysis.error_regions)

        ref_contrib   = min(ref_count * 0.1,   0.3)
        error_contrib = min(error_count * 0.05, 0.2)

        return round(min(0.5 + ref_contrib + error_contrib, 0.9), 2)