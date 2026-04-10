"""
Legal Document Semantic Chunker

Three-layer cascaded chunking strategy:
    Layer 1: Structural boundary detection  -- hard split on article numbers (rule-based)
    Layer 2: Semantic boundary refinement   -- gamma-based dynamic split for oversized segments
    Layer 3: Length constraint enforcement  -- hard truncation as a fallback
"""

import re
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class LegalChunk:
    """
    A single chunk of a legal document, carrying structural metadata.
    """
    chunk_id:    str
    content:     str
    chunk_type:  str             # "preamble" | "article" | "clause" | "semantic" | "overflow"
    article_num: Optional[str] = None    # e.g. "第十二条"
    clause_num:  Optional[str] = None    # e.g. "第一款"
    char_count:  int           = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.content)

    def to_dict(self) -> dict:
        return {
            "chunk_id":    self.chunk_id,
            "content":     self.content,
            "chunk_type":  self.chunk_type,
            "article_num": self.article_num,
            "clause_num":  self.clause_num,
            "char_count":  self.char_count,
        }


# ─────────────────────────────────────────────
#  Layer 1: Structural boundary detection
# ─────────────────────────────────────────────

class LegalStructureParser:
    """
    Detects article and clause markers in Chinese legal documents and
    splits the text along those structural boundaries.

    Supported formats:
        - Articles : 第X条  (hard boundary, top-level)
        - Clauses  : 第X款  (soft boundary, nested inside articles)
    """

    _RE_ARTICLE = re.compile(
        r'(?:^|\n)'
        r'(第\s*[一二三四五六七八九十百千\d]+\s*条)'
        r'[\s\u3000]*'
        r'([^\n]*)',
        re.MULTILINE
    )

    _RE_CLAUSE = re.compile(
        r'(?:^|\n)'
        r'(第\s*[一二三四五六七八九十百千\d]+\s*款)'
        r'[\s\u3000]*',
        re.MULTILINE
    )

    _RE_LIST_ITEM = re.compile(
        r'(?:^|\n)'
        r'(?:\(|（)?'
        r'(\d{1,2})'
        r'(?:\)|）|\.|\、)',
        re.MULTILINE
    )

    def split_by_structure(self, text: str) -> List[Dict]:
        """
        Split text along article boundaries.

        Returns a list of dicts:
            {
                "text":        str,
                "article_num": str | None,
                "clause_num":  str | None,
                "is_preamble": bool,
            }
        """
        segments = []
        matches  = list(self._RE_ARTICLE.finditer(text))

        if not matches:
            logger.info("LegalStructureParser: no article markers found, treating as unstructured")
            return [{"text": text, "article_num": None, "clause_num": None, "is_preamble": True}]

        # Text before the first article
        preamble = text[:matches[0].start()].strip()
        if preamble:
            segments.append({
                "text":        preamble,
                "article_num": None,
                "clause_num":  None,
                "is_preamble": True,
            })

        for i, m in enumerate(matches):
            article_num  = m.group(1).replace(" ", "").replace("\u3000", "")
            start        = m.start()
            end          = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            article_text = text[start:end].strip()
            segments.extend(self._split_by_clause(article_text, article_num))

        return segments

    def _split_by_clause(self, article_text: str, article_num: str) -> List[Dict]:
        """Split an article into clauses; return the whole article if no clause markers exist."""
        clause_matches = list(self._RE_CLAUSE.finditer(article_text))

        if not clause_matches:
            return [{
                "text":        article_text,
                "article_num": article_num,
                "clause_num":  None,
                "is_preamble": False,
            }]

        segs     = []
        preamble = article_text[:clause_matches[0].start()].strip()
        if preamble:
            segs.append({
                "text":        preamble,
                "article_num": article_num,
                "clause_num":  None,
                "is_preamble": False,
            })

        for j, cm in enumerate(clause_matches):
            clause_num = cm.group(1).replace(" ", "")
            c_start    = cm.start()
            c_end      = clause_matches[j + 1].start() if j + 1 < len(clause_matches) else len(article_text)
            segs.append({
                "text":        article_text[c_start:c_end].strip(),
                "article_num": article_num,
                "clause_num":  clause_num,
                "is_preamble": False,
            })

        return segs


# ─────────────────────────────────────────────
#  Layer 2: Semantic boundary refinement
# ─────────────────────────────────────────────

class SemanticBoundaryDetector:
    """
    For segments that exceed max_chunk_length, applies the dynamic
    semantic discrepancy algorithm from the paper to find split points.

    Core formula:
        gamma_i = 1 - cosine_similarity(s_{i-1}, s_i)
        threshold = Quantile(gamma_vals, (a - p) / a)
    """

    def __init__(self, embedding_model: SentenceTransformer, max_chunk_length: int = 512):
        self._model            = embedding_model
        self._max_chunk_length = max_chunk_length

    def split_if_needed(self, text: str) -> List[str]:
        """
        Return [text] unchanged if it fits within max_chunk_length.
        Otherwise split on semantic boundaries and return sub-segments.
        """
        if len(text) <= self._max_chunk_length:
            return [text]

        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text]   # cannot split further; Layer 3 will handle it

        embeddings = self._model.encode(sentences, show_progress_bar=False)
        gamma_vals = self._compute_gamma(embeddings)
        threshold  = self._adaptive_threshold(gamma_vals, len(sentences))
        boundaries = self._detect_boundaries(gamma_vals, threshold)

        return self._assemble_chunks(sentences, boundaries)

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split on Chinese sentence-ending punctuation and newlines."""
        pattern = re.compile(r'(?<=[。；！？\n])')
        parts   = pattern.split(text)
        return [s.strip() for s in parts if len(s.strip()) > 5]

    @staticmethod
    def _compute_gamma(embeddings: np.ndarray) -> List[float]:
        """
        gamma_i = 1 - cosine_sim(s_{i-1}, s_i)
        Higher values indicate a larger semantic jump between adjacent sentences.
        """
        gammas = []
        for i in range(1, len(embeddings)):
            sim = cosine_similarity(
                embeddings[i - 1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            gammas.append(float(1.0 - sim))
        return gammas

    @staticmethod
    def _adaptive_threshold(gamma_vals: List[float], n_sentences: int) -> float:
        """
        threshold = Quantile(gamma_vals, (a - p) / a)
        where p = expected number of chunks, a = total sentence count.
        """
        if not gamma_vals:
            return 0.5
        baseline_chunks = max(1, n_sentences // 5)
        alpha           = max(0.1, (n_sentences - baseline_chunks) / n_sentences)
        return float(np.quantile(gamma_vals, alpha))

    @staticmethod
    def _detect_boundaries(gamma_vals: List[float], threshold: float) -> List[int]:
        """Return sentence indices where a boundary should be inserted."""
        boundaries = [0]
        for i, g in enumerate(gamma_vals):
            if g > threshold:
                boundaries.append(i + 1)
        boundaries.append(len(gamma_vals) + 1)
        return sorted(set(boundaries))

    @staticmethod
    def _assemble_chunks(sentences: List[str], boundaries: List[int]) -> List[str]:
        chunks = []
        for i in range(len(boundaries) - 1):
            seg = sentences[boundaries[i]: boundaries[i + 1]]
            if seg:
                chunks.append("".join(seg))
        return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────
#  Layer 3: Length constraint enforcement
# ─────────────────────────────────────────────

class LengthConstraintSplitter:
    """
    Hard truncation fallback for chunks that are still too long after
    Layer 2. Uses a greedy sentence-packing approach.
    """

    def __init__(self, max_chunk_length: int = 512, min_chunk_length: int = 30):
        self._max = max_chunk_length
        self._min = min_chunk_length

    def enforce(self, chunks: List[str]) -> List[str]:
        result = []
        for chunk in chunks:
            if len(chunk) <= self._max:
                if len(chunk) >= self._min:
                    result.append(chunk)
                # chunks shorter than min_chunk_length are treated as noise
            else:
                result.extend(self._hard_split(chunk))
        return result

    def _hard_split(self, text: str) -> List[str]:
        """Greedily pack sentences until max_chunk_length is reached."""
        sentences  = re.split(r'(?<=[。；！？\n])', text)
        sub_chunks = []
        current    = ""
        for s in sentences:
            if not s.strip():
                continue
            if len(current) + len(s) <= self._max:
                current += s
            else:
                if len(current) >= self._min:
                    sub_chunks.append(current.strip())
                current = s
        if len(current) >= self._min:
            sub_chunks.append(current.strip())
        return sub_chunks


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

class LegalSemanticChunker:
    """
    Three-layer cascaded chunker for Chinese legal documents.

    Args:
        embedding_model_name: SentenceTransformer model name.
            Recommended options:
            - "BAAI/bge-m3"                           (multilingual, best quality)
            - "shibing624/text2vec-base-chinese"       (Chinese-only)
            - "paraphrase-multilingual-MiniLM-L12-v2"  (lightweight multilingual)
        max_chunk_length: Maximum characters per chunk (default 512).
        min_chunk_length: Minimum characters per chunk; shorter chunks are
                          discarded as noise (default 30).
    """

    def __init__(
        self,
        embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        max_chunk_length:     int = 512,
        min_chunk_length:     int = 30,
    ):
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self._embed_model = SentenceTransformer(embedding_model_name)

        self._parser   = LegalStructureParser()
        self._semantic = SemanticBoundaryDetector(self._embed_model, max_chunk_length)
        self._length   = LengthConstraintSplitter(max_chunk_length, min_chunk_length)

        self._max = max_chunk_length
        self._min = min_chunk_length

    def split(self, text: str) -> List[LegalChunk]:
        """
        Split a raw legal document into LegalChunk objects.

        Pipeline:
            text
             └─ Layer 1: LegalStructureParser      → structural segments
                  └─ Layer 2: SemanticBoundaryDetector  → sub-segments (oversized only)
                       └─ Layer 3: LengthConstraintSplitter → final chunks
        """
        if not text or not text.strip():
            return []

        # Layer 1
        structural_segs = self._parser.split_by_structure(text)
        logger.info(f"Layer 1: {len(structural_segs)} structural segments")

        # (content, chunk_type, article_num, clause_num)
        raw_chunks: List[Tuple[str, str, Optional[str], Optional[str]]] = []

        for seg in structural_segs:
            seg_text    = seg["text"]
            article_num = seg["article_num"]
            clause_num  = seg["clause_num"]

            if seg["is_preamble"]:
                base_type = "preamble"
            elif clause_num:
                base_type = "clause"
            else:
                base_type = "article"

            # Layer 2
            sub_segs = self._semantic.split_if_needed(seg_text)
            for sub in sub_segs:
                chunk_type = "semantic" if len(sub_segs) > 1 else base_type
                raw_chunks.append((sub, chunk_type, article_num, clause_num))

        # Layer 3
        final_texts  = self._length.enforce([c[0] for c in raw_chunks])
        final_chunks = self._realign_metadata(raw_chunks, final_texts)

        result = []
        for idx, (content, chunk_type, article_num, clause_num) in enumerate(final_chunks):
            result.append(LegalChunk(
                chunk_id    = f"chunk-{idx + 1:03d}",
                content     = content,
                chunk_type  = chunk_type,
                article_num = article_num,
                clause_num  = clause_num,
            ))

        type_counts = {
            t: sum(1 for c in result if c.chunk_type == t)
            for t in set(c.chunk_type for c in result)
        }
        logger.info(
            f"LegalSemanticChunker: {len(text)} chars → {len(result)} chunks | "
            f"types: {type_counts}"
        )
        return result

    def split_batch(self, texts: List[str]) -> List[List[LegalChunk]]:
        """Run split() on a list of documents."""
        return [self.split(t) for t in texts]

    @staticmethod
    def _realign_metadata(
        raw:         List[Tuple[str, str, Optional[str], Optional[str]]],
        final_texts: List[str],
    ) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
        """
        Layer 3 hard-truncation produces new text fragments whose metadata
        is still in the Layer 2 results. This method re-aligns them by
        scanning forward through raw until the final text fragment is found.
        Unmatched fragments are tagged as "overflow".
        """
        result  = []
        raw_idx = 0
        for ft in final_texts:
            while raw_idx < len(raw) and ft not in raw[raw_idx][0]:
                raw_idx += 1
            if raw_idx < len(raw):
                _, chunk_type, article_num, clause_num = raw[raw_idx]
                result.append((ft, chunk_type, article_num, clause_num))
            else:
                result.append((ft, "overflow", None, None))
        return result


# ─────────────────────────────────────────────
#  Quick verification
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_contract = """
甲方（委托方）：XX科技有限公司
乙方（受托方）：YY律师事务所

本协议由甲乙双方在平等自愿的基础上签订，双方同意遵守以下条款。

第一条 委托事项
甲方委托乙方就甲方与丙方之间的合同纠纷提供法律咨询及代理服务。
乙方应在委托范围内尽职履行，不得超越授权范围自行处置案件。

第二条 委托期限
本协议自双方签字盖章之日起生效，至案件终结或双方协议解除时止。
如需延期，双方应提前十五日以书面形式确认续期事宜。

第三条 费用与支付
第一款 代理费用
甲方应向乙方支付代理费人民币叁万元整（¥30,000.00）。
上述费用于本协议签署后三个工作日内一次性付清。
第二款 差旅及其他费用
因处理委托事项产生的差旅费、诉讼费、公证费等，由甲方另行承担。
乙方应在支出后五个工作日内凭票据向甲方报销。

第四条 保密义务
双方对在合作过程中知悉的对方商业秘密及客户信息负有保密义务。
保密期限自本协议签订之日起计算，持续至信息合法公开为止，不因本协议终止而解除。
未经对方书面同意，任何一方不得将上述信息披露给第三方。

第五条 违约责任
任何一方违反本协议约定，应向守约方支付违约金，金额为协议总金额的百分之二十。
违约金不足以弥补损失的，违约方应继续赔偿实际损失。

第六条 争议解决
本协议履行过程中发生的争议，双方应首先协商解决。
协商不成的，任何一方均可向甲方所在地有管辖权的人民法院提起诉讼。

第七条 其他
本协议一式两份，甲乙双方各执一份，具有同等法律效力。
本协议未尽事宜，双方可另行签订补充协议，补充协议与本协议具有同等效力。
"""

    chunker = LegalSemanticChunker(
        embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2",
        max_chunk_length     = 300,
        min_chunk_length     = 20,
    )

    chunks = chunker.split(sample_contract)

    print(f"\n{'='*60}")
    print(f"total chunks: {len(chunks)}")
    print(f"{'='*60}\n")
    for c in chunks:
        print(f"[{c.chunk_id}] type={c.chunk_type}  article={c.article_num}  clause={c.clause_num}  chars={c.char_count}")
        print(f"  {c.content[:80]}{'...' if c.char_count > 80 else ''}")
        print()