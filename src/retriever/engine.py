"""Exact Provenance Retrieval Engine for AMD AI League Match 3.

Enforces strict cite-or-refuse policy with mathematical literal verification.
Returns {"answer": "", "citations": []} on ungrounded or adversarial questions.
"""

import re
from typing import Dict, List, Any, Optional, Tuple

class ExactHostileRetriever:
    def __init__(self, indexer):
        self.indexer = indexer

    def query(self, question: str) -> Dict[str, Any]:
        """Processes a query and returns {"answer": "...", "citations": [...]}.

        Strict Cite-or-Refuse: returns empty answer and empty citations if ungrounded.
        """
        if not question or not question.strip():
            return {"answer": "", "citations": []}

        q_clean = question.strip()
        q_tokens = self._extract_query_tokens(q_clean)
        if not q_tokens:
            return {"answer": "", "citations": []}

        # 1. Rank candidate documents
        ranked_docs = self._rank_documents(q_clean, q_tokens)
        if not ranked_docs:
            return {"answer": "", "citations": []}

        # 2. Search for answer within top ranked documents
        for doc_relpath, score in ranked_docs:
            doc_info = self.indexer.documents.get(doc_relpath)
            if not doc_info or doc_info.get("is_encrypted", False):
                continue

            text = doc_info.get("text", "")
            if not text:
                continue

            answer = self._extract_candidate_answer(q_clean, text, doc_relpath)
            if answer:
                # 3. Grounding Verification
                if self._verify_literal_provenance(answer, text):
                    # Citations must be relative paths inside the corpus
                    return {
                        "answer": self._normalize_answer(answer),
                        "citations": [doc_relpath]
                    }

        # Abstention if no document provides verifiable proof
        return {"answer": "", "citations": []}

    def _extract_query_tokens(self, query: str) -> List[str]:
        # Stopwords to discard
        stopwords = {
            "what", "is", "the", "of", "in", "and", "a", "an", "for", "to", "how",
            "much", "many", "does", "do", "did", "are", "was", "were", "where",
            "when", "which", "who", "whom", "can", "could", "should", "would",
            "at", "by", "from", "on", "with", "about", "value", "default", "named"
        }
        tokens = re.findall(r"[A-Za-z0-9_\-\.]{2,}", query.lower())
        return [t for t in tokens if t not in stopwords]

    def _rank_documents(self, query: str, tokens: List[str]) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}

        q_lower = query.lower()

        for relpath, doc in self.indexer.documents.items():
            if doc.get("is_encrypted", False):
                continue

            text = doc.get("text", "").lower()
            relpath_lower = relpath.lower()
            score = 0.0

            # Bonus for filename overlap (e.g. tq40 in tq40_datasheet)
            for token in tokens:
                if token in relpath_lower:
                    score += 5.0
                # Term frequency in text
                count = text.count(token)
                if count > 0:
                    score += min(count, 10) * 1.0

            if score > 0:
                scores[relpath] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _extract_candidate_answer(self, query: str, text: str, relpath: str) -> Optional[str]:
        """Extracts the exact atomic answer based on question intent."""
        q_lower = query.lower()

        # Intent 1: Temperature questions ("junction temperature", "operating temperature")
        if "temperature" in q_lower:
            m = re.search(r"(?:junction|operating|maximum|ambient)?\s*temperature\s*(?:of|is|:|=)?\s*([0-9]+(?:\.[0-9]+)?\s*(?:°C|C|deg C|degrees C|K|F))", text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
            # Look for number followed by C / °C near temperature
            temp_idx = text.lower().find("temperature")
            if temp_idx != -1:
                window = text[max(0, temp_idx - 50):min(len(text), temp_idx + 100)]
                m2 = re.search(r"([0-9]+(?:\.[0-9]+)?\s*(?:°C|C|K|F))", window)
                if m2:
                    return m2.group(1).strip()

        # Intent 2: Code constants / defaults in Python or config
        # e.g., "What is the DEFAULT_TIMEOUT / port / buffer_size?"
        for var_match in re.finditer(r"([A-Z0-9_]{3,})\b", query):
            var_name = var_match.group(1)
            if var_name.lower() in ["what", "does", "the"]:
                continue
            # Look for variable assignment in text
            assign_regex = rf"\b{re.escape(var_name)}\s*[:=]\s*([^\r\n;#,]+)"
            m = re.search(assign_regex, text)
            if m:
                val = m.group(1).strip().strip("'\"")
                return val

        # Intent 3: Version numbers ("What version of X ...?")
        if "version" in q_lower:
            m = re.search(r"(?:version|v)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)+(?:-[a-z0-9\.]+)?)", text, re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # Intent 4: Key-value specifications ("What is the X of Y?")
        # Extract potential property name from query (e.g., "maximum junction temperature", "clock frequency")
        lines = text.splitlines()
        q_keywords = [t for t in self._extract_query_tokens(query) if len(t) > 3]

        best_line = None
        best_overlap = 0

        for line in lines:
            line_lower = line.lower()
            overlap = sum(1 for kw in q_keywords if kw in line_lower)
            if overlap > best_overlap:
                best_overlap = overlap
                best_line = line

        if best_line and best_overlap >= 2:
            # Check for colon or equal sign key-value pairs
            if ":" in best_line:
                parts = best_line.split(":", 1)
                val = parts[1].strip()
                if val and len(val) < 60:
                    return val
            elif "=" in best_line:
                parts = best_line.split("=", 1)
                val = parts[1].strip().strip("'\"")
                if val and len(val) < 60:
                    return val

        return None

    def _verify_literal_provenance(self, candidate: str, document_text: str) -> bool:
        """Grounding proof: Candidate MUST exist verbatim within document text."""
        if not candidate:
            return False

        # Clean candidate of surrounding quotes or whitespace
        cand = candidate.strip().strip("'\"")
        if not cand:
            return False

        # Verbatim substring search
        if cand in document_text:
            return True

        # Case-insensitive substring search
        if cand.lower() in document_text.lower():
            return True

        return False

    def _normalize_answer(self, raw_answer: str) -> str:
        """Normalizes answer according to Match 3 harness expectations."""
        # Strip extraneous punctuation, conversational leads, quotes
        ans = raw_answer.strip().strip("'\"`.,;:")
        # Collapse whitespace
        ans = re.sub(r"\s+", " ", ans)
        return ans
