"""Corpus Parser and Indexer for AMD AI League Match 3 (Hostile RAG).

Handles mixed, adversarial document corpuses:
- PDFs (including encrypted/corrupted files gracefully handled)
- Source code (.py, .js, .c, .h, etc.)
- Structured text (.json, .csv, .tsv, .yaml, .md, .txt)
- Diagrams & images (.png, .jpg)
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("Match3Indexer")

class HostileCorpusIndexer:
    def __init__(self, corpus_dir: str):
        self.corpus_dir = os.path.abspath(corpus_dir)
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.entity_index: Dict[str, List[str]] = {}

    def index_all(self) -> Dict[str, Any]:
        """Indexes all readable files in the corpus directory."""
        if not os.path.exists(self.corpus_dir):
            return {"status": "empty", "total_files": 0}

        for root, dirs, files in os.walk(self.corpus_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                relpath = os.path.relpath(filepath, self.corpus_dir).replace("\\", "/")
                self._index_file(filepath, relpath)

        return {
            "status": "ready",
            "total_files": len(self.documents),
            "files": list(self.documents.keys())
        }

    def _index_file(self, filepath: str, relpath: str):
        ext = os.path.splitext(filepath)[1].lower()
        text = ""
        is_encrypted = False

        try:
            if ext == ".pdf":
                text, is_encrypted = self._parse_pdf(filepath)
            elif ext in [".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".py", ".c", ".cpp", ".h", ".log"]:
                text = self._parse_text_file(filepath)
            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                text = self._parse_image(filepath)
            else:
                # Attempt generic text decode
                text = self._parse_text_file(filepath)
        except Exception as e:
            logger.warning(f"Skipping unreadable or corrupted file {relpath}: {e}")
            text = ""

        # Record the file in documents store
        self.documents[relpath] = {
            "relpath": relpath,
            "text": text if not is_encrypted else "",
            "is_encrypted": is_encrypted,
            "is_empty": (len(text.strip()) == 0),
            "size": os.path.getsize(filepath) if os.path.exists(filepath) else 0
        }

        if text.strip() and not is_encrypted:
            self._update_entity_index(relpath, text)

    def _parse_pdf(self, filepath: str) -> tuple[str, bool]:
        """Parses a PDF safely, detecting encryption and corruption."""
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            if reader.is_encrypted:
                try:
                    # Attempt empty password decrypt
                    decrypted = reader.decrypt("")
                    if not decrypted:
                        return "", True
                except Exception:
                    return "", True

            full_text = []
            for i, page in enumerate(reader.pages):
                try:
                    extracted = page.extract_text()
                    if extracted:
                        full_text.append(extracted)
                except Exception:
                    continue
            return "\n".join(full_text), False
        except Exception as e:
            logger.debug(f"Failed to parse PDF {filepath}: {e}")
            return "", False

    def _parse_text_file(self, filepath: str) -> str:
        for enc in ["utf-8", "latin-1", "cp1252", "ascii"]:
            try:
                with open(filepath, "r", encoding=enc, errors="replace") as f:
                    return f.read()
            except Exception:
                continue
        return ""

    def _parse_image(self, filepath: str) -> str:
        """Extracts OCR text from images if an OCR engine is available."""
        # Check pytesseract or easyocr
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(filepath)
            return pytesseract.image_to_string(img)
        except Exception:
            pass

        # Try cv2 + simple OCR or basic strings
        try:
            with open(filepath, "rb") as f:
                raw = f.read()
                # Extract printable ASCII strings embedded in image
                printable = re.findall(b"[A-Za-z0-9_ -]{4,}", raw)
                return " ".join([p.decode("latin-1") for p in printable[:50]])
        except Exception:
            return ""

    def _update_entity_index(self, relpath: str, text: str):
        # Tokenize and index keywords
        tokens = set(re.findall(r"[A-Za-z0-9_\-\.]{3,}", text.lower()))
        for token in tokens:
            if token not in self.entity_index:
                self.entity_index[token] = []
            self.entity_index[token].append(relpath)

    def save(self, output_path: str):
        data = {
            "corpus_dir": self.corpus_dir,
            "documents": self.documents,
            "entity_index": self.entity_index
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, index_path: str) -> "HostileCorpusIndexer":
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        indexer = cls(data.get("corpus_dir", ""))
        indexer.documents = data.get("documents", {})
        indexer.entity_index = data.get("entity_index", {})
        return indexer
