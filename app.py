#!/usr/bin/env python3
"""AMD AI League Match 3 (Hostile RAG Retrieval) Application Entrypoint.

Invocation contract:
1. Indexing (before evaluation clock):
   python3 /app/app.py --index /app/corpus
2. Querying (per question during evaluation):
   python3 /app/app.py --corpus /app/corpus --query-id <query-id> --query "<question>"

Output:
   /app/output/<query-id>_output.json
"""

import sys
import os
import json
import argparse
import tempfile
import logging

# Add src to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.indexer.corpus_parser import HostileCorpusIndexer
from src.retriever.engine import ExactHostileRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Match3App")

DEFAULT_INDEX_PATH = os.path.join(current_dir, "index.json")

def atomic_write_json(file_path: str, data: dict):
    """Crash-proof atomic JSON writer via temporary file replacement."""
    target_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(target_dir, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="match3_atomic_", suffix=".tmp")
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, file_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def run_index(corpus_dir: str, index_file: str = DEFAULT_INDEX_PATH):
    logger.info(f"Indexing corpus at: {corpus_dir}")
    indexer = HostileCorpusIndexer(corpus_dir)
    res = indexer.index_all()
    indexer.save(index_file)
    logger.info(f"Index complete: {res['total_files']} files indexed -> {index_file}")
    return res

def run_query(corpus_dir: str, query_id: str, query_text: str, output_dir: str = "/app/output", index_file: str = DEFAULT_INDEX_PATH):
    # Load index or build on the fly if missing
    if os.path.exists(index_file):
        indexer = HostileCorpusIndexer.load(index_file)
    else:
        logger.warning(f"Index file {index_file} not found. Indexing on the fly...")
        indexer = HostileCorpusIndexer(corpus_dir)
        indexer.index_all()

    retriever = ExactHostileRetriever(indexer)
    result = retriever.query(query_text)

    # Ensure required fields
    clean_result = {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", [])
    }

    output_path = os.path.join(output_dir, f"{query_id}_output.json")
    atomic_write_json(output_path, clean_result)
    logger.info(f"Answer written to {output_path}: {clean_result}")
    return clean_result

def parse_args():
    parser = argparse.ArgumentParser(description="ROMA Hostile RAG Agent for AMD AI League Match 3")
    parser.add_argument("--index", type=str, help="Corpus directory to index")
    parser.add_argument("--corpus", type=str, default="/app/corpus", help="Corpus directory path")
    parser.add_argument("--query-id", type=str, help="Query ID for tracking")
    parser.add_argument("--query", type=str, help="Question to answer")
    parser.add_argument("--output-dir", type=str, default="/app/output", help="Directory where results are written")
    parser.add_argument("--index-file", type=str, default=DEFAULT_INDEX_PATH, help="Path to index store")

    return parser.parse_args()

def main():
    args = parse_args()

    # Mode 1: Indexing phase
    if args.index:
        run_index(args.index, args.index_file)
        sys.exit(0)

    # Mode 2: Querying phase
    if args.query and args.query_id:
        run_query(
            corpus_dir=args.corpus,
            query_id=args.query_id,
            query_text=args.query,
            output_dir=args.output_dir,
            index_file=args.index_file
        )
        sys.exit(0)

    logger.error("Invalid arguments: must provide either --index <dir> or (--query-id <id> and --query '<text>')")
    parser = argparse.ArgumentParser()
    parser.print_help()
    sys.exit(1)

if __name__ == "__main__":
    main()
