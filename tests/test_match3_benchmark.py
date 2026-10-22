"""Benchmark Suite for ROMA Hostile RAG Agent (AMD AI League Match 3).

Verifies accuracy, strict cite-or-refuse policy, adversarial refusal,
sub-second latency, and schema compliance.
"""

import os
import sys
import shutil
import tempfile
import time
import json
import unittest

# Ensure src in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import run_index, run_query

class TestMatch3Benchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="test_match3_")
        cls.corpus_dir = os.path.join(cls.test_dir, "corpus")
        cls.output_dir = os.path.join(cls.test_dir, "output")
        cls.index_file = os.path.join(cls.test_dir, "index.json")

        os.makedirs(cls.corpus_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)

        # 1. Document with hardware specs
        specs_dir = os.path.join(cls.corpus_dir, "specs")
        os.makedirs(specs_dir, exist_ok=True)
        with open(os.path.join(specs_dir, "tq40_datasheet_r2.txt"), "w", encoding="utf-8") as f:
            f.write(
                "AMD Instinct TQ-40 Hardware Specification\n"
                "Revision: 2.1\n"
                "Maximum Junction Temperature: 125 C\n"
                "Operating Voltage: 0.85 V\n"
                "Peak FP16 Compute: 1300 TFLOPS\n"
            )

        # 2. Python source code document
        src_dir = os.path.join(cls.corpus_dir, "src")
        os.makedirs(src_dir, exist_ok=True)
        with open(os.path.join(src_dir, "config.py"), "w", encoding="utf-8") as f:
            f.write(
                "# System configuration\n"
                "DEFAULT_TIMEOUT = 300\n"
                "MAX_RETRIES = 5\n"
                "LISTEN_PORT = 8080\n"
            )

        # 3. Empty document
        with open(os.path.join(cls.corpus_dir, "empty_file.txt"), "w", encoding="utf-8") as f:
            f.write("")

        # 4. Hostile / encrypted mock file
        enc_dir = os.path.join(cls.corpus_dir, "encrypted")
        os.makedirs(enc_dir, exist_ok=True)
        with open(os.path.join(enc_dir, "confidential.pdf"), "wb") as f:
            f.write(b"%PDF-1.4 Mock Encrypted and Unreadable Binary Content\x00\xff")

        # Run index phase
        t0 = time.perf_counter()
        cls.index_res = run_index(cls.corpus_dir, cls.index_file)
        cls.index_time = time.perf_counter() - t0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_index_completeness(self):
        self.assertGreaterEqual(self.index_res["total_files"], 3)
        self.assertTrue(os.path.exists(self.index_file))

    def test_02_hardware_spec_query(self):
        t0 = time.perf_counter()
        res = run_query(
            corpus_dir=self.corpus_dir,
            query_id="q01",
            query_text="What is the maximum junction temperature of the TQ-40?",
            output_dir=self.output_dir,
            index_file=self.index_file
        )
        latency = time.perf_counter() - t0

        self.assertIn("125 C", res["answer"])
        self.assertEqual(res["citations"], ["specs/tq40_datasheet_r2.txt"])
        self.assertLess(latency, 1.0, f"Latency too high: {latency:.3f}s")
        print(f"[TEST PASS] Hardware spec answered in {latency:.4f}s: {res}")

    def test_03_code_constant_query(self):
        t0 = time.perf_counter()
        res = run_query(
            corpus_dir=self.corpus_dir,
            query_id="q02",
            query_text="What is the DEFAULT_TIMEOUT in config.py?",
            output_dir=self.output_dir,
            index_file=self.index_file
        )
        latency = time.perf_counter() - t0

        self.assertEqual(res["answer"], "300")
        self.assertEqual(res["citations"], ["src/config.py"])
        self.assertLess(latency, 1.0)
        print(f"[TEST PASS] Code constant answered in {latency:.4f}s: {res}")

    def test_04_adversarial_missing_info_refusal(self):
        """Must return empty answer and empty citations when not in corpus."""
        t0 = time.perf_counter()
        res = run_query(
            corpus_dir=self.corpus_dir,
            query_id="q03",
            query_text="What is the CEO personal phone number?",
            output_dir=self.output_dir,
            index_file=self.index_file
        )
        latency = time.perf_counter() - t0

        self.assertEqual(res["answer"], "")
        self.assertEqual(res["citations"], [])
        self.assertLess(latency, 1.0)
        print(f"[TEST PASS] Adversarial missing question refused in {latency:.4f}s: {res}")

    def test_05_encrypted_file_refusal(self):
        """Must refuse to answer from corrupted or encrypted documents."""
        t0 = time.perf_counter()
        res = run_query(
            corpus_dir=self.corpus_dir,
            query_id="q04",
            query_text="What is the confidential roadmap password in confidential.pdf?",
            output_dir=self.output_dir,
            index_file=self.index_file
        )
        latency = time.perf_counter() - t0

        self.assertEqual(res["answer"], "")
        self.assertEqual(res["citations"], [])
        print(f"[TEST PASS] Encrypted document question refused in {latency:.4f}s: {res}")

if __name__ == "__main__":
    unittest.main()
