"""The command-line endpoints, run as real subprocesses.

This is what "the endpoints work" means in CI: argparse wiring, exit codes,
sys.path setup in bin/*, and the two commands agreeing on a database file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests.helpers import TempTree

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "bin" / "sos-index"
SEARCH = REPO / "bin" / "sos-search"
SCORE = REPO / "eval" / "score.py"


def run(args, stdin: str = "", cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "SOS_FAKE_ENCODER": "1",
        "SOS_DEBOUNCE": "0.05",
        "PYTHONPATH": str(REPO),
    }
    return subprocess.run(
        [sys.executable, *map(str, args)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=cwd or REPO,
    )


class TestCli(TempTree):
    def index_once(self):
        return run([INDEX, self.root, "--db", self.db, "--once"])

    def test_index_backfills_and_exits(self) -> None:
        for n in ("cat.png", "dog.png"):
            self.write_image(n)
        p = self.index_once()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(self.db.exists())
        self.assertIn("2 embedded", p.stderr.replace("files, ", "").replace(" ", " "))

    def test_search_finds_indexed_file(self) -> None:
        self.write_image("bicycle.png")
        self.assertEqual(self.index_once().returncode, 0)
        p = run([SEARCH, "bicycle", "--db", self.db])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("bicycle.png", p.stdout)

    def test_search_timings_flag(self) -> None:
        self.write_image("cat.png")
        self.index_once()
        p = run([SEARCH, "cat", "--db", self.db, "--timings"])
        self.assertIn("encode", p.stdout)
        self.assertIn("knn", p.stdout)

    def test_search_respects_n(self) -> None:
        for i in range(5):
            self.write_image(f"i{i}.png", (i, 0, 0))
        self.index_once()
        p = run([SEARCH, "i1", "--db", self.db, "-n", "2"])
        self.assertEqual(len([ln for ln in p.stdout.splitlines() if ln.strip()]), 2)

    def test_search_without_db_fails_cleanly(self) -> None:
        p = run([SEARCH, "cat", "--db", self.root / "absent.db"])
        self.assertEqual(p.returncode, 2)
        self.assertIn("run sos-index first", p.stderr)

    def test_search_without_prompt_fails_cleanly(self) -> None:
        self.write_image("cat.png")
        self.index_once()
        p = run([SEARCH, "--db", self.db])
        self.assertEqual(p.returncode, 2)
        self.assertIn("--repl", p.stderr)

    def test_index_rejects_non_directory(self) -> None:
        p = run([INDEX, self.write_image("a.png"), "--db", self.db])
        self.assertEqual(p.returncode, 2)
        self.assertIn("not a directory", p.stderr)

    def test_repl_accepts_piped_queries(self) -> None:
        self.write_image("harbour.png")
        self.index_once()
        p = run([SEARCH, "--repl", "--db", self.db], stdin="harbour\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("harbour.png", p.stdout)

    def test_early_pipe_close_is_not_an_error(self) -> None:
        """`sos-search ... | head -1` must exit cleanly, not traceback."""
        for i in range(6):
            self.write_image(f"p{i}.png")
        self.index_once()
        cmd = f"{sys.executable} {SEARCH} p3 --db {self.db} -n 6 | head -1"
        p = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "SOS_FAKE_ENCODER": "1", "PYTHONPATH": str(REPO)},
            cwd=REPO,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("BrokenPipeError", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn("p3.png", p.stdout)

    def test_score_harness_end_to_end(self) -> None:
        self.write_image("cat.png")
        self.write_image("dog.png")
        self.index_once()
        q = self.root / "q.txt"
        q.write_text("# --- concrete object ---\n1. cat\n2. dog\n")
        p = run([SCORE, "--db", self.db, "--queries", q], stdin="y\nn\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("TOTAL", p.stdout)
        self.assertIn("1 / 2", p.stdout)
        self.assertIn("search latency p50", p.stdout)

    def test_score_refuses_empty_query_file(self) -> None:
        self.write_image("cat.png")
        self.index_once()
        q = self.root / "blank.txt"
        q.write_text("# --- concrete object ---\n1.\n2.\n")
        p = run([SCORE, "--db", self.db, "--queries", q])
        self.assertEqual(p.returncode, 2)
        self.assertIn("BEFORE searching", p.stderr)


if __name__ == "__main__":
    unittest.main()
