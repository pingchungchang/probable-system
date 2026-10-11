"""Lexical retrieval: tokenisation, BM25, and RRF fusion."""

from __future__ import annotations

import unittest

from sospoc.search import RRF_K, rrf
from sospoc.store import (
    Store,
    _split_extras,
    _variants,
    digest_of,
    fts_expression,
    path_tokens,
)
from tests.helpers import TempTree


class TestTokenisation(unittest.TestCase):
    def test_splits_separators_and_camel_case(self) -> None:
        self.assertIn("screenshot", _variants("Screenshot_2026-10-11_code.jpg"))
        self.assertIn("code", _variants("Screenshot_2026-10-11_code.jpg"))
        self.assertIn("2026", _variants("Screenshot_2026-10-11_code.jpg"))

    def test_keeps_whole_word_as_well_as_pieces(self) -> None:
        v = _variants("VectorIndex")
        self.assertIn("vectorindex", v)
        self.assertIn("vector", v)
        self.assertIn("index", v)

    def test_deduplicates_preserving_order(self) -> None:
        self.assertEqual(_variants("code code CODE"), ["code"])

    def test_lowercases(self) -> None:
        self.assertEqual(_variants("CODE"), ["code"])


class TestFtsExpression(unittest.TestCase):
    def test_or_operator_stays_uppercase(self) -> None:
        """Regression: lowercasing the joined expression broke every phrase.

        FTS5 operators are case-sensitive. A lowercase `or` is a bareword, so
        the query became an AND that also required the literal token "or" —
        matching nothing, silently.
        """
        expr = fts_expression("reciprocal rank fusion")
        self.assertIn(" OR ", expr)
        self.assertNotIn(" or ", expr)

    def test_every_token_is_quoted(self) -> None:
        for part in fts_expression("alpha beta").split(" OR "):
            self.assertTrue(part.startswith('"') and part.endswith('"'), part)

    def test_user_operators_are_data_not_syntax(self) -> None:
        for hostile in ("AND OR NOT NEAR", '"', "a*", "(b)", "x AND y"):
            expr = fts_expression(hostile)
            self.assertNotIn('""', expr)

    def test_empty_query(self) -> None:
        self.assertEqual(fts_expression("   "), "")
        self.assertEqual(fts_expression("!!!"), "")


class TestBm25(TempTree):
    def setUp(self) -> None:
        super().setUp()
        self.store = Store(self.db)
        self.ids = {}
        for name, text in [
            ("code.png", "def reciprocal_rank_fusion(rankings, k=60): pass"),
            ("terminal.png", "$ ls -la /var/log && tail -f syslog"),
            ("field.png", ""),
        ]:
            p = self.write_image(name)
            fid, _ = self.store.upsert(p, "image/png", digest_of(p))
            self.store.put_text(fid, text, path_tokens(p, self.root))
            self.ids[name] = fid

    def test_matches_ocr_content(self) -> None:
        hits = self.store.bm25("syslog", 10, "ocr_text")
        self.assertEqual(hits[0][0], self.ids["terminal.png"])

    def test_matches_multi_word_phrase(self) -> None:
        hits = self.store.bm25("reciprocal rank fusion", 10, "ocr_text")
        self.assertTrue(hits, "multi-word query returned nothing")
        self.assertEqual(hits[0][0], self.ids["code.png"])

    def test_matches_filename_even_without_ocr(self) -> None:
        hits = self.store.bm25("field", 10, "path_tokens")
        self.assertIn(self.ids["field.png"], [h for h, _ in hits])

    def test_unknown_term_returns_nothing(self) -> None:
        self.assertEqual(self.store.bm25("zzzznotpresent", 10, "ocr_text"), [])

    def test_empty_query_returns_nothing(self) -> None:
        self.assertEqual(self.store.bm25("", 10, "ocr_text"), [])

    def test_respects_limit(self) -> None:
        self.assertLessEqual(len(self.store.bm25("png", 2, "path_tokens")), 2)

    def test_forget_removes_from_index(self) -> None:
        self.store.forget(self.root / "terminal.png")
        self.assertEqual(self.store.bm25("syslog", 10, "ocr_text"), [])


class TestChannelSeparation(TempTree):
    """A file named after a term must beat one that merely mentions it.

    Regression: with filename and OCR scored as one BM25 document, a verbose
    screenshot mentioning "code" five times outranked a file called code.jpg —
    it sat at rank 11 of 16. BM25 cannot tell a curated label from incidental
    body text; two rankings fused by RRF can.
    """

    def setUp(self) -> None:
        super().setUp()
        self.store = Store(self.db)
        named = self.write_image("code.png")
        fid, _ = self.store.upsert(named, "image/png", digest_of(named))
        self.store.put_text(fid, "unrelated prose about weather", path_tokens(named, self.root))
        self.named = fid
        for i in range(6):
            p = self.write_image(f"verbose{i}.png")
            f2, _ = self.store.upsert(p, "image/png", digest_of(p))
            self.store.put_text(
                f2, "code code code code code and more code", path_tokens(p, self.root)
            )

    def test_filename_channel_isolates_the_named_file(self) -> None:
        hits = self.store.bm25("code", 10, "path_tokens")
        self.assertEqual(hits[0][0], self.named)

    def test_ocr_channel_does_not_contain_the_named_file(self) -> None:
        ids = [h for h, _ in self.store.bm25("code", 10, "ocr_text")]
        self.assertNotIn(self.named, ids)
        self.assertEqual(len(ids), 6)


class TestSplitExtras(unittest.TestCase):
    def test_prose_adds_nothing(self) -> None:
        self.assertEqual(_split_extras("the quick brown fox jumps"), [])

    def test_compound_words_are_split(self) -> None:
        self.assertEqual(_split_extras("VectorIndex"), ["vector", "index"])

    def test_does_not_duplicate_whole_document(self) -> None:
        """Regression: appending a full copy doubled doc length and skewed BM25."""
        long_prose = " ".join(["ordinary"] * 200)
        self.assertEqual(_split_extras(long_prose), [])


class TestRrf(unittest.TestCase):
    def test_single_ranking_is_monotonic(self) -> None:
        got = rrf([[10, 20, 30]])
        self.assertGreater(got[10], got[20])
        self.assertGreater(got[20], got[30])

    def test_agreement_across_channels_wins(self) -> None:
        got = rrf([[1, 2], [1, 3]])
        self.assertGreater(got[1], got[2])
        self.assertGreater(got[1], got[3])

    def test_exact_formula(self) -> None:
        self.assertAlmostEqual(rrf([[7]])[7], 1.0 / (RRF_K + 1))

    def test_a_document_only_one_channel_found_still_scores(self) -> None:
        self.assertIn(99, rrf([[1], [99]]))

    def test_empty_input(self) -> None:
        self.assertEqual(rrf([]), {})
        self.assertEqual(rrf([[], []]), {})


if __name__ == "__main__":
    unittest.main()
