#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vault"))


# Each section padded above the 20-word minimum so the parser keeps them.
SAMPLE_HTML = """
<html><body>
<div id="mw-content-text">
<table class="infobox"><tr><td>infobox should be dropped</td></tr></table>
<p>Mongolia is a landlocked country in East Asia. It has a long border with Russia to the north and the People's Republic of China to the south and southeast.</p>
<p>Mongolia is one of the largest landlocked countries in the world by land area.</p>
<h2>History<span class="mw-editsection">[edit]</span></h2>
<p>The Mongol Empire was founded by Genghis Khan in 1206. The empire eventually became the largest contiguous land empire in human history, spanning much of Eurasia.</p>
<h3>Modern era</h3>
<p>Mongolia became a parliamentary republic in 1992 after decades of socialist government. The capital and largest city is Ulaanbaatar, home to nearly half the country's population.</p>
<h2>References</h2>
<div class="reflist">references should be dropped</div>
<p>This bibliographic noise must not survive because the whole section is a junk heading and gets dropped.</p>
<h2>External links</h2>
<p>External link text that must not appear in any chunk because the heading is in the junk list.</p>
<h2>See also</h2>
<p>List of related articles that should never appear in retrieval results either, since this heading is also junk.</p>
</div>
</body></html>
"""


class ParseAndChunkTests(unittest.TestCase):
    def test_parse_article_drops_cruft_and_groups_by_heading(self):
        from world.ingest_wiki import parse_article_html
        sections = parse_article_html(SAMPLE_HTML)
        paths = [s.path for s in sections]
        self.assertEqual(paths[:3], ["Lead", "History", "Modern era"])
        for junk in ("References", "External links", "See also"):
            self.assertNotIn(junk, paths)
        joined = " ".join(s.text for s in sections)
        self.assertNotIn("infobox", joined)
        self.assertNotIn("references should be dropped", joined)
        self.assertNotIn("bibliographic noise", joined)
        self.assertNotIn("External link text", joined)
        self.assertNotIn("[edit]", joined)
        self.assertIn("Mongolia is a landlocked country", joined)
        self.assertIn("Genghis Khan", joined)

    def test_chunk_sections_uses_speakable_prefix(self):
        from world.ingest_wiki import parse_article_html, chunk_sections
        sections = parse_article_html(SAMPLE_HTML)
        chunks = chunk_sections(sections, "Mongolia", max_words=400, overlap=50)
        self.assertGreaterEqual(len(chunks), 3)
        # Lead section drops the "Lead" suffix — title alone is enough.
        self.assertTrue(chunks[0].content.startswith("Mongolia. "))
        history_chunk = next(c for c in chunks if "Genghis Khan" in c.content)
        self.assertTrue(history_chunk.content.startswith("Mongolia. History. "))
        # No em dash artifacts left in any chunk's prefix.
        for c in chunks:
            self.assertNotIn(" — ", c.content[:40])

    def test_chunk_sections_strips_leading_punctuation_in_title(self):
        from world.ingest_wiki import chunk_sections, Section
        long = " ".join(f"word{i}" for i in range(40))
        chunks = chunk_sections(
            [Section(path="Lead", text=long)],
            title="'N Sync (album)",
            max_words=400,
        )
        self.assertGreaterEqual(len(chunks), 1)
        # Leading quote+apostrophe stripped; "N Sync (album)." remains.
        self.assertTrue(chunks[0].content.startswith("N Sync"))

    def test_chunk_sections_slides_with_overlap(self):
        from world.ingest_wiki import chunk_sections, Section
        long_text = " ".join(f"word{i}" for i in range(1000))
        chunks = chunk_sections(
            [Section(path="Body", text=long_text)],
            title="Doc",
            max_words=400,
            overlap=50,
        )
        self.assertGreaterEqual(len(chunks), 3)
        for c in chunks:
            self.assertTrue(c.content.startswith("Doc. Body. "))
        # Verify overlap: last 50 words of chunk 0 appear at start of chunk 1.
        body0 = chunks[0].content.split("Doc. Body. ", 1)[1].split()
        body1 = chunks[1].content.split("Doc. Body. ", 1)[1].split()
        self.assertEqual(body0[-50:], body1[:50])

    def test_chunk_content_hash_is_stable(self):
        from world.ingest_wiki import chunk_sections, Section
        chunks_a = chunk_sections(
            [Section(path="X", text=" ".join(["abc"] * 30))], title="T", max_words=400
        )
        chunks_b = chunk_sections(
            [Section(path="X", text=" ".join(["abc"] * 30))], title="T", max_words=400
        )
        self.assertEqual(chunks_a[0].content_hash, chunks_b[0].content_hash)


class StubArticleTests(unittest.TestCase):
    SHORT_LEAD_HTML = """
    <html><body><div id="mw-content-text">
    <p>'Ayy is one of the districts of Karak governorate, Jordan.</p>
    </div></body></html>
    """

    def test_short_lead_is_kept(self):
        # 11-word stub article — Lead must survive the min-word filter
        # so the article remains searchable.
        from world.ingest_wiki import parse_article_html, chunk_sections
        sections = parse_article_html(self.SHORT_LEAD_HTML)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].path, "Lead")
        self.assertIn("Karak governorate", sections[0].text)
        chunks = chunk_sections(sections, "'Ayy")
        self.assertEqual(len(chunks), 1)


class CleanSpeakableTests(unittest.TestCase):
    def test_space_before_punctuation(self):
        from world.ingest_wiki import clean_speakable
        self.assertEqual(clean_speakable("Russia ."), "Russia.")
        self.assertEqual(clean_speakable("China ,"), "China,")
        self.assertEqual(clean_speakable("End !"), "End!")

    def test_citation_brackets_removed(self):
        from world.ingest_wiki import clean_speakable
        self.assertEqual(
            clean_speakable("It snowed [1] in winter [citation needed]."),
            "It snowed in winter.",
        )

    def test_pronunciation_parens_removed(self):
        from world.ingest_wiki import clean_speakable
        self.assertEqual(
            clean_speakable("Beijing (/beɪˈdʒɪŋ/) is the capital."),
            "Beijing is the capital.",
        )

    def test_collapses_multi_whitespace(self):
        from world.ingest_wiki import clean_speakable
        self.assertEqual(
            clean_speakable("a   b\t\tc\nd"),
            "a b c d",
        )

    def test_nfkc_normalizes_unicode(self):
        from world.ingest_wiki import clean_speakable
        # NFKC turns the precomposed ﬃ ligature into "ffi".
        self.assertEqual(clean_speakable("oﬃce"), "office")


if __name__ == "__main__":
    unittest.main()
