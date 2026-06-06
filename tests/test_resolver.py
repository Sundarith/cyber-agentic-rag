import unittest

from agentic_rag import CorpusIndex, EntityResolver
from agentic_rag.resolver import normalize_name
from agentic_rag.schema import Chunk


class EntityResolverTests(unittest.TestCase):
    def corpus(self):
        return CorpusIndex(
            [
                Chunk(
                    id="attack-T1057",
                    identifier="T1057",
                    source="MITRE ATT&CK",
                    type="technique",
                    name="Process Discovery",
                    text="# T1057\nAdversaries enumerate processes. Related to CAPEC-573.",
                ),
                # Wrong-type collision: a malware chunk sharing the technique name.
                Chunk(
                    id="malware-S9999",
                    identifier="S9999",
                    source="MITRE ATT&CK",
                    type="malware",
                    name="Process Discovery",
                    text="# S9999\nMalware family also called Process Discovery.",
                ),
                Chunk(
                    id="attack-T1059",
                    identifier="T1059",
                    source="MITRE ATT&CK",
                    type="technique",
                    name="Command and Scripting Interpreter",
                    text="# T1059\nAdversaries abuse interpreters.",
                ),
                Chunk(
                    id="capec-CAPEC-573",
                    identifier="CAPEC-573",
                    source="CAPEC",
                    type="attack_pattern",
                    name="Process Footprinting",
                    text="# CAPEC-573\nRelated Root Causes (CWE)\n- CWE-200",
                ),
            ]
        )

    def test_normalize_strips_punctuation_and_case(self):
        self.assertEqual(normalize_name("Process Discovery!"), "process discovery")
        self.assertEqual(normalize_name("  Foo   Bar "), "foo bar")

    def test_exact_name_resolves(self):
        resolver = EntityResolver(self.corpus())
        result = resolver.resolve("Process Discovery")
        self.assertEqual(result.chosen_id, "T1057")
        self.assertEqual(result.mode, "exact")
        self.assertEqual(result.type, "technique")

    def test_type_priority_drops_wrong_type_collision(self):
        resolver = EntityResolver(self.corpus())
        result = resolver.resolve("Process Discovery", prefer_types=["technique"])
        # Despite a same-name malware chunk, the technique wins and this is not
        # treated as ambiguity (different types).
        self.assertEqual(result.chosen_id, "T1057")
        self.assertFalse(result.ambiguous)

    def test_unknown_name_returns_none_mode(self):
        resolver = EntityResolver(self.corpus())
        result = resolver.resolve("totally unrelated xyzzy")
        self.assertFalse(result.resolved)
        self.assertEqual(result.mode, "none")
        self.assertEqual(result.chosen_id, "")

    def test_token_overlap_fallback(self):
        resolver = EntityResolver(self.corpus())
        # No exact/substring match for the full phrase, but tokens overlap the
        # technique name strongly.
        result = resolver.resolve("Scripting Interpreter Command")
        self.assertEqual(result.chosen_id, "T1059")
        self.assertEqual(result.mode, "fallback")

    def test_same_type_tie_flags_ambiguous(self):
        corpus = CorpusIndex(
            [
                Chunk(id="a", identifier="T2000", source="MITRE ATT&CK", type="technique",
                      name="Credential Dumping", text="x"),
                Chunk(id="b", identifier="T2001", source="MITRE ATT&CK", type="technique",
                      name="Credential Dumping", text="y"),
            ]
        )
        resolver = EntityResolver(corpus)
        result = resolver.resolve("Credential Dumping", prefer_types=["technique"])
        self.assertTrue(result.ambiguous)
        self.assertGreaterEqual(len(result.candidates), 2)

    def test_resolve_question_strips_leading_stopword(self):
        resolver = EntityResolver(self.corpus())
        result = resolver.resolve_question(
            "For Process Discovery, which CWE weakness is connected through its CAPEC?"
        )
        self.assertEqual(result.chosen_id, "T1057")
        self.assertEqual(result.mode, "exact")

    def test_resolve_question_lowercase_phrasing(self):
        resolver = EntityResolver(self.corpus())
        # No capitalization cue at all.
        result = resolver.resolve_question("for process footprinting, what is the root-cause weakness?")
        self.assertEqual(result.chosen_id, "CAPEC-573")
        self.assertEqual(result.mode, "exact")

    def test_alias_index_resolves_descriptive_phrase(self):
        # CorpusIndex accepts an alias map (normalized phrase -> ids).
        corpus = CorpusIndex(
            [
                Chunk(id="cwe-200", identifier="CWE-200", source="CWE", type="weakness",
                      name="Exposure of Sensitive Information to an Unauthorized Actor", text="x"),
            ],
            aliases={"information disclosure": ["CWE-200"]},
        )
        resolver = EntityResolver(corpus)
        result = resolver.resolve("information disclosure")
        self.assertEqual(result.chosen_id, "CWE-200")
        self.assertEqual(result.mode, "alias")

    def test_prefer_types_picks_weakness_for_reverse(self):
        # A name shared by a technique and a weakness resolves to the weakness
        # when the (reverse) question prefers weaknesses.
        corpus = CorpusIndex(
            [
                Chunk(id="t", identifier="T7000", source="MITRE ATT&CK", type="technique",
                      name="Spoofing", text="x"),
                Chunk(id="w", identifier="CWE-7000", source="CWE", type="weakness",
                      name="Spoofing", text="y"),
            ]
        )
        resolver = EntityResolver(corpus)
        self.assertEqual(resolver.resolve("Spoofing", prefer_types=["weakness"]).chosen_id, "CWE-7000")
        self.assertEqual(resolver.resolve("Spoofing", prefer_types=["technique"]).chosen_id, "T7000")


if __name__ == "__main__":
    unittest.main()
