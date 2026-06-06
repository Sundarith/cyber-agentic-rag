import sys
import types
import unittest

from agentic_rag.corpus import CorpusIndex
from agentic_rag.retrieval import LegacyHybridRetriever, LexicalRetriever, build_search_backend
from agentic_rag.schema import Chunk


class RetrievalAdapterTests(unittest.TestCase):
    def toy_corpus(self):
        return CorpusIndex(
            [
                Chunk(
                    id="cwe-79",
                    identifier="CWE-79",
                    source="CWE",
                    type="weakness",
                    name="Cross-site Scripting",
                    text="CWE-79 covers cross-site scripting.",
                )
            ]
        )

    def test_lexical_retriever_delegates_to_corpus(self):
        retriever = LexicalRetriever(self.toy_corpus())
        results = retriever.search("cross-site scripting", k=1)
        self.assertEqual(results[0].identifier, "CWE-79")
        self.assertEqual(results[0].tool, "search")

    def test_build_search_backend(self):
        corpus = self.toy_corpus()
        self.assertIsInstance(build_search_backend(corpus, "lexical"), LexicalRetriever)
        self.assertIsInstance(build_search_backend(corpus, "legacy_hybrid"), LegacyHybridRetriever)
        with self.assertRaises(ValueError):
            build_search_backend(corpus, "unknown")

    def test_legacy_hybrid_imports_lazily_and_converts_chunks(self):
        module_name = "_fake_better_rag_for_test"
        fake = types.ModuleType(module_name)
        calls = []

        def retrieve(query, k=8):
            calls.append((query, k))
            return [
                (
                    {
                        "id": "cwe-89",
                        "identifier": "CWE-89",
                        "source": "CWE",
                        "type": "weakness",
                        "name": "SQL Injection",
                        "text": "CWE-89 covers SQL injection.",
                    },
                    0.75,
                )
            ]

        fake.retrieve = retrieve
        sys.modules[module_name] = fake
        try:
            retriever = LegacyHybridRetriever(self.toy_corpus(), module_name=module_name)
            self.assertIsNone(retriever._module)
            results = retriever.search("sql injection", k=3)
            self.assertIs(retriever._module, fake)
            self.assertEqual(calls, [("sql injection", 3)])
            self.assertEqual(results[0].identifier, "CWE-89")
            self.assertEqual(results[0].tool, "hybrid_search")
            self.assertAlmostEqual(results[0].score, 0.75)
        finally:
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
