import unittest

from agentic_rag.corpus import CorpusIndex
from agentic_rag.retrieval import LexicalRetriever, build_search_backend
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
        with self.assertRaises(ValueError):
            build_search_backend(corpus, "unknown")


if __name__ == "__main__":
    unittest.main()
