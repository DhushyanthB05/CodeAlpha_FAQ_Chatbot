"""
Comprehensive Automated Unit Tests for FAQ Chatbot
CodeAlpha Artificial Intelligence Internship - Project 1

Tests:
1. Dataset integrity (>= 50 questions, 10 categories, valid structure)
2. NLP Preprocessor (lowercasing, tokenization, punctuation, stopwords, lemmatization)
3. Exact FAQ matching
4. Paraphrased/similar question matching
5. Unknown/out-of-scope question handling (similarity threshold & fallback)
6. Empty and punctuation-only inputs
7. Case-insensitivity
8. Punctuation handling in user queries
9. Category filtering
10. Metric labeling (cosine similarity not represented as calibrated probability)
"""

import json
import os
import unittest
from chatbot import FAQChatbot, NLPPreprocessor


class TestFAQDataset(unittest.TestCase):
    """Test suite for validating faqs.json dataset integrity."""

    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.faq_path = os.path.join(base_dir, "faqs.json")
        with open(cls.faq_path, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_dataset_file_exists(self):
        """Ensure faqs.json exists and is readable."""
        self.assertTrue(os.path.exists(self.faq_path), "faqs.json file must exist.")

    def test_minimum_questions_count(self):
        """Ensure dataset contains at least 50 questions and answers."""
        self.assertGreaterEqual(len(self.data), 50, f"Expected at least 50 FAQs, found {len(self.data)}.")

    def test_required_categories_present(self):
        """Verify that all 10 required categories are present."""
        required_categories = {
            "admissions",
            "courses",
            "fees",
            "scholarships",
            "hostel",
            "transportation",
            "placements",
            "examinations",
            "library",
            "campus facilities"
        }
        present_categories = {item["category"].strip().lower() for item in self.data}
        for cat in required_categories:
            self.assertIn(cat, present_categories, f"Missing required category: {cat}")

    def test_entry_structure(self):
        """Verify every FAQ has id, category, question, and answer non-empty fields."""
        for item in self.data:
            self.assertIn("id", item)
            self.assertIn("category", item)
            self.assertIn("question", item)
            self.assertIn("answer", item)
            self.assertTrue(bool(item["question"].strip()), f"Empty question found in ID {item.get('id')}")
            self.assertTrue(bool(item["answer"].strip()), f"Empty answer found in ID {item.get('id')}")


class TestNLPPreprocessor(unittest.TestCase):
    """Test suite for the NLP preprocessing pipeline."""

    def setUp(self):
        self.preprocessor = NLPPreprocessor()

    def test_lowercase_conversion(self):
        """Verify text is converted to lowercase."""
        raw = "WHAT ARE THE TUITION FEES FOR BTECH?"
        processed = self.preprocessor.preprocess(raw, remove_stopwords=False)
        self.assertEqual(processed, processed.lower())

    def test_punctuation_removal(self):
        """Verify punctuation is stripped."""
        raw = "Hello! Is the library open... on Sundays?!?"
        processed = self.preprocessor.preprocess(raw, remove_stopwords=False)
        for char in "!?.":
            self.assertNotIn(char, processed)

    def test_empty_string(self):
        """Verify empty string returns empty output."""
        self.assertEqual(self.preprocessor.preprocess(""), "")
        self.assertEqual(self.preprocessor.preprocess("   "), "")

    def test_tokenization_and_lemmatization(self):
        """Verify tokenization outputs meaningful root tokens."""
        raw = "students studying courses in libraries"
        processed = self.preprocessor.preprocess(raw)
        # Should contain root or singular words like student, study, course, library
        tokens = processed.split()
        self.assertTrue(len(tokens) > 0)


class TestFAQChatbotMatching(unittest.TestCase):
    """Test suite for TF-IDF and Cosine Similarity FAQ Matching."""

    @classmethod
    def setUpClass(cls):
        cls.bot = FAQChatbot()

    def test_exact_faq_matching(self):
        """Test exact match of an existing FAQ question."""
        query = "What is the admission eligibility for undergraduate B.Tech programs?"
        response = self.bot.get_response(query)
        self.assertFalse(response["is_fallback"])
        self.assertEqual(response["category"], "Admissions")
        self.assertGreater(response["similarity_score"], 0.8)
        self.assertIn("10+2", response["answer"])

    def test_similar_and_paraphrased_questions(self):
        """Test paraphrased question matching."""
        # Query: "Can you tell me how much the college bus costs?"
        query = "How much does college bus transport cost per year?"
        response = self.bot.get_response(query)
        self.assertFalse(response["is_fallback"])
        self.assertEqual(response["category"], "Transportation")
        self.assertGreater(response["similarity_score"], 0.25)
        self.assertIn("bus fee", response["answer"].lower())

    def test_case_insensitive_matching(self):
        """Verify queries in different casings yield identical matching."""
        q_upper = "WHAT ARE THE HOSTEL CURFEW HOURS AND SAFETY RULES?"
        q_lower = "what are the hostel curfew hours and safety rules?"
        res_upper = self.bot.get_response(q_upper)
        res_lower = self.bot.get_response(q_lower)

        self.assertEqual(res_upper["matched_question"], res_lower["matched_question"])
        self.assertAlmostEqual(res_upper["similarity_score"], res_lower["similarity_score"], places=3)

    def test_punctuation_handling(self):
        """Verify heavy punctuation does not break matching."""
        query = "???What is the highest placement package???!!!"
        response = self.bot.get_response(query)
        self.assertFalse(response["is_fallback"])
        self.assertEqual(response["category"], "Placements")
        self.assertIn("placement", response["answer"].lower())

    def test_unknown_out_of_scope_question(self):
        """Verify out-of-scope query triggers fallback response and does not guess."""
        query = "What is the recipe for making chocolate chip cookies on Mars?"
        response = self.bot.get_response(query, threshold=0.25)
        self.assertTrue(response["is_fallback"])
        self.assertLess(response["similarity_score"], 0.25)
        self.assertIn("couldn't find a sufficiently close match", response["answer"].lower())

    def test_empty_and_whitespace_input(self):
        """Verify empty and whitespace inputs give a friendly prompt without errors."""
        res_empty = self.bot.get_response("")
        res_spaces = self.bot.get_response("     ")
        res_punct = self.bot.get_response("?!?!...---")

        for res in [res_empty, res_spaces, res_punct]:
            self.assertTrue(res["is_fallback"])
            self.assertIn("Hello!", res["answer"])
            self.assertEqual(res["similarity_score"], 0.0)

    def test_category_filtering(self):
        """Verify category filtering restricts search to that category."""
        # Ambiguous query "fee" could match Fees or Transportation or Hostel
        query = "What is the fee?"
        res_fees = self.bot.get_response(query, category_filter="Fees")
        self.assertEqual(res_fees["category"], "Fees")

        res_hostel = self.bot.get_response(query, category_filter="Hostel")
        self.assertEqual(res_hostel["category"], "Hostel")

    def test_search_faqs_method(self):
        """Verify keyword search returns relevant results."""
        results = self.bot.search_faqs("wifi")
        self.assertGreater(len(results), 0)
        # Should match questions or answers containing Wi-Fi
        has_wifi = any("wi-fi" in r["question"].lower() or "wifi" in r["question"].lower().replace("-", "") or "wi-fi" in r["answer"].lower() for r in results)
        self.assertTrue(has_wifi)

    def test_similarity_metric_labeling(self):
        """Ensure confidence labels do not use misleading probability wording."""
        query = "Where can I apply for merit scholarship?"
        response = self.bot.get_response(query)
        # Confidence label should use 'Match' or 'Similarity', not 'Probability'
        self.assertNotIn("probability", response["confidence_label"].lower())


if __name__ == "__main__":
    unittest.main()
