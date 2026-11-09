"""
Campus FAQ Chatbot Engine
This module implements the core Natural Language Processing (NLP) pipeline
and Machine Learning matching using TF-IDF and Cosine Similarity.
"""

import json
import logging
import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Built-in fallback English stopwords (used if NLTK stopwords corpus is missing/offline)
FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "while", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Domain & question intent words to preserve even if considered stopwords in general NLP
DOMAIN_PRESERVE_WORDS = {"can", "how", "much", "many", "cost", "fee", "what", "when", "where", "which", "who", "why"}


def normalize_term(text: str) -> str:
    """Normalize text by removing hyphens and non-alphanumeric chars for flexible keyword search."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.replace("-", "")).lower().strip()


class NLPPreprocessor:
    """Handles text preprocessing including lowercasing, tokenization,
    punctuation removal, stopword filtering, and lemmatization with graceful fallbacks.
    """

    def __init__(self):
        self._init_nltk_resources()

    def _init_nltk_resources(self):
        """Safely load NLTK resources with automatic download and robust fallbacks."""
        # 1. Stopwords
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                self.stop_words = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                self.stop_words = set(stopwords.words("english"))
        except Exception as e:
            logger.warning(f"Could not load NLTK stopwords ({e}). Using built-in fallback stopwords.")
            self.stop_words = FALLBACK_STOPWORDS.copy()

        # Preserve critical intent words
        self.stop_words = self.stop_words - DOMAIN_PRESERVE_WORDS

        # 2. Tokenizer
        self.use_nltk_tokenizer = False
        try:
            import nltk
            try:
                nltk.word_tokenize("test sentence")
                self.use_nltk_tokenizer = True
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                nltk.word_tokenize("test sentence")
                self.use_nltk_tokenizer = True
        except Exception as e:
            logger.warning(f"NLTK tokenizer unavailable ({e}). Using regex tokenizer.")
            self.use_nltk_tokenizer = False

        # 3. Lemmatizer
        self.lemmatizer = None
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            try:
                lem = WordNetLemmatizer()
                lem.lemmatize("students")
                self.lemmatizer = lem
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                lem = WordNetLemmatizer()
                lem.lemmatize("students")
                self.lemmatizer = lem
        except Exception as e:
            logger.warning(f"NLTK WordNet Lemmatizer unavailable ({e}). Lemmatization skipped.")
            self.lemmatizer = None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into alphanumeric words, stripping punctuation."""
        if not text:
            return []
        
        # Replace hyphens with spaces for compound terms like wi-fi -> wi fi
        normalized = text.replace("-", " ")

        if self.use_nltk_tokenizer:
            try:
                import nltk
                raw_tokens = nltk.word_tokenize(normalized)
                return [t for t in raw_tokens if any(c.isalnum() for c in t)]
            except Exception:
                pass
        
        # Regex fallback tokenizer
        return re.findall(r"\b[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*\b", normalized)

    def preprocess(self, text: str, remove_stopwords: bool = True) -> str:
        """Full NLP pipeline:
        1. Lowercase
        2. Normalize hyphenated words
        3. Tokenize
        4. Remove punctuation
        5. Stopword removal (optional)
        6. Lemmatize
        Returns a cleaned, normalized string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Lowercase
        text_lower = text.lower().strip()

        # Tokenize (also handles hyphen separation)
        tokens = self.tokenize(text_lower)

        # Remove single punctuation characters
        tokens = [t for t in tokens if t not in string.punctuation]

        # Stopword removal
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Lemmatization
        if self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            except Exception:
                pass

        return " ".join(tokens)


class FAQChatbot:
    """FAQ Chatbot using dual TF-IDF vectorization and Cosine Similarity."""

    DEFAULT_THRESHOLD = 0.25

    FALLBACK_RESPONSES = [
        "I'm sorry, I couldn't find a direct answer to your question in our campus database. "
        "Please check the spelling or try rephrasing your question.",
        "Could you please rephrase or specify a topic (e.g., Admissions, Hostel, Fees, Courses, Placements)?",
        "Our helpdesk is also available at helpdesk@campus.edu or call +1 (800) 555-0199 for personal assistance."
    ]

    def __init__(self, faq_file_path: Optional[str] = None, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.preprocessor = NLPPreprocessor()
        
        # Determine FAQ file path
        if faq_file_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.faq_file_path = os.path.join(base_dir, "faqs.json")
        else:
            self.faq_file_path = faq_file_path

        self.faqs: List[Dict[str, Any]] = []
        self.question_corpus: List[str] = []
        self.enriched_corpus: List[str] = []
        
        self.question_vectorizer: Optional[TfidfVectorizer] = None
        self.question_tfidf_matrix = None
        self.enriched_vectorizer: Optional[TfidfVectorizer] = None
        self.enriched_tfidf_matrix = None

        self.load_dataset()
        self.train()

    def load_dataset(self) -> None:
        """Load FAQ entries from the JSON dataset."""
        if not os.path.exists(self.faq_file_path):
            raise FileNotFoundError(f"FAQ dataset file not found at: {self.faq_file_path}")

        with open(self.faq_file_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        if not isinstance(self.faqs, list) or len(self.faqs) == 0:
            raise ValueError("FAQ dataset must be a non-empty list of question-answer objects.")

        logger.info(f"Loaded {len(self.faqs)} FAQs across {len(self.get_categories())} categories.")

    def train(self) -> None:
        """Fit TF-IDF vectorizers on both pure question corpus and enriched corpus."""
        # 1. Question-only corpus for high-precision exact matching
        self.question_corpus = [
            self.preprocessor.preprocess(faq.get("question", ""))
            for faq in self.faqs
        ]
        self.question_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        self.question_tfidf_matrix = self.question_vectorizer.fit_transform(self.question_corpus)

        # 2. Enriched corpus (question + keywords + category) for high-recall semantic matching
        self.enriched_corpus = [
            self.preprocessor.preprocess(
                f"{faq.get('question', '')} {faq.get('category', '')} {' '.join(faq.get('keywords', []))}"
            )
            for faq in self.faqs
        ]
        self.enriched_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        self.enriched_tfidf_matrix = self.enriched_vectorizer.fit_transform(self.enriched_corpus)

        logger.info(
            f"Trained TF-IDF vectorizers. Question vocab: {len(self.question_vectorizer.vocabulary_)}, "
            f"Enriched vocab: {len(self.enriched_vectorizer.vocabulary_)}"
        )

    def get_categories(self) -> List[str]:
        """Return a sorted list of unique FAQ categories."""
        categories = {faq.get("category", "General") for faq in self.faqs}
        return sorted(list(categories))

    def get_faqs_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Return all FAQs belonging to a specific category."""
        return [faq for faq in self.faqs if faq.get("category", "").lower() == category.lower()]

    def search_faqs(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform a keyword-based text search across questions, answers, and keywords."""
        if not query or not query.strip():
            return []

        q_clean = query.lower().strip()
        q_norm = normalize_term(query)

        results = []
        for faq in self.faqs:
            if category and category.lower() != "all" and faq.get("category", "").lower() != category.lower():
                continue

            q_text = faq.get("question", "").lower()
            a_text = faq.get("answer", "").lower()
            keywords = [kw.lower() for kw in faq.get("keywords", [])]

            # Direct substring check
            direct_match = (
                q_clean in q_text or
                q_clean in a_text or
                any(q_clean in kw for kw in keywords)
            )

            # Normalized check (strips hyphens/punctuation e.g. "wifi" matches "Wi-Fi")
            norm_match = False
            if not direct_match and q_norm:
                norm_match = (
                    q_norm in normalize_term(q_text) or
                    q_norm in normalize_term(a_text) or
                    any(q_norm in normalize_term(kw) for kw in keywords)
                )

            if direct_match or norm_match:
                results.append(faq)
        return results

    def get_suggested_questions(self, limit: int = 5, category: Optional[str] = None) -> List[str]:
        """Return a representative sample of questions for user suggestion chips."""
        candidates = self.faqs
        if category and category.lower() != "all":
            candidates = [f for f in self.faqs if f.get("category", "").lower() == category.lower()]
        
        selected = []
        seen_cats = set()
        for f in candidates:
            cat = f.get("category")
            if cat not in seen_cats:
                selected.append(f.get("question"))
                seen_cats.add(cat)
            if len(selected) >= limit:
                break
        
        if len(selected) < limit:
            for f in candidates:
                q = f.get("question")
                if q not in selected:
                    selected.append(q)
                if len(selected) >= limit:
                    break
        return selected

    def compute_similarity(self, user_query: str) -> np.ndarray:
        """Preprocess user query, compute cosine similarities across question and enriched indices,
        and combine them for maximum precision and recall.
        """
        processed_query = self.preprocessor.preprocess(user_query)
        if not processed_query.strip():
            return np.zeros(len(self.faqs))

        # 1. Similarity against exact question index
        q_vec = self.question_vectorizer.transform([processed_query])
        q_sims = cosine_similarity(q_vec, self.question_tfidf_matrix).flatten()

        # 2. Similarity against enriched index (keywords + category)
        e_vec = self.enriched_vectorizer.transform([processed_query])
        e_sims = cosine_similarity(e_vec, self.enriched_tfidf_matrix).flatten()

        # 3. Combine: exact question matches retain up to 1.0; enriched matches provide broad recall
        # We take the maximum of the direct question match or 0.9 * enriched match
        combined = np.maximum(q_sims, 0.90 * e_sims)
        return combined

    def get_response(
        self,
        user_query: str,
        category_filter: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate chatbot response for a given user query.

        Args:
            user_query: The raw string question asked by the user.
            category_filter: Optional category name to restrict or prioritize matching.
            threshold: Optional override for cosine similarity cutoff.

        Returns:
            A dictionary containing:
                - answer (str): The FAQ answer or fallback message.
                - matched_question (str or None): The best matching FAQ question.
                - category (str or None): Matched category name.
                - similarity_score (float): Raw cosine similarity score [0.0, 1.0].
                - confidence_label (str): Human-readable similarity band description.
                - is_fallback (bool): Whether a fallback response was triggered.
                - top_matches (list): Up to 3 relevant alternative questions and scores.
        """
        active_threshold = self.threshold if threshold is None else threshold

        # 1. Handle empty or punctuation-only input
        cleaned_input = user_query.strip() if user_query else ""
        if not cleaned_input or not any(c.isalnum() for c in cleaned_input):
            return {
                "answer": "Hello! I am the Campus FAQ Assistant. Please ask any question about admissions, courses, fees, hostels, placements, examinations, or campus facilities.",
                "matched_question": None,
                "category": None,
                "similarity_score": 0.0,
                "confidence_label": "Empty / General Greeting",
                "is_fallback": True,
                "top_matches": []
            }

        # 2. Compute similarity against all FAQs
        similarities = self.compute_similarity(cleaned_input)

        # 3. Apply category filter if specified
        if category_filter and category_filter.lower() != "all":
            for idx, faq in enumerate(self.faqs):
                if faq.get("category", "").lower() != category_filter.lower():
                    similarities[idx] = -1.0

        # Check if all similarities are non-positive
        if np.max(similarities) <= 0.0:
            return {
                "answer": f"{self.FALLBACK_RESPONSES[0]}\n\n"
                          f"💡 *Suggestion:* You can select a category from the sidebar or try one of the suggested sample questions.",
                "matched_question": None,
                "category": None,
                "similarity_score": 0.0,
                "confidence_label": "No Match (Score: 0.000)",
                "is_fallback": True,
                "top_matches": []
            }

        # 4. Get best match and ranking
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        # Prepare top 3 candidate matches for transparency
        top_indices = np.argsort(similarities)[::-1][:3]
        top_matches = [
            {
                "id": self.faqs[i]["id"],
                "category": self.faqs[i]["category"],
                "question": self.faqs[i]["question"],
                "similarity_score": round(float(similarities[i]), 4)
            }
            for i in top_indices
            if similarities[i] > 0.05
        ]

        # 5. Determine confidence label (NOTE: Cosine similarity is a geometric metric, not a probability)
        if best_score >= 0.65:
            confidence_label = f"High Match ({best_score:.2f})"
        elif best_score >= active_threshold:
            confidence_label = f"Moderate Match ({best_score:.2f})"
        else:
            confidence_label = f"Low Similarity ({best_score:.2f} < {active_threshold:.2f})"

        # 6. Check similarity threshold
        if best_score < active_threshold:
            suggested_text = ""
            if top_matches:
                suggested_text = "\n\n**Did you mean one of these?**\n" + "\n".join(
                    [f"- *{m['question']}* ({m['category']})" for m in top_matches]
                )

            return {
                "answer": (
                    f"I couldn't find a sufficiently close match for your query in our FAQ database "
                    f"(best similarity score was {best_score:.2f}, which is below the threshold of {active_threshold:.2f})."
                    f"{suggested_text}\n\n"
                    f"You may also contact our support team at **helpdesk@campus.edu**."
                ),
                "matched_question": self.faqs[best_idx]["question"],
                "category": self.faqs[best_idx]["category"],
                "similarity_score": round(best_score, 4),
                "confidence_label": confidence_label,
                "is_fallback": True,
                "top_matches": top_matches
            }

        # 7. Valid FAQ match returned
        matched_faq = self.faqs[best_idx]
        return {
            "answer": matched_faq["answer"],
            "matched_question": matched_faq["question"],
            "category": matched_faq["category"],
            "similarity_score": round(best_score, 4),
            "confidence_label": confidence_label,
            "is_fallback": False,
            "top_matches": top_matches
        }
