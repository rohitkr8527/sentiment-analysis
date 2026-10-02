import os
import unittest
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.text_preprocessing import normalize_text


class TestSentimentModel(unittest.TestCase):
    """
    Unit and integration tests for sentiment analysis model,
    vectorizer, and preprocessing pipeline.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.getenv("MODEL_PATH", "models/model.pkl")
        cls.vectorizer_path = os.getenv("VECTORIZER_PATH", "models/tfidf_vectorizer.pkl")
        cls.holdout_path = os.getenv("HOLDOUT_DATA_PATH", "data/processed/test_tfidf.csv")

        # Load vectorizer
        if os.path.exists(cls.vectorizer_path):
            with open(cls.vectorizer_path, "rb") as f:
                cls.vectorizer = pickle.load(f)
        else:
            cls.vectorizer = None

        # Load model
        if os.path.exists(cls.model_path):
            with open(cls.model_path, "rb") as f:
                cls.model = pickle.load(f)
        else:
            cls.model = None

    def test_artifacts_exist(self):
        """Verify model and vectorizer artifacts are generated and non-empty."""
        self.assertIsNotNone(self.model, f"Model file not found at {self.model_path}")
        self.assertIsNotNone(self.vectorizer, f"Vectorizer file not found at {self.vectorizer_path}")
        self.assertTrue(os.path.getsize(self.model_path) > 0)
        self.assertTrue(os.path.getsize(self.vectorizer_path) > 0)

    def test_text_normalization(self):
        """Verify text preprocessing pipeline cleans raw text properly."""
        raw_text = "Check out https://example.com! The battery lasts 10 hours and is amazing!!!"
        cleaned = normalize_text(raw_text)

        # Check URL removed
        self.assertNotIn("http", cleaned)
        # Check digits removed
        self.assertNotIn("10", cleaned)
        # Check lowercase
        self.assertEqual(cleaned, cleaned.lower())
        # Check punctuation removed
        self.assertNotIn("!", cleaned)
        # Check content retained
        self.assertIn("battery", cleaned)
        self.assertIn("amazing", cleaned)

    def test_empty_text_normalization(self):
        """Verify text normalization handles None, empty, or whitespace strings safely."""
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("   "), "")

    def test_model_signature_and_dimensions(self):
        """Verify vectorizer feature dimension matches model expectation and produces valid predictions."""
        if self.model is None or self.vectorizer is None:
            self.skipTest("Model or vectorizer not available")

        sample_text = normalize_text("This product is fantastic and exceeded all my expectations.")
        features = self.vectorizer.transform([sample_text])

        self.assertEqual(features.shape[1], len(self.vectorizer.get_feature_names_out()))

        preds = self.model.predict(features)
        self.assertEqual(len(preds), 1)
        self.assertIn(int(preds[0]), [0, 1])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)
            self.assertEqual(probs.shape, (1, 2))
            self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)

    def test_semantic_sentiment_prediction(self):
        """Verify clear positive and negative sentences classify correctly."""
        if self.model is None or self.vectorizer is None:
            self.skipTest("Model or vectorizer not available")

        pos_text = normalize_text("I absolutely love this! It is wonderful, great, and fantastic.")
        neg_text = normalize_text("Terrible experience, completely broken, awful customer service, hate it.")

        pos_feat = self.vectorizer.transform([pos_text])
        neg_feat = self.vectorizer.transform([neg_text])

        pos_pred = self.model.predict(pos_feat)[0]
        neg_pred = self.model.predict(neg_feat)[0]

        self.assertEqual(int(pos_pred), 1, "Expected positive sentiment (1)")
        self.assertEqual(int(neg_pred), 0, "Expected negative sentiment (0)")

    def test_model_holdout_performance_thresholds(self):
        """Verify performance metrics on holdout test set exceed acceptable threshold (75%)."""
        if self.model is None or not os.path.exists(self.holdout_path):
            self.skipTest(f"Holdout dataset not found at {self.holdout_path}")

        holdout_df = pd.read_csv(self.holdout_path)
        X_holdout = holdout_df.iloc[:, :-1].values
        y_holdout = holdout_df.iloc[:, -1].values.astype(int)

        y_pred = self.model.predict(X_holdout)

        acc = accuracy_score(y_holdout, y_pred)
        prec = precision_score(y_holdout, y_pred, zero_division=0)
        rec = recall_score(y_holdout, y_pred, zero_division=0)
        f1 = f1_score(y_holdout, y_pred, zero_division=0)

        min_threshold = 0.75
        self.assertGreaterEqual(acc, min_threshold, f"Accuracy {acc:.3f} below threshold {min_threshold}")
        self.assertGreaterEqual(prec, min_threshold, f"Precision {prec:.3f} below threshold {min_threshold}")
        self.assertGreaterEqual(rec, min_threshold, f"Recall {rec:.3f} below threshold {min_threshold}")
        self.assertGreaterEqual(f1, min_threshold, f"F1 score {f1:.3f} below threshold {min_threshold}")


if __name__ == "__main__":
    unittest.main()
