import unittest
from fastapi.testclient import TestClient
from fastapi_app.app import app


class FastAPIAppTests(unittest.TestCase):
    """Integration test suite for FastAPI web and REST endpoints."""

    @classmethod
    def setUpClass(cls):
        # Using TestClient as a context manager ensures lifespan events run
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_health_check(self):
        """GET /health must return 200 and healthy status."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertTrue(data["model_loaded"])

    def test_ready_check(self):
        """GET /ready must return 200 when model is loaded."""
        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")

    def test_home_page(self):
        """GET / returns HTML user interface."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("Sentiment Analysis", resp.text)
        self.assertIn("<form", resp.text)

    def test_web_predict_form_positive(self):
        """POST /predict with form data classifies positive text."""
        resp = self.client.post("/predict", data={"text": "I love this product, it is absolutely amazing!"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Positive", resp.text)
        self.assertIn("😊", resp.text)

    def test_web_predict_form_negative(self):
        """POST /predict with form data classifies negative text."""
        resp = self.client.post("/predict", data={"text": "This is completely broken, terrible and useless."})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Negative", resp.text)
        self.assertIn("😞", resp.text)

    def test_api_v1_predict_positive(self):
        """POST /api/v1/predict returns structured JSON response."""
        payload = {"text": "Excellent service and incredible build quality! Highly recommended."}
        resp = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["sentiment"], "Positive")
        self.assertEqual(data["label"], 1)
        self.assertGreaterEqual(data["confidence"], 0.5)
        self.assertIn("latency_ms", data)
        self.assertEqual(data["text"], payload["text"])

    def test_api_v1_predict_negative(self):
        """POST /api/v1/predict classifies negative text properly."""
        payload = {"text": "Awful experience, never buying this again. Total waste of money."}
        resp = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["sentiment"], "Negative")
        self.assertEqual(data["label"], 0)
        self.assertGreaterEqual(data["confidence"], 0.5)

    def test_api_v1_validation_error(self):
        """POST /api/v1/predict rejects invalid / empty input with 422 Unprocessable Entity."""
        resp = self.client.post("/api/v1/predict", json={"text": ""})
        self.assertEqual(resp.status_code, 422)

        resp_missing = self.client.post("/api/v1/predict", json={})
        self.assertEqual(resp_missing.status_code, 422)

    def test_metrics_endpoint(self):
        """GET /metrics returns Prometheus format metrics."""
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("app_request_count", resp.text)
        self.assertIn("model_prediction_count", resp.text)


if __name__ == "__main__":
    unittest.main()
