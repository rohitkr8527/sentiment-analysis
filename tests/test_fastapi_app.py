import unittest
from fastapi.testclient import TestClient
from fastapi_app.app import app

class FastAPIAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home_page(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "<title>Sentiment Analysis</title>" in resp.text

    def test_predict_page(self):
        resp = self.client.post("/predict", data={"text": "I love this!"})
        assert resp.status_code == 200, f"Unexpected status code: {resp.status_code} Body: {resp.text}"
        txt = resp.text
        assert ("Positive" in txt) or ("Negative" in txt), \
            "Response should contain either 'Positive' or 'Negative'"

if __name__ == "__main__":
    unittest.main()
