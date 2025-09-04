import unittest
from fastapi.testclient import TestClient
from fastapi_app.app import app

class FastAPIAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home_page(self):
        resp = cls.client.get("/")  # type: ignore[name-defined]
        # Status check
        assert resp.status_code == 200
        # If the home route returns HTML, assert on the title tag in text
        assert "<title>Sentiment Analysis</title>" in resp.text

    def test_predict_page(self):
        # Send JSON body; switch to data={"text": "..."} if the route expects form data
        resp = self.client.post("/predict", json={"text": "I love this!"})
        assert resp.status_code == 200
        # If the endpoint returns JSON like {"prediction": "Positive"}
        if resp.headers.get("content-type", "").startswith("application/json"):
            body = resp.json()
            assert body.get("prediction") in ["Positive", "Negative"]
        else:
            # If endpoint returns rendered HTML, assert on response text
            txt = resp.text
            assert ("Positive" in txt) or ("Negative" in txt), \
                "Response should contain either 'Positive' or 'Negative'"

if __name__ == "__main__":
    unittest.main()
