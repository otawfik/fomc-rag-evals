"""End-to-end through the HTTP API: loads the real models and index (slow; run with -m slow)."""
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


def test_ask_end_to_end():
    from fomc.api import app

    with TestClient(app) as c:
        assert c.get("/health").json()["ok"]
        r = c.post("/ask", json={"question": "Who voted against the action at the July 30, 2025 meeting?"}).json()
        assert r["source"] == "fomc-2025-07-30" and r["supported"]
        assert "Bowman" in r["answer"] and "Waller" in r["answer"]
        assert r["sources"][0]["url"].startswith("https://www.federalreserve.gov/")
        bad = c.post("/ask", json={"question": "?"})
        assert bad.status_code == 422
