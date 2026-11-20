import pytest
from fastapi.testclient import TestClient

from diagnotech.api import create_app
from diagnotech.config import Settings


@pytest.fixture()
def client(tmp_path):
    settings = Settings()
    settings.db_path = tmp_path / "api.db"
    settings.index_dir = tmp_path / "index"  # empty: index not built
    settings.processed_dir = tmp_path / "processed"
    settings.log_dir = tmp_path / "logs"
    settings.external_research_enabled = False
    settings.llm_provider = "none"
    app = create_app(settings)
    return TestClient(app)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["index_ready"] is False


def test_kb_status_without_index(client):
    response = client.get("/api/kb/status")
    assert response.status_code == 200
    assert response.json()["chunks"] == 0


def test_conversation_lifecycle(client):
    created = client.post("/api/conversations", json={"title": "اختبار"})
    assert created.status_code == 200
    cid = created.json()["id"]

    chat = client.post("/api/chat", json={"conversation_id": cid, "message": "السلام عليكم"})
    assert chat.status_code == 200
    assert chat.json()["state"] == "greeting"

    messages = client.get(f"/api/conversations/{cid}/messages")
    assert len(messages.json()) == 2

    listing = client.get("/api/conversations")
    assert any(c["id"] == cid for c in listing.json())

    deleted = client.delete(f"/api/conversations/{cid}")
    assert deleted.status_code == 200
    assert client.get(f"/api/conversations/{cid}/messages").status_code == 404


def test_chat_out_of_scope_without_index(client):
    response = client.post("/api/chat", json={"message": "ما هي عاصمة فرنسا؟"})
    assert response.status_code == 200
    assert response.json()["state"] == "out_of_scope"
    assert response.json()["conversation_id"]


def test_frontend_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'lang="ar"' in response.text
    assert 'dir="rtl"' in response.text
