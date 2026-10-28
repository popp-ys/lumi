import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture

def api(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    backup_dir = data_dir / "backups"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"api_token": "test-token"}), encoding="utf-8")
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "BACKUP_DIR", backup_dir)
    monkeypatch.setattr(main, "DB_PATH", data_dir / "lumi.sqlite3")
    monkeypatch.setattr(main, "CONFIG_PATH", config_path)
    monkeypatch.setitem(main.CONFIG, "api_token", "test-token")

    async def fake_ollama(messages, actions):
        return "Hello, I am Lumi."

    monkeypatch.setattr(main, "ollama_chat", fake_ollama)
    with TestClient(main.app) as client:
        yield client


def auth():
    return {"Authorization": "Bearer test-token"}


def test_auth_is_required(api):
    assert api.get("/api/memories").status_code == 401
    assert api.get("/api/memories", headers={"Authorization": "Bearer wrong"}).status_code == 403


def test_memory_crud(api):
    response = api.post("/api/memories", headers=auth(), json={"content": "Poppy is my server project", "category": "project"})
    assert response.status_code == 200
    memory_id = response.json()["memory"]["id"]
    memories = api.get("/api/memories", headers=auth()).json()["memories"]
    assert memories[0]["content"] == "Poppy is my server project"
    assert api.delete(f"/api/memories/{memory_id}", headers=auth()).status_code == 200


def test_chat_is_persisted_and_backed_up(api):
    response = api.post("/api/chat", headers=auth(), json={"message": "Hey Lumi", "session_id": "test-session"})
    assert response.status_code == 200
    assert response.json()["reply"] == "Hello, I am Lumi."
    history = api.get("/api/history?session_id=test-session", headers=auth()).json()["messages"]
    assert [item["role"] for item in history] == ["user", "assistant"]
    backup = api.post("/api/backup", headers=auth())
    assert backup.status_code == 200
    assert (Path(backup.json()["directory"]) / backup.json()["filename"]).exists()


def test_url_and_explicit_memory_helpers():
    assert main.validate_url("https://example.com/path")
    assert not main.validate_url("javascript:alert(1)")
    assert not main.validate_url("file:///etc/passwd")
    assert main.explicit_memory_from_text("Remember that Poppy is my server") == "Poppy is my server"
    assert main.explicit_memory_from_text("Hello Lumi") is None
