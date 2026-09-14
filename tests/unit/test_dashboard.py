from fastapi.testclient import TestClient
from src.dashboard.api import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_portfolio():
    response = client.get("/api/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert "equity" in data
    assert "daily_pnl" in data
    assert "positions" in data


def test_performance():
    response = client.get("/api/performance")
    assert response.status_code == 200
    data = response.json()
    assert "sharpe" in data
    assert "max_dd" in data
    assert "win_rate" in data


def test_agent_log():
    response = client.get("/api/agent/log?limit=10")
    assert response.status_code == 200


def test_agent_pause():
    response = client.post("/api/agent/pause")
    assert response.status_code == 200
    assert response.json()["status"] == "paused"


def test_agent_resume():
    response = client.post("/api/agent/resume")
    assert response.status_code == 200
    assert response.json()["status"] == "resumed"


def test_agent_close_all():
    response = client.post("/api/agent/close-all")
    assert response.status_code == 200
    assert response.json()["status"] == "all positions closed"


def test_agent_kill_switch():
    response = client.post("/api/agent/kill-switch")
    assert response.status_code == 200
    assert response.json()["status"] == "kill switch activated"


def test_regime():
    response = client.get("/api/regime")
    assert response.status_code == 200
    data = response.json()
    assert "regime" in data
    assert "confidence" in data


def test_trades():
    response = client.get("/api/trades?limit=10")
    assert response.status_code == 200


def test_websocket_connect():
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("ping")
        # Connection stays open, no exception thrown