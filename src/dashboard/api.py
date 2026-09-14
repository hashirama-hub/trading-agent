from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import json
import asyncio
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="Trading Agent Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        disconnected = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(conn)
        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


class Trade(BaseModel):
    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    status: str
    pnl: float = 0.0


class PortfolioUpdate(BaseModel):
    equity: float
    daily_pnl: float
    positions: List[Dict]
    margin_usage: float = 0.0


class AgentLog(BaseModel):
    timestamp: str
    thought: str
    action: str
    result: str


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast({"type": "ping", "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/portfolio")
async def get_portfolio():
    return PortfolioUpdate(
        equity=10000.0,
        daily_pnl=0.0,
        positions=[],
        margin_usage=0.0,
    )


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    return []


@app.get("/api/performance")
async def get_performance():
    return {
        "sharpe": 0.0,
        "max_dd": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "avg_pnl": 0.0,
    }


@app.get("/api/agent/log")
async def get_agent_log(limit: int = 20):
    return []


@app.post("/api/agent/pause")
async def pause_agent():
    return {"status": "paused"}


@app.post("/api/agent/resume")
async def resume_agent():
    return {"status": "resumed"}


@app.post("/api/agent/close-all")
async def close_all_positions():
    return {"status": "all positions closed"}


@app.post("/api/agent/kill-switch")
async def kill_switch():
    return {"status": "kill switch activated"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/regime")
async def get_current_regime():
    return {"regime": "trend", "confidence": 0.85}