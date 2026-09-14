import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class EpisodicMemory:
    def __init__(self, db_path: str = "data/trade_journal.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                decision TEXT,
                outcome TEXT,
                pnl REAL,
                reasoning TEXT,
                market_context TEXT,
                regime TEXT,
                model_version TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                error TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_trade(self, trade: Dict):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO trades (timestamp, symbol, decision, outcome, pnl, reasoning, market_context, regime, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            trade.get("symbol"),
            trade.get("decision"),
            trade.get("outcome"),
            trade.get("pnl"),
            trade.get("reasoning"),
            json.dumps(trade.get("market_context", {})),
            trade.get("regime"),
            trade.get("model_version", "v1"),
        ))
        conn.commit()
        conn.close()

    def log_error(self, error: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO errors (timestamp, error) VALUES (?, ?)", (datetime.utcnow().isoformat(), error))
        conn.commit()
        conn.close()

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_performance_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM trades WHERE outcome IS NOT NULL").fetchall()
        conn.close()
        if not rows:
            return {"total": 0, "win_rate": 0, "avg_pnl": 0, "max_dd": 0}

        trades = [dict(r) for r in rows]
        wins = [t for t in trades if float(t["pnl"]) > 0]
        losses = [t for t in trades if float(t["pnl"]) <= 0]

        eq_curve = []
        cumulative = 0.0
        for t in trades:
            cumulative += float(t["pnl"])
            eq_curve.append(cumulative)

        peak = 0.0
        max_dd = 0.0
        for eq in eq_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        return {
            "total": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "avg_pnl": sum(float(t["pnl"]) for t in trades) / len(trades),
            "max_dd": max_dd,
        }

    def get_all_errors(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM errors ORDER BY timestamp DESC LIMIT 50").fetchall()
        conn.close()
        return [dict(r) for r in rows]