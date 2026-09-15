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
                model_version TEXT,
                confidence REAL,
                setup_type TEXT,
                entry_price REAL,
                exit_price REAL,
                actual_pnl REAL,
                hold_time_seconds INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                error TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                total_trades INTEGER,
                win_rate REAL,
                avg_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                regime_performance TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_trade(self, trade: Dict):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO trades (timestamp, symbol, decision, outcome, pnl, reasoning,
                market_context, regime, model_version, confidence, setup_type,
                entry_price, exit_price, actual_pnl, hold_time_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            trade.get("symbol"),
            trade.get("decision"),
            trade.get("outcome", "pending"),
            trade.get("pnl", 0),
            trade.get("reasoning"),
            json.dumps(trade.get("market_context", {})),
            trade.get("regime"),
            trade.get("model_version", "v1"),
            trade.get("confidence"),
            trade.get("setup_type"),
            trade.get("entry_price"),
            trade.get("exit_price"),
            trade.get("actual_pnl"),
            trade.get("hold_time_seconds"),
        ))
        conn.commit()
        conn.close()

    def update_trade_outcome(self, trade_id: int, outcome: str, actual_pnl: float,
                             exit_price: float = None, hold_time_seconds: int = None):
        """Update a trade with its actual outcome after market resolution."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE trades SET outcome=?, actual_pnl=?, exit_price=?, hold_time_seconds=?
            WHERE id=?
        """, (outcome, actual_pnl, exit_price, hold_time_seconds, trade_id))
        conn.commit()
        conn.close()

    def get_pending_trades(self) -> List[Dict]:
        """Get trades that haven't been resolved yet."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM trades WHERE outcome='pending' ORDER BY timestamp ASC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

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
        rows = conn.execute("SELECT * FROM trades WHERE outcome IS NOT NULL AND outcome != 'pending'").fetchall()
        conn.close()
        if not rows:
            return {"total": 0, "win_rate": 0, "avg_pnl": 0, "max_dd": 0, "sharpe": 0}

        trades = [dict(r) for r in rows]
        wins = [t for t in trades if float(t.get("actual_pnl") or t.get("pnl") or 0) > 0]
        losses = [t for t in trades if float(t.get("actual_pnl") or t.get("pnl") or 0) <= 0]

        pnls = [float(t.get("actual_pnl") or t.get("pnl") or 0) for t in trades]

        eq_curve = []
        cumulative = 0.0
        for pnl in pnls:
            cumulative += pnl
            eq_curve.append(cumulative)

        peak = 0.0
        max_dd = 0.0
        for eq in eq_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        std_pnl = (sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5 if len(pnls) > 1 else 1
        sharpe = (avg_pnl / std_pnl) * (252 ** 0.5) if std_pnl > 0 else 0

        return {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "avg_pnl": avg_pnl,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
        }

    def get_performance_by_regime(self) -> Dict[str, Dict]:
        """Get performance breakdown by market regime."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT regime, COUNT(*) as count,
                   AVG(CASE WHEN actual_pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                   AVG(COALESCE(actual_pnl, pnl)) as avg_pnl
            FROM trades
            WHERE outcome IS NOT NULL AND outcome != 'pending' AND regime IS NOT NULL
            GROUP BY regime
        """).fetchall()
        conn.close()
        return {dict(r)["regime"]: dict(r) for r in rows}

    def get_performance_by_confidence(self) -> Dict[str, Dict]:
        """Get performance breakdown by confidence bucket."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                CASE
                    WHEN confidence >= 0.8 THEN 'high'
                    WHEN confidence >= 0.6 THEN 'medium'
                    ELSE 'low'
                END as confidence_bucket,
                COUNT(*) as count,
                AVG(CASE WHEN actual_pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(COALESCE(actual_pnl, pnl)) as avg_pnl
            FROM trades
            WHERE outcome IS NOT NULL AND outcome != 'pending' AND confidence IS NOT NULL
            GROUP BY confidence_bucket
        """).fetchall()
        conn.close()
        return {dict(r)["confidence_bucket"]: dict(r) for r in rows}

    def get_winning_patterns(self, limit: int = 10) -> List[Dict]:
        """Extract winning trade patterns for learning."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM trades
            WHERE outcome = 'win' AND COALESCE(actual_pnl, pnl) > 0
            ORDER BY COALESCE(actual_pnl, pnl) DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def save_performance_snapshot(self, stats: Dict):
        """Save a performance snapshot for trend analysis."""
        conn = sqlite3.connect(self.db_path)
        regime_perf = json.dumps(self.get_performance_by_regime())
        conn.execute("""
            INSERT INTO performance_snapshots (timestamp, total_trades, win_rate, avg_pnl, max_drawdown, sharpe_ratio, regime_performance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            stats.get("total", 0),
            stats.get("win_rate", 0),
            stats.get("avg_pnl", 0),
            stats.get("max_dd", 0),
            stats.get("sharpe", 0),
            regime_perf,
        ))
        conn.commit()
        conn.close()

    def get_all_errors(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM errors ORDER BY timestamp DESC LIMIT 50").fetchall()
        conn.close()
        return [dict(r) for r in rows]
