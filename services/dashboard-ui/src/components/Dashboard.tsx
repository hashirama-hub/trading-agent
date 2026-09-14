"use client";

import { useEffect, useState } from "react";

interface PortfolioData {
  equity: number;
  daily_pnl: number;
  positions: Array<{ symbol: string; qty: number; entry: number }>;
}

interface AgentLog {
  timestamp: string;
  thought: string;
  action: string;
  result: string;
}

export default function Dashboard() {
  const [portfolio, setPortfolio] = useState<PortfolioData>({
    equity: 0,
    daily_pnl: 0,
    positions: [],
  });
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    fetch("/api/portfolio")
      .then((r) => r.json())
      .then(setPortfolio);

    fetch("/api/agent/log?limit=20")
      .then((r) => r.json())
      .then(setLogs);

    const socket = new WebSocket("ws://localhost:8000/ws");
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "portfolio") setPortfolio(data.payload);
    };
    setWs(socket);

    return () => socket.close();
  }, []);

  return (
    <div className="dashboard min-h-screen bg-gray-900 text-white p-6">
      <h1 className="text-3xl font-bold mb-6">Trading Agent Dashboard</h1>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg text-gray-400">Equity</h2>
          <p className="text-2xl font-bold">${portfolio.equity.toLocaleString()}</p>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg text-gray-400">Daily PnL</h2>
          <p
            className={`text-2xl font-bold ${portfolio.daily_pnl >= 0 ? "text-green-400" : "text-red-400"}`}
          >
            ${portfolio.daily_pnl.toLocaleString()}
          </p>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-lg text-gray-400">Positions</h2>
          <p className="text-2xl font-bold">{portfolio.positions.length}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-xl font-bold mb-4">Positions</h2>
          {portfolio.positions.length === 0 ? (
            <p className="text-gray-500">No open positions</p>
          ) : (
            <ul>
              {portfolio.positions.map((pos, i) => (
                <li key={i} className="flex justify-between py-2 border-b border-gray-700">
                  <span>{pos.symbol}</span>
                  <span>
                    {pos.qty} @ ${pos.entry.toLocaleString()}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-xl font-bold mb-4">Agent Log</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {logs.map((log, i) => (
              <div key={i} className="bg-gray-900 rounded p-3 text-sm">
                <p className="text-gray-500">{log.timestamp}</p>
                <p className="text-blue-400">Action: {log.action}</p>
                <p>{log.thought}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}