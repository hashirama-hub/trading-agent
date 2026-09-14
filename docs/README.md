# LLM Agent Trading System

## Overview

An autonomous trading system powered by LLM agents, executing trades on Binance testnet with risk management and real-time dashboard.

## Architecture

- **agent-core**: LLM agent orchestration (DeepSeek)
- **executor**: Order execution via Binance API
- **dashboard-api**: REST API for the web UI
- **dashboard-ui**: React-based trading dashboard

## Quick Start

```bash
cp .env.example .env
# Fill in your API keys
docker compose up --build
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Redis | 6379 | Cache & message broker |
| ChromaDB | 8000 | Vector store for memory |
| TimescaleDB | 5432 | Time-series data |
| Dashboard API | 8000 | REST API |
| Dashboard UI | 3000 | Web interface |