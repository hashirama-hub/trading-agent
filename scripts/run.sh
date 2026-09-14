#!/bin/bash
set -e

export DEEPSEEK_API_KEY=$(cat secrets/deepseek_api_key 2>/dev/null || echo "")
export BINANCE_API_KEY=$(cat secrets/binance_api_key 2>/dev/null || echo "")
export BINANCE_API_SECRET=$(cat secrets/binance_api_secret 2>/dev/null || echo "")
export DB_PASSWORD=$(cat secrets/db_password 2>/dev/null || echo "changeme")

docker compose up -d --build
echo "All services started. Dashboard at http://localhost:3000"