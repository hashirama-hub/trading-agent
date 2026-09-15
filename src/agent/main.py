import asyncio
import logging
import os
import sys

from src.agent.graph import build_agent_graph
from src.tools.binance_client import BinanceClient, PortfolioState as BSPortfolioState
from src.tools.market_data import get_market_data, get_multi_timeframe_data, get_portfolio_state as get_market_portfolio
from src.tools.technical_analysis import calculate_technicals, calculate_multi_timeframe_technicals, binance_client as ta_bc
from src.risk.guard import RiskGuard
from src.memory.store import AgentMemory
from src.executor.paper_trading import PaperTradingEngine
from src.utils.logging_config import setup_logging
from src.utils.health_check import HealthCheck

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    # Configuration
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4.1-flash")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    binance_testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    loop_interval = int(os.getenv("AGENT_LOOP_INTERVAL", "60"))
    max_iterations = int(os.getenv("MAX_ITERATIONS_PER_LOOP", "10"))

    # Initialize Binance client
    binance_client = BinanceClient(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
        testnet=binance_testnet,
    )
    await binance_client.connect()

    # Set module-level client references for tools
    import src.tools.market_data as md_module
    import src.tools.technical_analysis as ta_module
    md_module.binance_client = binance_client
    ta_module.binance_client = binance_client

    # Initialize memory
    memory = AgentMemory(
        db_path=os.getenv("DATABASE_URL", "data/trade_journal.db"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
    )

    # Initialize portfolio and risk guard
    portfolio_state = BSPortfolioState(equity=10000, positions=[], daily_pnl=0, available_margin=5000, max_drawdown_today=0)
    risk_guard = RiskGuard(portfolio_state)

    # Initialize paper trading engine
    paper_engine = PaperTradingEngine(initial_equity=portfolio_state.equity)

    # Import and register trading tools for LangGraph
    from src.agent.nodes import get_market_data as tool_get_market_data, get_portfolio_state as tool_get_portfolio_state, place_order
    tools = [tool_get_market_data, tool_get_portfolio_state, place_order]

    # Build agent graph
    graph = build_agent_graph(
        tools=tools,
        memory=memory,
        risk_guard=risk_guard,
        deepseek_model=deepseek_model,
        deepseek_api_key=deepseek_api_key,
        max_iterations=max_iterations,
    )

    # Health check
    health = HealthCheck()

    logger.info(f"Trading Agent started: {deepseek_model}")
    logger.info(f"Binance testnet: {binance_testnet}")
    logger.info(f"Loop interval: {loop_interval}s, Max iterations: {max_iterations}")
    logger.info(f"Risk: max {int(float(os.getenv('MAX_RISK_PER_TRADE', '0.10'))*100)}%/trade, max DD {int(float(os.getenv('MAX_DAILY_LOSS', '0.05'))*100)}%")

    # Main loop
    while True:
        try:
            # Get current portfolio
            portfolio = paper_engine.get_portfolio_state()
            risk_guard.portfolio = portfolio

            # Get multi-timeframe market data
            market_data = {}
            symbols = ["BTCUSDT", "ETHUSDT"]
            for symbol in symbols:
                try:
                    mtf_data = await get_multi_timeframe_data(symbol)
                    mtf_indicators = await calculate_multi_timeframe_technicals(symbol)

                    market_data[symbol] = {
                        "close": mtf_data.get("1h", list(mtf_data.values())[-1]).close if mtf_data else 0,
                        "multi_timeframe": {
                            tf: snap.dict() for tf, snap in mtf_data.items()
                        },
                        "technicals": mtf_indicators.current.dict(),
                        "htf_trend": mtf_indicators.htf_trend,
                        "htf_regime": mtf_indicators.htf_regime,
                        "signal_alignment": mtf_indicators.signal_alignment,
                        "mtf_summary": mtf_indicators.summary,
                    }
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")

            # Build state
            initial_state = {
                "market_data": market_data,
                "portfolio": {
                    "equity": portfolio.equity,
                    "available_margin": portfolio.available_margin,
                    "positions": portfolio.positions,
                    "daily_pnl": portfolio.daily_pnl,
                    "max_drawdown_today": portfolio.max_drawdown_today,
                },
                "memory": {},
                "current_plan": None,
                "iteration": 0,
                "max_iterations": max_iterations,
                "last_decision": None,
                "errors": [],
            }

            # Run agent graph
            try:
                result = await graph.ainvoke(initial_state)
            except Exception as e:
                logger.error(f"Agent graph error: {e}")
                memory.log_error(str(e))

            # Update portfolio
            portfolio = paper_engine.get_portfolio_state()
            health.check()

            await asyncio.sleep(loop_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down trading agent...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            memory.log_error(str(e))
            await asyncio.sleep(30)

    await binance_client.close()
    logger.info("Trading agent stopped.")


if __name__ == "__main__":
    asyncio.run(main())
