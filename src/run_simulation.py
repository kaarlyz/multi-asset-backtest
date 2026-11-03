#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.bot.telegram_service import TelegramBotService
from src.chronos.db import ChronosDB
from src.chronos.replay import HistoricalReplayEngine
from src.engine.portfolio import SimPortfolio
from src.engine.runner import SimulationRunner
from src.orchestrator import MultiAgentOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5 Interactive Backtesting & Simulation Runner."
    )
    parser.add_argument(
        "--mode",
        choices=["autonomous", "interactive"],
        default="autonomous",
        help="Execution mode: autonomous or interactive (human-in-the-loop Telegram).",
    )
    parser.add_argument(
        "--asset",
        default="BTCUSDT",
        help="Trading pair / asset symbol (default: BTCUSDT).",
    )
    parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Replay start time in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default="2024-03-31",
        help="Replay end time in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Bar interval (e.g., 1d, 1h).",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Delay in seconds between simulation steps for visual observation.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10000.0,
        help="Initial simulated cash (default: 10,000 USDT).",
    )
    default_db = str(project_root / "data" / "chronos.duckdb")
    parser.add_argument(
        "--db-path",
        default=default_db,
        help=f"DuckDB database path (default: {default_db}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"🚀 Initializing Chronos Backtest Engine [{args.mode.upper()}]...")
    print(f"Asset: {args.asset} | Interval: {args.interval} | Range: {args.start} to {args.end}")

    # 1. Initialize DB
    db = ChronosDB(args.db_path)
    base_asset = args.asset.replace("USDT", "")
    db.insert_asset(
        asset_id=f"CRYPTO_{base_asset}",
        asset_class="CRYPTO",
        symbol=args.asset,
        base_asset=base_asset,
        quote_asset="USDT",
        is_active=True,
    )

    # 2. Locate market data parquet
    market_dir = project_root / "data" / "parquet" / "market" / "CRYPTO" / args.asset
    market_path = market_dir / f"interval={args.interval}"
    if not market_path.exists():
        market_path = market_dir

    # 3. Setup Replay Engine
    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

    replay_engine = HistoricalReplayEngine(
        db=db,
        symbol=args.asset,
        start_time=start_dt,
        end_time=end_dt,
        step_size=args.interval,
        interval=args.interval,
        market_data=market_path if market_path.exists() else None,
    )

    # 4. Telegram Service
    telegram_service = TelegramBotService(
        replay_engine=replay_engine,
        db=db,
    )

    # 5. Orchestrator, Portfolio, and Runner
    orchestrator = MultiAgentOrchestrator(
        replay_engine=replay_engine,
        db=db,
    )

    portfolio = SimPortfolio(
        initial_cash=args.initial_cash,
    )

    runner = SimulationRunner(
        replay_engine=replay_engine,
        orchestrator=orchestrator,
        portfolio=portfolio,
        telegram_service=telegram_service,
        mode=args.mode,
        step_delay=args.step_delay,
    )

    # 6. Execute Simulation Loop
    summary = runner.run()

    print("\n" + "=" * 55)
    print(f"📊 BACKTEST RESULTS [{args.mode.upper()}] - {args.asset}")
    print("=" * 55)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:22}: {v:,.2f}")
        else:
            print(f"  {k:22}: {v}")
    print("=" * 55)

    db.close()


if __name__ == "__main__":
    main()
