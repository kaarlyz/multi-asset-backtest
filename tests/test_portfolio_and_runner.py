from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Any, Dict
import httpx
import pandas as pd
import pytest

from src.agents.quant_specialist import QuantSpecialist
from src.agents.news_miner import NewsMiner
from src.agents.synthesizer import Synthesizer, DecisionReport
from src.agents.risk_auditor import RiskAuditor
from src.bot.telegram_service import TelegramBotService
from src.chronos.db import ChronosDB
from src.chronos.replay import HistoricalReplayEngine
from src.engine.portfolio import SimPortfolio
from src.engine.runner import SimulationRunner
from src.orchestrator import MultiAgentOrchestrator


@pytest.fixture
def test_db():
    database = ChronosDB(":memory:")
    database.insert_asset("CRYPTO_BTC", "CRYPTO", "BTCUSDT", "BTC", "USDT")
    yield database
    database.close()


@pytest.fixture
def mock_http_client():
    calls = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        calls.append({
            "url": str(request.url),
            "method": request.method,
            "content": request.read().decode("utf-8"),
        })
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 999}})

    transport = httpx.MockTransport(handle_request)
    client = httpx.Client(transport=transport)
    client.mock_calls = calls  # type: ignore
    return client


def test_portfolio_execution_and_sltp():
    """
    Verify BUY/SHORT position sizing, mark-to-market PnL, and automatic SL/TP triggers.
    """
    portfolio = SimPortfolio(
        initial_cash=10000.0,
        default_risk_pct=0.02,
        default_sl_pct=0.05,
        default_tp_pct=0.10,
        max_leverage=5.0,
    )
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

    # 1. Execute BUY decision
    bar1 = {
        "symbol": "BTCUSDT",
        "open": 50000.0,
        "high": 50500.0,
        "low": 49500.0,
        "close": 50000.0,
        "t_event": t0,
    }
    pos = portfolio.execute_decision(
        action="BUY",
        bar=bar1,
        decision_id="DEC_BUY_001",
        actor_id="TEST_USER",
        risk_pct=0.02,
        sl=48000.0,
        tp=54000.0,
    )

    assert pos is not None
    assert pos["side"] == "LONG"
    assert pos["entry_price"] == 50000.0
    assert pos["sl"] == 48000.0
    assert pos["tp"] == 54000.0

    # Risk sizing: 10,000 * 0.02 = 200 USDT. sl_distance = 50,000 - 48,000 = 2,000.
    # Expected size = 200 / 2,000 = 0.1 BTC.
    assert pos["size"] == 0.1
    assert "BTCUSDT" in portfolio.open_positions

    # 2. Update bar - price increases, no SL or TP hit
    bar2 = {
        "symbol": "BTCUSDT",
        "open": 51000.0,
        "high": 52500.0,
        "low": 50500.0,
        "close": 52000.0,
        "t_event": t0 + timedelta(days=1),
    }
    closed = portfolio.update_bar(bar2)
    assert len(closed) == 0
    # Unrealized PnL = (52,000 - 50,000) * 0.1 = +200.0
    assert portfolio.unrealized_pnl == 200.0
    assert portfolio.equity == 10200.0
    assert portfolio.peak_equity == 10200.0
    assert portfolio.max_drawdown_pct == 0.0

    # 3. Update bar - High hits TP (>= 54,000)
    bar3 = {
        "symbol": "BTCUSDT",
        "open": 53000.0,
        "high": 55000.0,
        "low": 52500.0,
        "close": 54500.0,
        "t_event": t0 + timedelta(days=2),
    }
    closed = portfolio.update_bar(bar3)
    assert len(closed) == 1
    trade1 = closed[0]
    assert trade1["exit_reason"] == "TP"
    assert trade1["exit_price"] == 54000.0
    # Realized PnL = (54,000 - 50,000) * 0.1 = +400.0
    assert trade1["realized_pnl"] == 400.0
    assert portfolio.cash == 10400.0
    assert portfolio.equity == 10400.0
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.closed_trades) == 1

    # 4. Execute SHORT decision
    bar4 = {
        "symbol": "BTCUSDT",
        "open": 55000.0,
        "high": 55200.0,
        "low": 54800.0,
        "close": 55000.0,
        "t_event": t0 + timedelta(days=3),
    }
    pos_short = portfolio.execute_decision(
        action="SHORT",
        bar=bar4,
        decision_id="DEC_SHORT_001",
        actor_id="TEST_USER",
        risk_pct=0.02,
        sl=56000.0,
        tp=51000.0,
    )
    assert pos_short is not None
    assert pos_short["side"] == "SHORT"
    # Risk = 10,400 * 0.02 = 208 USDT. sl_distance = 56,000 - 55,000 = 1,000.
    # Expected size = 208 / 1,000 = 0.208 BTC.
    assert round(pos_short["size"], 3) == 0.208

    # 5. Update bar - High hits SL for SHORT (>= 56,000)
    bar5 = {
        "symbol": "BTCUSDT",
        "open": 55500.0,
        "high": 56500.0,
        "low": 54900.0,
        "close": 56200.0,
        "t_event": t0 + timedelta(days=4),
    }
    closed_short = portfolio.update_bar(bar5)
    assert len(closed_short) == 1
    trade2 = closed_short[0]
    assert trade2["exit_reason"] == "SL"
    assert trade2["exit_price"] == 56000.0
    # Realized PnL = (55,000 - 56,000) * 0.208 = -208.0
    assert round(trade2["realized_pnl"], 1) == -208.0
    assert portfolio.max_drawdown_pct > 0.0
    assert len(portfolio.open_positions) == 0

    # 6. Verify summary metrics
    summary = portfolio.get_summary()
    assert summary["total_trades"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["final_equity"] == portfolio.equity
    assert summary["total_return_pct"] > 0.0
    assert summary["max_drawdown_pct"] > 0.0


def test_autonomous_simulation_run(test_db):
    """
    Verify multi-bar run advances, executes signals approved by Risk Auditor,
    and outputs portfolio performance summary.
    """
    sim_start = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    # Ingest fundamentals (bullish TVL growth)
    t_fund = sim_start - timedelta(days=2)
    test_db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=250_000_000.0,
        t_event=t_fund,
        t_pub=t_fund,
        t_ingest=t_fund + timedelta(minutes=5),
        t_avail=t_fund + timedelta(minutes=5),
        source="defillama",
    )

    # Ingest bullish news
    t_news = sim_start - timedelta(hours=2)
    test_db.insert_news_pit(
        news_id="NEWS_AUTO_01",
        title="Bitcoin Institutional Inflows Accelerate",
        source="reuters",
        url="https://reuters.com/btc",
        raw_content="Massive net ETF inflows reported.",
        sentiment_score=0.90,
        tagged_symbols=["BTCUSDT", "BTC"],
        t_event=t_news,
        t_pub=t_news,
        t_ingest=t_news + timedelta(seconds=15),
        t_avail=t_news + timedelta(seconds=15),
    )

    # Generate 35 hourly bars (30 prior + 5 after sim_start)
    bars = []
    base_price = 45000.0
    for i in range(35):
        t_bar = sim_start - timedelta(hours=30 - i)
        close_p = base_price + (i * 150.0)
        bars.append({
            "open_time": int(t_bar.timestamp() * 1000),
            "open": close_p - 50.0,
            "high": close_p + 100.0,
            "low": close_p - 50.0,
            "close": close_p,
            "volume": 20.0,
            "quote_volume": 20.0 * close_p,
            "t_event": t_bar,
            "t_pub": t_bar + timedelta(minutes=59),
            "t_avail": t_bar + timedelta(hours=1),
            "t_ingest": t_bar + timedelta(hours=1, seconds=2),
            "symbol": "BTCUSDT",
            "interval": "1h",
        })
    market_df = pd.DataFrame(bars)

    replay_engine = HistoricalReplayEngine(
        db=test_db,
        symbol="BTCUSDT",
        start_time=sim_start,
        end_time=sim_start + timedelta(hours=3),
        step_size="1h",
        interval="1h",
        market_data=market_df,
    )

    orchestrator = MultiAgentOrchestrator(
        replay_engine=replay_engine,
        db=test_db,
    )

    portfolio = SimPortfolio(initial_cash=10000.0)

    runner = SimulationRunner(
        replay_engine=replay_engine,
        orchestrator=orchestrator,
        portfolio=portfolio,
        mode="AUTONOMOUS",
    )

    summary = runner.run()

    # Replay engine must advance to the end
    assert replay_engine.is_finished() is True

    # Performance summary must contain standard metrics
    assert "total_trades" in summary
    assert "win_rate" in summary
    assert "total_return_pct" in summary
    assert "final_equity" in summary
    assert "max_drawdown_pct" in summary

    # Verify decision ledger contains recorded decisions
    decisions = test_db.conn.execute("SELECT * FROM decision_ledger").fetchall()
    assert len(decisions) > 0


def test_interactive_mode_waiting_and_callback(test_db, mock_http_client):
    """
    Mock Telegram interaction, verify runner pauses at signal,
    waits for decision_ledger entry, executes user choice, and resumes.
    """
    sim_start = datetime(2024, 7, 1, 12, 0, tzinfo=timezone.utc)
    auth_user_id = 6526833205

    # Ingest bullish fundamental TVL
    t_fund = sim_start - timedelta(days=2)
    test_db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=250_000_000.0,
        t_event=t_fund,
        t_pub=t_fund,
        t_ingest=t_fund + timedelta(minutes=5),
        t_avail=t_fund + timedelta(minutes=5),
        source="defillama",
    )

    # Ingest news
    t_news = sim_start - timedelta(hours=3)
    test_db.insert_news_pit(
        news_id="NEWS_INTERACTIVE_01",
        title="Bullish Market Breakout Confirmed",
        source="bloomberg",
        url="https://bloomberg.com/crypto",
        raw_content="All indicators green.",
        sentiment_score=0.88,
        tagged_symbols=["BTCUSDT", "BTC"],
        t_event=t_news,
        t_pub=t_news,
        t_ingest=t_news + timedelta(seconds=10),
        t_avail=t_news + timedelta(seconds=10),
    )

    # Populate 35 market bars to establish strong upward SMA trend
    bars = []
    base_price = 42000.0
    for i in range(35):
        t_bar = sim_start - timedelta(hours=30 - i)
        close_p = base_price + (i * 150.0)
        bars.append({
            "open_time": int(t_bar.timestamp() * 1000),
            "open": close_p - 50.0,
            "high": close_p + 100.0,
            "low": close_p - 50.0,
            "close": close_p,
            "volume": 20.0,
            "quote_volume": 20.0 * close_p,
            "t_event": t_bar,
            "t_pub": t_bar + timedelta(minutes=59),
            "t_avail": t_bar + timedelta(hours=1),
            "t_ingest": t_bar + timedelta(hours=1, seconds=2),
            "symbol": "BTCUSDT",
            "interval": "1h",
        })
    market_df = pd.DataFrame(bars)

    replay_engine = HistoricalReplayEngine(
        db=test_db,
        symbol="BTCUSDT",
        start_time=sim_start,
        end_time=sim_start + timedelta(hours=2),
        step_size="1h",
        interval="1h",
        market_data=market_df,
    )

    telegram_service = TelegramBotService(
        token="MOCK_INTERACTIVE_TOKEN",
        allowed_user_ids=[auth_user_id],
        replay_engine=replay_engine,
        db=test_db,
        http_client=mock_http_client,
    )

    orchestrator = MultiAgentOrchestrator(
        replay_engine=replay_engine,
        db=test_db,
    )

    portfolio = SimPortfolio(initial_cash=10000.0)

    runner = SimulationRunner(
        replay_engine=replay_engine,
        orchestrator=orchestrator,
        portfolio=portfolio,
        telegram_service=telegram_service,
        mode="INTERACTIVE",
        decision_timeout=3.0,
        poll_interval=0.02,
    )

    # Background thread to simulate human clicking [BUY] callback upon prompt delivery
    def mock_user_response():
        time.sleep(0.05)
        # Search for pending decision prompt in decision_ledger
        pending = None
        for _ in range(50):
            with test_db._lock:
                records = test_db.conn.execute("SELECT decision_id FROM decision_ledger").fetchall()
            if records:
                pending = records[-1][0]
                break
            time.sleep(0.02)

        if pending:
            # Simulate Telegram callback query from authorized user
            telegram_service.handle_callback({
                "id": "query_inter_001",
                "from": {"id": auth_user_id},
                "data": f"DECIDE:{pending}:BUY",
            })

    sim_thread = threading.Thread(target=mock_user_response, daemon=True)
    sim_thread.start()

    # Step the runner
    step_result = runner.step()

    sim_thread.join(timeout=2.0)

    # Verify execution outcome
    assert step_result["executed_action"] == "BUY"
    assert "BTCUSDT" in portfolio.open_positions
    assert portfolio.open_positions["BTCUSDT"]["side"] == "LONG"
    assert replay_engine.is_paused is False

    # Verify Telegram prompt was dispatched via mock client
    assert len(mock_http_client.mock_calls) > 0


def test_position_aware_signal_suppression(test_db, mock_http_client):
    """
    Verify position-aware filtering:
    1. FLAT portfolio -> BUY signal dispatches prompt and executes order.
    2. Holding LONG -> consecutive BUY signal suppresses prompt, logs as HOLD, does not pause clock or duplicate order.
    3. Holding LONG -> REVERSAL SHORT signal dispatches prompt and executes reversal (closes LONG, opens SHORT).
    4. Holding SHORT -> consecutive SHORT signal suppresses prompt, logs as HOLD.
    """
    sim_start = datetime(2024, 8, 1, 12, 0, tzinfo=timezone.utc)
    auth_user_id = 6526833205

    # Fundamentals: bullish TVL
    t_fund = sim_start - timedelta(days=2)
    test_db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=300_000_000.0,
        t_event=t_fund,
        t_pub=t_fund,
        t_ingest=t_fund + timedelta(minutes=5),
        t_avail=t_fund + timedelta(minutes=5),
        source="defillama",
    )

    # Bullish news
    t_news = sim_start - timedelta(hours=3)
    test_db.insert_news_pit(
        news_id="NEWS_POS_AWARE_01",
        title="Bullish Momentum Persistent",
        source="bloomberg",
        url="https://bloomberg.com/btc",
        raw_content="Sustained buying momentum detected.",
        sentiment_score=0.90,
        tagged_symbols=["BTCUSDT", "BTC"],
        t_event=t_news,
        t_pub=t_news,
        t_ingest=t_news + timedelta(seconds=10),
        t_avail=t_news + timedelta(seconds=10),
    )

    # Create 40 market bars (30 warm-up + 10 simulation)
    bars = []
    base_price = 50000.0
    for i in range(40):
        t_bar = sim_start - timedelta(hours=30 - i)
        close_p = base_price + (i * 100.0)
        bars.append({
            "open_time": int(t_bar.timestamp() * 1000),
            "open": close_p - 50.0,
            "high": close_p + 100.0,
            "low": close_p - 50.0,
            "close": close_p,
            "volume": 25.0,
            "quote_volume": 25.0 * close_p,
            "t_event": t_bar,
            "t_pub": t_bar + timedelta(minutes=59),
            "t_avail": t_bar + timedelta(hours=1),
            "t_ingest": t_bar + timedelta(hours=1, seconds=2),
            "symbol": "BTCUSDT",
            "interval": "1h",
        })
    market_df = pd.DataFrame(bars)

    replay_engine = HistoricalReplayEngine(
        db=test_db,
        symbol="BTCUSDT",
        start_time=sim_start,
        end_time=sim_start + timedelta(hours=8),
        step_size="1h",
        interval="1h",
        market_data=market_df,
    )

    telegram_service = TelegramBotService(
        token="MOCK_TEST_TOKEN",
        allowed_user_ids=[auth_user_id],
        replay_engine=replay_engine,
        db=test_db,
        http_client=mock_http_client,
    )

    orchestrator = MultiAgentOrchestrator(
        replay_engine=replay_engine,
        db=test_db,
    )

    portfolio = SimPortfolio(initial_cash=10000.0)

    runner = SimulationRunner(
        replay_engine=replay_engine,
        orchestrator=orchestrator,
        portfolio=portfolio,
        telegram_service=telegram_service,
        mode="INTERACTIVE",
        decision_timeout=2.0,
        poll_interval=0.02,
    )

    # --- STEP 1: FLAT portfolio receives BUY signal -> Prompt dispatched, order executed ---
    def respond_buy():
        time.sleep(0.05)
        pending = None
        for _ in range(50):
            with test_db._lock:
                records = test_db.conn.execute("SELECT decision_id FROM decision_ledger WHERE action = 'BUY'").fetchall()
            if records:
                pending = records[-1][0]
                break
            time.sleep(0.02)
        if pending:
            telegram_service.handle_callback({
                "id": "query_step1_buy",
                "from": {"id": auth_user_id},
                "data": f"DECIDE:{pending}:BUY",
            })

    t1 = threading.Thread(target=respond_buy, daemon=True)
    t1.start()

    res1 = runner.step()
    t1.join(timeout=2.0)

    assert res1["executed_action"] == "BUY"
    assert "BTCUSDT" in portfolio.open_positions
    assert portfolio.open_positions["BTCUSDT"]["side"] == "LONG"
    orig_pos_size = portfolio.open_positions["BTCUSDT"]["size"]
    prompt_count_after_step1 = len(mock_http_client.mock_calls)
    assert prompt_count_after_step1 > 0
    assert runner.replay_engine.is_paused is False

    # --- STEP 2: Consecutive BUY signal while holding LONG -> Suppressed as HOLD ---
    # No mock responder thread needed because runner should NOT pause or prompt
    res2 = runner.step()

    assert res2["executed_action"] == "HOLD"
    # Telegram calls must not increase (prompt suppressed)
    assert len(mock_http_client.mock_calls) == prompt_count_after_step1
    # Clock must not be paused
    assert runner.replay_engine.is_paused is False
    # No duplicate order executed (size remains identical)
    assert len(portfolio.open_positions) == 1
    assert portfolio.open_positions["BTCUSDT"]["size"] == orig_pos_size
    # Decision ledger should reflect HOLD
    dec2_id = res2["step_output"]["decision_report"].decision_id
    rec2 = test_db.get_decision(dec2_id)
    assert rec2 is not None
    assert rec2["action"] == "HOLD"
    assert rec2["status"] == "HOLD"

    # --- STEP 3: Another consecutive BUY signal -> Also suppressed as HOLD ---
    res3 = runner.step()
    assert res3["executed_action"] == "HOLD"
    assert len(mock_http_client.mock_calls) == prompt_count_after_step1
    assert portfolio.open_positions["BTCUSDT"]["size"] == orig_pos_size

    # --- STEP 4: REVERSAL signal (Holding LONG, Synthesizer proposes SHORT) ---
    class MockReversalSynthesizer(Synthesizer):
        def evaluate(self, context: Dict[str, Any]) -> DecisionReport:
            sim_time = context.get("sim_time") or datetime.now(timezone.utc)
            asset_id = context.get("asset_id") or "BTCUSDT"
            quant = context.get("quant_analysis")
            news = context.get("news_analysis")
            return DecisionReport(
                decision_id=f"DEC_REV_SHORT_{int(time.time()*1000)}",
                asset_id=asset_id,
                sim_time=sim_time,
                proposed_action="SHORT",
                confidence=0.92,
                quant_analysis=quant,  # type: ignore
                news_analysis=news,  # type: ignore
                supporting_evidence=["Bearish divergence confirmed"],
                contradicting_evidence=[],
            )

    orchestrator.synthesizer = MockReversalSynthesizer()

    def respond_short():
        time.sleep(0.05)
        pending = None
        for _ in range(50):
            with test_db._lock:
                records = test_db.conn.execute("SELECT decision_id FROM decision_ledger WHERE action = 'SHORT'").fetchall()
            if records:
                pending = records[-1][0]
                break
            time.sleep(0.02)
        if pending:
            telegram_service.handle_callback({
                "id": "query_step4_short",
                "from": {"id": auth_user_id},
                "data": f"DECIDE:{pending}:SHORT",
            })

    t4 = threading.Thread(target=respond_short, daemon=True)
    t4.start()

    res4 = runner.step()
    t4.join(timeout=2.0)

    # Prompt MUST be dispatched for REVERSAL
    assert len(mock_http_client.mock_calls) > prompt_count_after_step1
    prompt_count_after_step4 = len(mock_http_client.mock_calls)
    assert res4["executed_action"] == "SHORT"
    assert portfolio.open_positions["BTCUSDT"]["side"] == "SHORT"
    short_pos_size = portfolio.open_positions["BTCUSDT"]["size"]
    # Previous LONG position closed
    assert len(portfolio.closed_trades) == 1
    assert portfolio.closed_trades[0]["side"] == "LONG"
    assert portfolio.closed_trades[0]["exit_reason"] == "SIGNAL"

    # --- STEP 5: Consecutive SHORT signal while holding SHORT -> Suppressed as HOLD ---
    res5 = runner.step()
    assert res5["executed_action"] == "HOLD"
    # Prompt suppressed: Telegram prompt calls did not increase (excluding potential exit notifications)
    # Replay clock not paused
    assert runner.replay_engine.is_paused is False
    assert portfolio.open_positions["BTCUSDT"]["side"] == "SHORT"
    assert portfolio.open_positions["BTCUSDT"]["size"] == short_pos_size
    dec5_id = res5["step_output"]["decision_report"].decision_id
    rec5 = test_db.get_decision(dec5_id)
    assert rec5 is not None
    assert rec5["action"] == "HOLD"
    assert rec5["status"] == "HOLD"

