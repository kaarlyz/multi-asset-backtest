from datetime import datetime, timedelta, timezone
import pytest
import pandas as pd

from src.agents.quant_specialist import QuantSpecialist, QuantitativeAnalysis
from src.agents.news_miner import NewsMiner, NewsSentimentAnalysis
from src.agents.synthesizer import Synthesizer, DecisionReport
from src.agents.risk_auditor import RiskAuditor, AuditVerdict
from src.bot.telegram_service import TelegramBotService
from src.chronos.db import ChronosDB
from src.chronos.replay import HistoricalReplayEngine, TemporalLeakageError
from src.orchestrator import MultiAgentOrchestrator


@pytest.fixture
def test_db():
    db = ChronosDB(":memory:")
    db.insert_asset("CRYPTO_BTC", "CRYPTO", "BTCUSDT", "BTC", "USDT")
    yield db
    db.close()


@pytest.fixture
def base_sim_time():
    return datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_quant_specialist_metrics(test_db, base_sim_time):
    """
    Verify calculation of TVL momentum, volatility, and trend indicators without look-ahead bias.
    """
    sim_time = base_sim_time

    # Insert TVL at t - 35d, t - 10d, t - 1d, and future t + 2d
    t_35d = sim_time - timedelta(days=35)
    t_10d = sim_time - timedelta(days=10)
    t_1d = sim_time - timedelta(days=1)
    t_future = sim_time + timedelta(days=2)

    test_db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=100_000_000.0,
        t_event=t_35d,
        t_pub=t_35d,
        t_ingest=t_35d + timedelta(minutes=5),
        t_avail=t_35d + timedelta(minutes=5),
        source="defillama",
    )
    test_db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=120_000_000.0,
        t_event=t_10d,
        t_pub=t_10d,
        t_ingest=t_10d + timedelta(minutes=5),
        t_avail=t_10d + timedelta(minutes=5),
        source="defillama",
    )
    test_db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=150_000_000.0,
        t_event=t_1d,
        t_pub=t_1d,
        t_ingest=t_1d + timedelta(minutes=5),
        t_avail=t_1d + timedelta(minutes=5),
        source="defillama",
    )
    # Future TVL that MUST NOT be seen
    test_db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=999_999_999.0,
        t_event=t_future,
        t_pub=t_future,
        t_ingest=t_future + timedelta(minutes=5),
        t_avail=t_future + timedelta(minutes=5),
        source="defillama",
    )

    # Build 25 historical hourly bars leading up to sim_time
    bars = []
    base_price = 40000.0
    for i in range(25):
        bar_time = sim_time - timedelta(hours=25 - i)
        close_p = base_price + (i * 200.0)
        bars.append({
            "symbol": "BTCUSDT",
            "t_event": bar_time,
            "t_avail": bar_time + timedelta(seconds=10),
            "open": close_p - 100.0,
            "high": close_p + 150.0,
            "low": close_p - 150.0,
            "close": close_p,
            "volume": 20.0,
            "quote_volume": 20.0 * close_p,
        })

    agent = QuantSpecialist()
    context = {
        "sim_time": sim_time,
        "asset_id": "CRYPTO_BTC",
        "symbol": "BTCUSDT",
        "bars": bars,
        "db": test_db,
    }

    analysis = agent.evaluate(context)

    # Invariant: Current TVL is 150M (not the future 999M)
    assert analysis.metrics["tvl"] == 150_000_000.0

    # Invariant: 7d TVL momentum is (150M - 120M)/120M = +0.25 (25% increase)
    assert pytest.approx(analysis.metrics["tvl_7d_momentum"], rel=1e-3) == 0.25

    # Invariant: 30d TVL momentum is (150M - 100M)/100M = +0.50 (50% increase)
    assert pytest.approx(analysis.metrics["tvl_30d_momentum"], rel=1e-3) == 0.50

    # Invariant: SMA trend and volatility computed
    assert analysis.metrics["sma_20"] is not None
    assert analysis.metrics["sma_20"] < analysis.metrics["current_price"]
    assert analysis.metrics["realized_volatility"] >= 0.0
    assert analysis.metrics["atr"] > 0.0
    assert analysis.score > 0.0  # Bullish alignment

    # Leakage guard: future bar injected into context must raise TemporalLeakageError
    future_bars = list(bars) + [{
        "symbol": "BTCUSDT",
        "t_event": sim_time + timedelta(hours=1),
        "t_avail": sim_time + timedelta(hours=1),
        "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50000.0,
        "volume": 10.0, "quote_volume": 500000.0,
    }]
    with pytest.raises(TemporalLeakageError):
        agent.evaluate({**context, "bars": future_bars})


def test_news_miner_temporal_guard(test_db, base_sim_time):
    """
    Verify news miner returns ONLY news where t_avail <= sim_time.
    """
    sim_time = base_sim_time

    # 1. Past news available at sim_time - 2 hours
    t_past = sim_time - timedelta(hours=2)
    test_db.insert_news_pit(
        news_id="NEWS_VALID_001",
        title="Institutional Spot Bitcoin ETF Adoption Expands Globally",
        source="bloomberg",
        url="https://bloomberg.com/news/etf",
        raw_content="Regulatory approval confirmed for multiple issuers.",
        sentiment_score=0.85,
        tagged_symbols=["BTC", "BTCUSDT"],
        t_event=t_past - timedelta(minutes=10),
        t_pub=t_past - timedelta(minutes=5),
        t_ingest=t_past,
        t_avail=t_past,
    )

    # 2. Future news available at sim_time + 1 hour (LEAKAGE if accessed)
    t_future = sim_time + timedelta(hours=1)
    test_db.insert_news_pit(
        news_id="NEWS_FUTURE_002",
        title="Major Bridge Exploit Drains 10,000 BTC",
        source="coindesk",
        url="https://coindesk.com/hack",
        raw_content="Exploit underway, massive security breach.",
        sentiment_score=-0.95,
        tagged_symbols=["BTC", "BTCUSDT"],
        t_event=t_future - timedelta(minutes=10),
        t_pub=t_future - timedelta(minutes=5),
        t_ingest=t_future,
        t_avail=t_future,
    )

    miner = NewsMiner()
    context = {
        "sim_time": sim_time,
        "asset_id": "BTC",
        "db": test_db,
    }

    analysis = miner.evaluate(context)

    # Invariant: News miner only retrieved NEWS_VALID_001, never NEWS_FUTURE_002
    assert analysis.news_count == 1
    assert analysis.events[0]["news_id"] == "NEWS_VALID_001"
    assert analysis.event_breakdown["ADOPTION"] >= 1
    assert analysis.score > 0.5  # Bullish sentiment

    # Leakage guard: if news with future t_avail is passed in context, TemporalLeakageError is raised
    with pytest.raises(TemporalLeakageError):
        miner.evaluate({
            "sim_time": sim_time,
            "asset_id": "BTC",
            "news": [
                {
                    "news_id": "FUTURE_ATTEMPT",
                    "title": "Future News",
                    "t_avail": sim_time + timedelta(minutes=10),
                    "sentiment_score": 0.5,
                }
            ],
        })


def test_synthesizer_decision_logic(base_sim_time):
    """
    Verify convergence of bullish quant + bullish news results in BUY,
    divergent evidence lowers confidence or gives IGNORE.
    """
    sim_time = base_sim_time
    synthesizer = Synthesizer()

    # Case 1: Bullish convergence (Quant bullish + News bullish)
    quant_bullish = QuantitativeAnalysis(
        asset_id="BTCUSDT",
        sim_time=sim_time,
        score=0.65,
        metrics={"current_price": 45000.0, "volume": 100.0, "quote_volume": 4500000.0},
        signals=["TVL_7D_STRONG_INFLOW", "PRICE_ABOVE_SMA20"],
    )
    news_bullish = NewsSentimentAnalysis(
        asset_id="BTCUSDT",
        sim_time=sim_time,
        score=0.75,
        rolling_sentiment_7d=0.75,
        news_count=3,
        events=[{"title": "Institutional ETF inflow hits record", "sentiment_score": 0.8}],
    )

    report_buy = synthesizer.evaluate({
        "sim_time": sim_time,
        "asset_id": "BTCUSDT",
        "quant_analysis": quant_bullish,
        "news_analysis": news_bullish,
    })

    assert report_buy.proposed_action == "BUY"
    assert report_buy.confidence >= 0.70
    assert len(report_buy.supporting_evidence) > 0

    # Case 2: Bearish convergence (Quant bearish + News bearish)
    quant_bearish = QuantitativeAnalysis(
        asset_id="BTCUSDT",
        sim_time=sim_time,
        score=-0.60,
        metrics={"current_price": 40000.0, "volume": 100.0, "quote_volume": 4000000.0},
        signals=["TVL_7D_OUTFLOW", "PRICE_BELOW_SMA20"],
    )
    news_bearish = NewsSentimentAnalysis(
        asset_id="BTCUSDT",
        sim_time=sim_time,
        score=-0.70,
        rolling_sentiment_7d=-0.70,
        news_count=2,
        events=[{"title": "Regulatory crackdown on major exchange", "sentiment_score": -0.8}],
    )

    report_short = synthesizer.evaluate({
        "sim_time": sim_time,
        "asset_id": "BTCUSDT",
        "quant_analysis": quant_bearish,
        "news_analysis": news_bearish,
    })

    assert report_short.proposed_action == "SHORT"
    assert report_short.confidence >= 0.70

    # Case 3: Divergence (Quant bullish + News severe bearish hack)
    report_divergent = synthesizer.evaluate({
        "sim_time": sim_time,
        "asset_id": "BTCUSDT",
        "quant_analysis": quant_bullish,
        "news_analysis": news_bearish,
    })

    assert report_divergent.proposed_action == "IGNORE"
    assert report_divergent.confidence <= 0.40
    assert len(report_divergent.contradicting_evidence) > 0


def test_risk_auditor_veto_mechanisms(base_sim_time):
    """
    Verify auditor correctly vetoes on high volatility spike, insufficient liquidity, or future leakage attempt.
    """
    sim_time = base_sim_time
    auditor = RiskAuditor(
        min_volume=1.0,
        min_quote_volume=10000.0,
        max_realized_volatility=0.08,
    )

    # Base valid report
    base_quant = QuantitativeAnalysis(
        asset_id="BTCUSDT",
        sim_time=sim_time,
        score=0.70,
        metrics={
            "current_price": 50000.0,
            "volume": 25.0,
            "quote_volume": 1250000.0,
            "realized_volatility": 0.03,
            "atr": 500.0,
        },
    )
    base_news = NewsSentimentAnalysis(
        asset_id="BTCUSDT",
        sim_time=sim_time,
        score=0.60,
        rolling_sentiment_7d=0.60,
        news_count=1,
        events=[{"title": "Adoption surges", "sentiment_score": 0.6, "t_avail": sim_time.isoformat()}],
    )
    valid_report = DecisionReport(
        decision_id="DEC_TEST_001",
        asset_id="BTCUSDT",
        sim_time=sim_time,
        proposed_action="BUY",
        confidence=0.85,
        quant_analysis=base_quant,
        news_analysis=base_news,
        current_price=50000.0,
    )

    # 1. Normal safe condition -> APPROVED
    verdict_safe = auditor.evaluate({
        "sim_time": sim_time,
        "report": valid_report,
    })
    assert verdict_safe.status == "APPROVED"
    assert len(verdict_safe.veto_reasons) == 0

    # 2. Volatility guard: realized_volatility = 0.15 > 0.08 -> VETOED
    hi_vol_quant = QuantitativeAnalysis(
        asset_id="BTCUSDT",
        sim_time=sim_time,
        score=0.70,
        metrics={
            "current_price": 50000.0,
            "volume": 25.0,
            "quote_volume": 1250000.0,
            "realized_volatility": 0.15,
            "atr": 3500.0,
        },
    )
    hi_vol_report = DecisionReport(
        decision_id="DEC_TEST_HIVOL",
        asset_id="BTCUSDT",
        sim_time=sim_time,
        proposed_action="BUY",
        confidence=0.85,
        quant_analysis=hi_vol_quant,
        news_analysis=base_news,
    )
    verdict_vol = auditor.evaluate({
        "sim_time": sim_time,
        "report": hi_vol_report,
    })
    assert verdict_vol.status == "VETOED"
    assert any("VOLATILITY_GUARD_EXCEEDED" in r for r in verdict_vol.veto_reasons)

    # 3. Insufficient liquidity: volume = 0.05, quote_volume = 2500.0 -> VETOED
    low_liq_quant = QuantitativeAnalysis(
        asset_id="BTCUSDT",
        sim_time=sim_time,
        score=0.70,
        metrics={
            "current_price": 50000.0,
            "volume": 0.05,
            "quote_volume": 2500.0,
            "realized_volatility": 0.03,
            "atr": 400.0,
        },
    )
    low_liq_report = DecisionReport(
        decision_id="DEC_TEST_LOWLIQ",
        asset_id="BTCUSDT",
        sim_time=sim_time,
        proposed_action="BUY",
        confidence=0.85,
        quant_analysis=low_liq_quant,
        news_analysis=base_news,
    )
    verdict_liq = auditor.evaluate({
        "sim_time": sim_time,
        "report": low_liq_report,
    })
    assert verdict_liq.status == "VETOED"
    assert any("INSUFFICIENT_LIQUIDITY" in r for r in verdict_liq.veto_reasons)

    # 4. Lookahead bias attempt: report contains news event from future -> VETOED
    future_news = NewsSentimentAnalysis(
        asset_id="BTCUSDT",
        sim_time=sim_time,
        score=0.60,
        rolling_sentiment_7d=0.60,
        news_count=1,
        events=[{
            "title": "Future Leakage Event",
            "sentiment_score": 0.8,
            "t_avail": (sim_time + timedelta(hours=2)).isoformat(),
        }],
    )
    leakage_report = DecisionReport(
        decision_id="DEC_TEST_LEAK",
        asset_id="BTCUSDT",
        sim_time=sim_time,
        proposed_action="BUY",
        confidence=0.85,
        quant_analysis=base_quant,
        news_analysis=future_news,
    )
    verdict_leak = auditor.evaluate({
        "sim_time": sim_time,
        "report": leakage_report,
    })
    assert verdict_leak.status == "VETOED"
    assert any("LOOKAHEAD_BIAS_DETECTED" in r for r in verdict_leak.veto_reasons)


def test_orchestrator_pipeline_end_to_end(test_db, base_sim_time):
    """
    Run a complete simulated step through the entire multi-agent pipeline and assert correct state transitions.
    """
    sim_time = base_sim_time

    # Populate 30 market bars
    bars = []
    base_price = 42000.0
    for i in range(30):
        t_bar = sim_time - timedelta(hours=30 - i)
        close_p = base_price + (i * 150.0)
        bars.append({
            "open_time": int(t_bar.timestamp() * 1000),
            "open": close_p - 50.0,
            "high": close_p + 100.0,
            "low": close_p - 50.0,
            "close": close_p,
            "volume": 15.0,
            "quote_volume": 15.0 * close_p,
            "t_event": t_bar,
            "t_pub": t_bar + timedelta(minutes=59),
            "t_avail": t_bar + timedelta(hours=1),
            "t_ingest": t_bar + timedelta(hours=1, seconds=2),
            "symbol": "BTCUSDT",
            "interval": "1h",
        })
    market_df = pd.DataFrame(bars)

    # Ingest TVL fundamentals
    t_fund = sim_time - timedelta(days=2)
    test_db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=200_000_000.0,
        t_event=t_fund,
        t_pub=t_fund,
        t_ingest=t_fund + timedelta(minutes=5),
        t_avail=t_fund + timedelta(minutes=5),
        source="defillama",
    )

    # Ingest news
    t_news = sim_time - timedelta(hours=3)
    test_db.insert_news_pit(
        news_id="NEWS_PIPELINE_01",
        title="Institutional ETF Inflows Break Records",
        source="bloomberg",
        url="https://bloomberg.com/etf",
        raw_content="Record-setting institutional adoption.",
        sentiment_score=0.85,
        tagged_symbols=["BTCUSDT", "BTC"],
        t_event=t_news,
        t_pub=t_news,
        t_ingest=t_news + timedelta(seconds=30),
        t_avail=t_news + timedelta(seconds=30),
    )

    replay_engine = HistoricalReplayEngine(
        db=test_db,
        symbol="BTCUSDT",
        start_time=sim_time,
        end_time=sim_time + timedelta(days=1),
        step_size="1h",
        market_data=market_df,
    )

    telegram_bot = TelegramBotService(
        token="",  # Test mode
        allowed_user_ids=[123456],
        replay_engine=replay_engine,
        db=test_db,
    )

    orchestrator = MultiAgentOrchestrator(
        replay_engine=replay_engine,
        telegram_service=telegram_bot,
        db=test_db,
    )

    # Step 1: Run pipeline step
    step_result = orchestrator.step(advance=False)

    assert step_result["sim_time"] == sim_time
    assert step_result["quant_analysis"].score > 0.0
    assert step_result["news_analysis"].score > 0.0
    assert step_result["decision_report"].proposed_action == "BUY"
    assert step_result["audit_verdict"].status == "APPROVED"
    assert step_result["dispatched"] is True

    # Assert decision ledger recorded the approved dispatched signal
    dec_id = step_result["decision_report"].decision_id
    assert test_db.has_decision(dec_id)
    saved_rec = test_db.get_decision(dec_id)
    assert saved_rec["action"] == "BUY"
    assert saved_rec["status"] == "SIGNAL_DISPATCHED"

    # Step 2: Now test pipeline with Risk Auditor veto condition
    strict_auditor = RiskAuditor(max_realized_volatility=0.00001)  # Tighter than 4.3e-5 to trigger veto
    orchestrator.risk_auditor = strict_auditor

    veto_result = orchestrator.step(advance=False)
    assert veto_result["audit_verdict"].status == "VETOED"
    assert veto_result["dispatched"] is False

    veto_dec_id = veto_result["decision_report"].decision_id
    assert test_db.has_decision(veto_dec_id)
    veto_rec = test_db.get_decision(veto_dec_id)
    assert veto_rec["status"] == "VETOED"
