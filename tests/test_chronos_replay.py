from datetime import datetime, timedelta, timezone
import pytest
import pandas as pd

from src.chronos.db import ChronosDB
from src.chronos.replay import HistoricalReplayEngine, TemporalLeakageError


@pytest.fixture
def memory_db():
    database = ChronosDB(":memory:")
    database.insert_asset("CRYPTO_BTC", "CRYPTO", "BTCUSDT", "BTC", "USDT")

    t_ingest = datetime.now(timezone.utc)
    # Fundamental metric available at 2024-01-01 10:05
    t_event1 = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    database.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=100.0,
        t_event=t_event1,
        t_pub=t_event1,
        t_ingest=t_ingest,
        t_avail=t_event1 + timedelta(minutes=5),
        source="defillama",
    )
    # Fundamental metric available at 2024-01-01 12:05
    t_event2 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    database.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=200.0,
        t_event=t_event2,
        t_pub=t_event2,
        t_ingest=t_ingest,
        t_avail=t_event2 + timedelta(minutes=5),
        source="defillama",
    )

    yield database
    database.close()


@pytest.fixture
def sample_market_df():
    # 3 hourly bars
    # Bar 0: 09:00 - 10:00 (t_avail 10:00:10)
    # Bar 1: 10:00 - 11:00 (t_avail 11:00:10)
    # Bar 2: 11:00 - 12:00 (t_avail 12:00:10)
    rows = [
        {
            "open_time": int(datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc).timestamp() * 1000),
            "open": 42000.0,
            "high": 42100.0,
            "low": 41900.0,
            "close": 42050.0,
            "volume": 10.0,
            "close_time": int(datetime(2024, 1, 1, 9, 59, 59, 999000, tzinfo=timezone.utc).timestamp() * 1000),
            "quote_volume": 420500.0,
            "count": 50,
            "taker_buy_volume": 5.0,
            "taker_buy_quote_volume": 210250.0,
            "t_event": datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
            "t_pub": datetime(2024, 1, 1, 9, 59, 59, 999000, tzinfo=timezone.utc),
            "t_avail": datetime(2024, 1, 1, 10, 0, 9, 999000, tzinfo=timezone.utc),
            "t_ingest": datetime(2024, 1, 1, 10, 0, 10, tzinfo=timezone.utc),
            "symbol": "BTCUSDT",
            "interval": "1h",
        },
        {
            "open_time": int(datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc).timestamp() * 1000),
            "open": 42050.0,
            "high": 42300.0,
            "low": 42000.0,
            "close": 42250.0,
            "volume": 12.0,
            "close_time": int(datetime(2024, 1, 1, 10, 59, 59, 999000, tzinfo=timezone.utc).timestamp() * 1000),
            "quote_volume": 507000.0,
            "count": 60,
            "taker_buy_volume": 6.0,
            "taker_buy_quote_volume": 253500.0,
            "t_event": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            "t_pub": datetime(2024, 1, 1, 10, 59, 59, 999000, tzinfo=timezone.utc),
            "t_avail": datetime(2024, 1, 1, 11, 0, 9, 999000, tzinfo=timezone.utc),
            "t_ingest": datetime(2024, 1, 1, 11, 0, 10, tzinfo=timezone.utc),
            "symbol": "BTCUSDT",
            "interval": "1h",
        },
        {
            "open_time": int(datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc).timestamp() * 1000),
            "open": 42250.0,
            "high": 42500.0,
            "low": 42200.0,
            "close": 42400.0,
            "volume": 15.0,
            "close_time": int(datetime(2024, 1, 1, 11, 59, 59, 999000, tzinfo=timezone.utc).timestamp() * 1000),
            "quote_volume": 636000.0,
            "count": 75,
            "taker_buy_volume": 8.0,
            "taker_buy_quote_volume": 339200.0,
            "t_event": datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
            "t_pub": datetime(2024, 1, 1, 11, 59, 59, 999000, tzinfo=timezone.utc),
            "t_avail": datetime(2024, 1, 1, 12, 0, 9, 999000, tzinfo=timezone.utc),
            "t_ingest": datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc),
            "symbol": "BTCUSDT",
            "interval": "1h",
        },
    ]
    return pd.DataFrame(rows)


def test_replay_stepper_advances_correctly(memory_db, sample_market_df):
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)

    engine = HistoricalReplayEngine(
        db=memory_db,
        symbol="BTCUSDT",
        start_time=start,
        end_time=end,
        step_size="1h",
        market_data=sample_market_df,
    )

    assert engine.current_sim_time == start
    assert not engine.is_finished()
    assert not engine.is_paused

    # Step 1
    state1 = engine.step()
    expected_time1 = start + timedelta(hours=1)
    assert engine.current_sim_time == expected_time1
    assert state1["sim_time"] == expected_time1

    # Pause / Resume controls
    engine.pause()
    assert engine.is_paused is True
    engine.resume()
    assert engine.is_paused is False

    # Step 2
    state2 = engine.step()
    expected_time2 = start + timedelta(hours=2)
    assert engine.current_sim_time == expected_time2
    assert state2["sim_time"] == expected_time2

    # Step 3 (reaches end_time)
    state3 = engine.step()
    assert engine.current_sim_time == end
    assert engine.is_finished() is True


def test_replay_strict_pit_no_future_leakage(memory_db, sample_market_df):
    """
    Verify that at every point during replay:
    1. Market bar returned has t_avail <= current_sim_time.
    2. Fundamentals snapshot has t_avail <= current_sim_time.
    3. Future items are never present.
    """
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)

    engine = HistoricalReplayEngine(
        db=memory_db,
        symbol="BTCUSDT",
        start_time=start,
        end_time=end,
        step_size="30m",
        market_data=sample_market_df,
    )

    # At start (10:00:00 UTC):
    # Bar 0 became available at 10:00:09.999 (9 seconds in future).
    # So at 10:00:00, NO bar should be available!
    initial_bar = engine.get_current_bar()
    assert initial_bar is None

    # Fundamentals: record 1 available at 10:05. So at 10:00, 0 records.
    initial_fund = engine.get_current_fundamentals()
    assert len(initial_fund) == 0

    # Step 1: 10:30:00
    engine.step()
    assert engine.current_sim_time == datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)

    # Bar 0 (t_avail 10:00:09) is now available, but Bar 1 (t_avail 11:00:09) is NOT
    bar_1030 = engine.get_current_bar()
    assert bar_1030 is not None
    assert bar_1030["close"] == 42050.0
    assert bar_1030["t_avail"].to_pydatetime() <= engine.current_sim_time

    # Fundamentals: record 1 (10:05) is now available (metric_value = 100.0)
    fund_1030 = engine.get_current_fundamentals()
    assert len(fund_1030) == 1
    assert fund_1030[0]["metric_value"] == 100.0
    assert fund_1030[0]["t_avail"] <= engine.current_sim_time

    # Step through entire replay and assert invariant holds on every step
    while not engine.is_finished():
        state = engine.step()
        sim_time = state["sim_time"]

        if state["bar"] is not None:
            bar_avail = state["bar"]["t_avail"]
            if hasattr(bar_avail, "to_pydatetime"):
                bar_avail = bar_avail.to_pydatetime()
            assert bar_avail <= sim_time, f"Future bar leak: {bar_avail} > {sim_time}"

        for f in state["fundamentals"]:
            f_avail = f["t_avail"]
            if hasattr(f_avail, "to_pydatetime"):
                f_avail = f_avail.to_pydatetime()
            assert f_avail <= sim_time, f"Future fundamental leak: {f_avail} > {sim_time}"


def test_replay_with_ingested_bulk_data():
    """Verify replay engine works against actual partitioned parquet and DuckDB."""
    db = ChronosDB("data/chronos.duckdb")
    start = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc)

    engine = HistoricalReplayEngine(
        db=db,
        symbol="BTCUSDT",
        start_time=start,
        end_time=end,
        step_size="1h",
        market_data="data/parquet/market/CRYPTO/BTCUSDT",
        interval="1h",
    )

    steps = 0
    while not engine.is_finished():
        state = engine.step()
        steps += 1
        assert state["bar"] is not None
        assert state["bar"]["t_avail"].to_pydatetime() <= state["sim_time"]

    assert steps == 4
    db.close()
