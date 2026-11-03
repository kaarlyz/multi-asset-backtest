from datetime import datetime, timedelta, timezone
import pytest
import duckdb
import pandas as pd

from src.chronos.db import ChronosDB
from src.chronos.timekeeper import TimeKeeper
from src.ingestion.binance_s3 import BinanceS3Ingestion
from src.ingestion.defillama import DefiLlamaIngestion


@pytest.fixture
def db():
    database = ChronosDB(":memory:")
    yield database
    database.close()


def test_duckdb_connection_and_schema(db):
    """Test connection and presence of all required PIT tables."""
    tables = [
        r[0]
        for r in db.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    ]
    expected_tables = {
        "dim_assets",
        "fact_fundamentals_pit",
        "fact_news_pit",
        "decision_ledger",
    }
    assert expected_tables.issubset(set(tables)), f"Missing tables. Found: {tables}"


def test_temporal_integrity(db):
    """
    Test temporal integrity:
    Query get_crypto_snapshot(asset, sim_time) NEVER returns any record where t_avail > sim_time.
    """
    db.insert_asset("CRYPTO_BTC", "CRYPTO", "BTCUSDT", "BTC", "USDT")

    t_ingest = datetime.now(timezone.utc)

    # Record 1: Available at 10:05
    rec1_event = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    rec1_pub = rec1_event
    rec1_avail = rec1_pub + timedelta(minutes=5)
    db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=100.0,
        t_event=rec1_event,
        t_pub=rec1_pub,
        t_ingest=t_ingest,
        t_avail=rec1_avail,
        source="defillama",
    )

    # Record 2: Available at 11:05
    rec2_event = datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
    rec2_pub = rec2_event
    rec2_avail = rec2_pub + timedelta(minutes=5)
    db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=200.0,
        t_event=rec2_event,
        t_pub=rec2_pub,
        t_ingest=t_ingest,
        t_avail=rec2_avail,
        source="defillama",
    )

    # Record 3: Available at 12:05 (Future relative to sim_time)
    rec3_event = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    rec3_pub = rec3_event
    rec3_avail = rec3_pub + timedelta(minutes=5)
    db.insert_fundamental_pit(
        metric_id="TVL",
        asset_id="CRYPTO_BTC",
        metric_value=300.0,
        t_event=rec3_event,
        t_pub=rec3_pub,
        t_ingest=t_ingest,
        t_avail=rec3_avail,
        source="defillama",
    )

    # Sim time: 11:30 (Record 1 and Record 2 are available; Record 3 is in future)
    sim_time = datetime(2024, 1, 1, 11, 30, tzinfo=timezone.utc)

    # Test by asset_id
    snapshot = db.get_crypto_snapshot("CRYPTO_BTC", sim_time)
    assert len(snapshot) == 1
    # Strict PIT assertion: no returned record has t_avail > sim_time
    for item in snapshot:
        assert item["t_avail"] <= sim_time, f"Leak detected! t_avail {item['t_avail']} > sim_time {sim_time}"
    assert snapshot[0]["metric_value"] == 200.0

    # Test by symbol
    snapshot_by_sym = db.get_crypto_snapshot("BTCUSDT", sim_time)
    assert len(snapshot_by_sym) == 1
    assert snapshot_by_sym[0]["metric_value"] == 200.0

    # Sim time before any record is available (10:04:59)
    sim_time_early = datetime(2024, 1, 1, 10, 4, 59, tzinfo=timezone.utc)
    snapshot_early = db.get_crypto_snapshot("CRYPTO_BTC", sim_time_early)
    assert len(snapshot_early) == 0


def test_decision_ledger_idempotency(db):
    """Test decision ledger idempotency: inserting duplicate decision_id fails or is rejected."""
    now = datetime.now(timezone.utc)
    db.record_decision(
        decision_id="DEC_20240101_001",
        sim_run_id="RUN_ALPHA",
        sim_time=now,
        asset_id="CRYPTO_BTC",
        interface="AUTO_AGENT",
        actor_id="MODEL_V1",
        action="BUY",
        agent_confidence=0.92,
        evidence_snapshot={"tvl_trend": "UP", "sma": 42000.0},
        entry_price=42150.0,
        stop_loss=41000.0,
        take_profit=45000.0,
        position_size=1.5,
        status="EXECUTED",
    )

    # Inserting duplicate decision_id must raise an integrity/constraint error
    with pytest.raises((duckdb.ConstraintException, Exception)):
        db.record_decision(
            decision_id="DEC_20240101_001",
            sim_run_id="RUN_ALPHA",
            sim_time=now,
            asset_id="CRYPTO_BTC",
            interface="AUTO_AGENT",
            actor_id="MODEL_V1",
            action="SELL",
            agent_confidence=0.50,
            status="REJECTED",
        )

    # Confirm original record was not overwritten
    res = db.conn.execute("SELECT action FROM decision_ledger WHERE decision_id = 'DEC_20240101_001'").fetchone()
    assert res[0] == "BUY"


def test_binance_pit_timestamp_invariants():
    """Verify Binance S3 ingestion creates correct PIT columns and publication lag."""
    ingester = BinanceS3Ingestion(lag_seconds=10)
    # Mock Binance CSV row: open_time=1704067200000 (2024-01-01 00:00:00), close_time=1704153599999 (2024-01-01 23:59:59.999)
    raw_csv = (
        "1704067200000,42000.0,43000.0,41500.0,42500.0,1200.5,1704153599999,"
        "51021250.0,15000,600.2,25510625.0,0\n"
    )
    df = ingester.parse_klines_csv(raw_csv, symbol="BTCUSDT", interval="1d")

    required_cols = {"t_event", "t_pub", "t_ingest", "t_avail", "symbol", "interval"}
    assert required_cols.issubset(set(df.columns))

    row = df.iloc[0]
    expected_t_event = pd.to_datetime(1704067200000, unit="ms", utc=True)
    expected_t_pub = pd.to_datetime(1704153599999, unit="ms", utc=True)
    expected_t_avail = expected_t_pub + pd.Timedelta(seconds=10)

    assert row["t_event"] == expected_t_event
    assert row["t_pub"] == expected_t_pub
    assert row["t_avail"] == expected_t_avail
    assert row["t_avail"] > row["t_pub"]


def test_defillama_pit_lag_invariants():
    """Verify DefiLlama ingestion applies strict 5-minute publication lag."""
    collector = DefiLlamaIngestion(lag_minutes=5)
    mock_items = [
        {"date": 1704067200, "tvl": 50000000.0},  # 2024-01-01 00:00:00
        {"date": 1704153600, "tvl": 52000000.0},  # 2024-01-02 00:00:00
    ]
    records = collector.parse_tvl_records(mock_items, asset_id="CRYPTO_BTC")
    assert len(records) == 2

    for rec in records:
        assert rec["t_avail"] == rec["t_pub"] + timedelta(minutes=5)
        assert rec["metric_id"] == "TVL"
        assert rec["asset_id"] == "CRYPTO_BTC"
        assert rec["provenance_hash"] is not None


def test_timekeeper_invariants():
    """Verify TimeKeeper controls simulation clock and filters PIT records."""
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    tk = TimeKeeper(start_time=t0)

    t_past = t0 - timedelta(minutes=1)
    t_now = t0
    t_future = t0 + timedelta(seconds=1)

    assert tk.is_available(t_past) is True
    assert tk.is_available(t_now) is True
    assert tk.is_available(t_future) is False

    records = [
        {"id": 1, "t_avail": t_past},
        {"id": 2, "t_avail": t_now},
        {"id": 3, "t_avail": t_future},
    ]
    filtered = tk.filter_pit_records(records)
    assert [r["id"] for r in filtered] == [1, 2]

    # Advance timekeeper past future
    tk.step(seconds=5)
    assert tk.is_available(t_future) is True
