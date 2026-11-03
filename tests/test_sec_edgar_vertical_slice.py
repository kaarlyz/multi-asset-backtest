from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pytest

from src.chronos.db import ChronosDB
from src.chronos.equity_pit import EquityPITReader
from src.features.revenue_growth import calculate_quarterly_revenue_growth_yoy
from src.ingestion.sec_edgar import SecEdgarIngestion, REVENUE_CONCEPTS


@pytest.fixture
def test_db():
    db = ChronosDB(":memory:")
    yield db
    db.close()


@pytest.fixture
def real_duckdb():
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "data" / "chronos.duckdb"
    db = ChronosDB(str(db_path))
    yield db
    db.close()


def test_raw_ingestion_and_provenance():
    """Verify raw JSON exists, manifest SHA256 matches, and fact extraction parsed expected rows."""
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "equities" / "sec_edgar"
    json_path = raw_dir / "CIK0000320193.json"
    manifest_path = raw_dir / "CIK0000320193_manifest.json"

    # Ingest if not already on disk
    ingestion = SecEdgarIngestion(raw_data_dir=raw_dir)
    if not (json_path.exists() and manifest_path.exists()):
        ingestion.fetch_raw(cik="0000320193")

    assert json_path.exists(), f"Raw JSON missing at {json_path}"
    assert manifest_path.exists(), f"Manifest missing at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for field in ["cik", "source_url", "retrieval_timestamp", "byte_size", "sha256"]:
        assert field in manifest, f"Missing manifest field {field}"

    with open(json_path, "rb") as f:
        raw_bytes = f.read()

    computed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    assert computed_sha256 == manifest["sha256"], "Manifest SHA256 does not match raw JSON file"
    assert len(raw_bytes) == manifest["byte_size"], "Manifest byte size mismatch"

    records = ingestion.parse_facts(json_path)
    assert len(records) > 0, "Parsed facts list is empty"

    required_attrs = {
        "cik",
        "entity_name",
        "concept",
        "accession_number",
        "form",
        "filed_date",
        "accepted_datetime",
        "period_start",
        "period_end",
        "fiscal_year",
        "fiscal_period",
        "val",
        "unit",
    }
    for rec in records:
        assert required_attrs.issubset(rec.keys()), f"Missing canonical attributes in {rec}"
        assert rec["concept"] in REVENUE_CONCEPTS


def test_publication_time_leakage(test_db):
    """
    Query as_of prior to filing date -> assert returns None or prior period.
    Query as_of after filing date -> assert returns the new filing.
    """
    reader = EquityPITReader(test_db)
    cik = "0000320193"
    concept = "RevenueFromContractWithCustomerExcludingAssessedTax"

    # Prior period filing: Q4 2023, filed 2023-11-03
    t_prior_filed = "2023-11-03"
    t_prior_accepted = datetime(2023, 11, 3, 10, 0, tzinfo=timezone.utc)
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name="Apple Inc.",
        concept=concept,
        accession_number="0000320193-23-000106",
        form="10-K",
        filed_date=t_prior_filed,
        accepted_datetime=t_prior_accepted,
        period_start="2022-09-25",
        period_end="2023-09-30",
        fiscal_year=2023,
        fiscal_period="FY",
        val=383285000000.0,
        unit="USD",
    )

    # New period filing: Q1 2024, filed 2024-02-02
    t_new_filed = "2024-02-02"
    t_new_accepted = datetime(2024, 2, 2, 10, 0, tzinfo=timezone.utc)
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name="Apple Inc.",
        concept=concept,
        accession_number="0000320193-24-000006",
        form="10-Q",
        filed_date=t_new_filed,
        accepted_datetime=t_new_accepted,
        period_start="2023-10-01",
        period_end="2023-12-30",
        fiscal_year=2024,
        fiscal_period="Q1",
        val=119575000000.0,
        unit="USD",
    )

    # Query 1: as_of before the new filing was published (2024-02-01)
    as_of_before = datetime(2024, 2, 1, 0, 0, tzinfo=timezone.utc)

    # Specific period query for new period before filing must return None
    res_period_before = reader.get_as_of_fundamental(
        cik=cik, concept=concept, as_of_datetime=as_of_before, period_end="2023-12-30"
    )
    assert res_period_before is None, "Information leakage: returned unfiled future fundamental"

    # General query before new filing must return the prior period
    res_latest_before = reader.get_as_of_fundamental(
        cik=cik, concept=concept, as_of_datetime=as_of_before
    )
    assert res_latest_before is not None
    assert res_latest_before["period_end"] == "2023-09-30"
    assert res_latest_before["val"] == 383285000000.0

    # Query 2: as_of after the new filing was published (2024-02-03)
    as_of_after = datetime(2024, 2, 3, 0, 0, tzinfo=timezone.utc)

    # Specific period query for new period after filing must return the new data
    res_period_after = reader.get_as_of_fundamental(
        cik=cik, concept=concept, as_of_datetime=as_of_after, period_end="2023-12-30"
    )
    assert res_period_after is not None
    assert res_period_after["period_end"] == "2023-12-30"
    assert res_period_after["val"] == 119575000000.0

    # General query after new filing must return the new period
    res_latest_after = reader.get_as_of_fundamental(
        cik=cik, concept=concept, as_of_datetime=as_of_after
    )
    assert res_latest_after is not None
    assert res_latest_after["period_end"] == "2023-12-30"
    assert res_latest_after["val"] == 119575000000.0


def test_restatement_handling_pit(test_db):
    """
    - Create fixture with original filing on Date T1 (revenue = 100M).
    - Create restated filing on Date T2 (revenue = 90M for the same period_end).
    - Assert query at as_of between T1 and T2 returns 100M.
    - Assert query at as_of after T2 returns 90M.
    """
    reader = EquityPITReader(test_db)
    cik = "0000320193"
    concept = "Revenues"
    period_end = "2023-03-31"

    t1 = datetime(2023, 5, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2023, 11, 1, 10, 0, tzinfo=timezone.utc)

    # T1: Original filing
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name="Apple Inc.",
        concept=concept,
        accession_number="0000320193-23-ORIG01",
        form="10-Q",
        filed_date="2023-05-01",
        accepted_datetime=t1,
        period_start="2023-01-01",
        period_end=period_end,
        fiscal_year=2023,
        fiscal_period="Q2",
        val=100000000.0,
        unit="USD",
    )

    # T2: Restatement filing for the same period_end
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name="Apple Inc.",
        concept=concept,
        accession_number="0000320193-23-REST02",
        form="10-Q",
        filed_date="2023-11-01",
        accepted_datetime=t2,
        period_start="2023-01-01",
        period_end=period_end,
        fiscal_year=2023,
        fiscal_period="Q2",
        val=90000000.0,
        unit="USD",
    )

    # 1. As-of before T1 -> None
    as_of_before_t1 = datetime(2023, 4, 1, tzinfo=timezone.utc)
    res_before_t1 = reader.get_as_of_fundamental(cik, concept, as_of_before_t1, period_end=period_end)
    assert res_before_t1 is None

    # 2. As-of between T1 and T2 -> 100M (original filing)
    as_of_mid = datetime(2023, 8, 1, tzinfo=timezone.utc)
    res_mid = reader.get_as_of_fundamental(cik, concept, as_of_mid, period_end=period_end)
    assert res_mid is not None
    assert res_mid["val"] == 100000000.0
    assert res_mid["accession_number"] == "0000320193-23-ORIG01"

    # 3. As-of after T2 -> 90M (restated filing)
    as_of_post_t2 = datetime(2023, 12, 1, tzinfo=timezone.utc)
    res_post = reader.get_as_of_fundamental(cik, concept, as_of_post_t2, period_end=period_end)
    assert res_post is not None
    assert res_post["val"] == 90000000.0
    assert res_post["accession_number"] == "0000320193-23-REST02"


def test_duplicate_filing_idempotency(test_db):
    """Ingesting the same accession_number twice does not duplicate rows or crash."""
    record = {
        "cik": "0000320193",
        "entity_name": "Apple Inc.",
        "concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "accession_number": "0000320193-IDEM-001",
        "form": "10-Q",
        "filed_date": "2024-02-02",
        "accepted_datetime": "2024-02-02T10:00:00Z",
        "period_start": "2023-10-01",
        "period_end": "2023-12-30",
        "fiscal_year": 2024,
        "fiscal_period": "Q1",
        "val": 119575000000.0,
        "unit": "USD",
    }

    # First insert
    test_db.insert_sec_fundamentals_batch([record])
    count_1 = test_db.conn.execute("SELECT count(*) FROM fact_sec_fundamentals_pit").fetchone()[0]
    assert count_1 == 1

    # Second insert with identical primary key
    test_db.insert_sec_fundamentals_batch([record])
    count_2 = test_db.conn.execute("SELECT count(*) FROM fact_sec_fundamentals_pit").fetchone()[0]
    assert count_2 == 1, "Idempotency violated: duplicate row inserted"


def test_revenue_growth_calculation_reproducibility(real_duckdb):
    """Compute YoY revenue growth on real ingested Apple data for 2024-03-01 and assert exact reproducible result."""
    reader = EquityPITReader(real_duckdb)
    as_of = datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc)

    growth_data = calculate_quarterly_revenue_growth_yoy(reader, "0000320193", as_of)

    assert growth_data is not None, "Failed to calculate YoY revenue growth for Apple as of 2024-03-01"
    assert growth_data["current_period_end"] == "2023-12-30"
    assert growth_data["current_val"] == 119575000000.0
    assert growth_data["prev_period_end"] == "2022-12-31"
    assert growth_data["prev_val"] == 117154000000.0

    expected_growth = (119575000000.0 - 117154000000.0) / 117154000000.0
    assert abs(growth_data["growth_yoy"] - expected_growth) < 1e-9
    assert "0000320193-24-000006" in growth_data["source_accessions"]


def test_as_of_2024_01_15_returns_fy2023_q4(real_duckdb):
    """
    Assert as of 2024-01-15 (after 10-K filed on 2023-11-03 but before Q1 FY24 filed on 2024-02-02),
    calculate_quarterly_revenue_growth_yoy returns latest period_end == '2023-09-30' (Q4 FY2023).
    """
    reader = EquityPITReader(real_duckdb)
    as_of = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    cik = "0000320193"

    res = calculate_quarterly_revenue_growth_yoy(reader, cik, as_of)
    assert res is not None, "Expected growth calculation result for Apple as of 2024-01-15"
    assert res["current_period_end"] == "2023-09-30"
    assert res["current_val"] == 89498000000.0
    assert res["prev_period_end"] == "2022-09-24"
    assert res["prev_val"] == 90146000000.0

    expected_growth = (89498000000.0 - 90146000000.0) / 90146000000.0
    assert abs(res["growth_yoy"] - expected_growth) < 1e-6
    assert abs(res["growth_yoy"] - (-0.007188)) < 1e-4

    # Assert source_accessions contains all 4 accession numbers (both 10-K and 10-Q for FY23 and FY22)
    assert len(res["source_accessions"]) == 4
    assert res["source_accessions"] == [
        "0000320193-22-000070",  # FY22 Q3 10-Q (original filed 2022-07-29)
        "0000320193-22-000108",  # FY22 10-K (filed 2022-10-28)
        "0000320193-23-000077",  # FY23 Q3 10-Q (filed 2023-08-04)
        "0000320193-23-000106",  # FY23 10-K (filed 2023-11-03)
    ]

    # Direct helper verification
    derived_q4 = reader.get_derived_q4_fundamental(
        cik, "RevenueFromContractWithCustomerExcludingAssessedTax", 2023, as_of
    )
    assert derived_q4 is not None
    assert derived_q4["val"] == 89498000000.0
    assert derived_q4["fiscal_period"] == "Q4"
    assert derived_q4["is_derived"] is True
    assert derived_q4["accession_number"] == "0000320193-23-000106+0000320193-23-000077"

    pit_q4 = reader.get_as_of_fundamental(
        cik, "RevenueFromContractWithCustomerExcludingAssessedTax", as_of, fiscal_period="Q4"
    )
    assert pit_q4 is not None
    assert pit_q4["val"] == 89498000000.0
    assert pit_q4["period_end"] == "2023-09-30"


def test_temporal_boundary_before_and_after_10k_acceptance(real_duckdb):
    """
    Assert as of 2023-11-01 (before 10-K filed): latest available period is 2023-07-01 (Q3).
    Assert as of 2023-11-05 (after 10-K filed): latest available period is 2023-09-30 (Q4 derived).
    """
    reader = EquityPITReader(real_duckdb)
    cik = "0000320193"

    # Before 10-K accepted: 2023-11-01 -> latest period is Q3 (2023-07-01)
    as_of_before = datetime(2023, 11, 1, 0, 0, tzinfo=timezone.utc)
    res_before = calculate_quarterly_revenue_growth_yoy(reader, cik, as_of_before)
    assert res_before is not None
    assert res_before["current_period_end"] == "2023-07-01"

    # After 10-K accepted: 2023-11-05 -> latest period is Q4 derived (2023-09-30)
    as_of_after = datetime(2023, 11, 5, 0, 0, tzinfo=timezone.utc)
    res_after = calculate_quarterly_revenue_growth_yoy(reader, cik, as_of_after)
    assert res_after is not None
    assert res_after["current_period_end"] == "2023-09-30"


def test_missing_9m_data_prevents_derivation(test_db):
    """
    If 9M data is missing for a fiscal year, Q4 standalone returns None,
    does not impute zero or fabricate numbers.
    """
    reader = EquityPITReader(test_db)
    cik = "0000320193"
    concept = "RevenueFromContractWithCustomerExcludingAssessedTax"
    as_of = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)

    # Insert only annual 10-K filing without cumulative 9M filing
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name="Apple Inc.",
        concept=concept,
        accession_number="0000320193-23-000106",
        form="10-K",
        filed_date="2023-11-03",
        accepted_datetime=datetime(2023, 11, 3, 10, 0, tzinfo=timezone.utc),
        period_start="2022-09-25",
        period_end="2023-09-30",
        fiscal_year=2023,
        fiscal_period="FY",
        val=383285000000.0,
        unit="USD",
    )

    # Missing 9M prevents derivation -> must return None
    derived = reader.get_derived_q4_fundamental(cik, concept, 2023, as_of)
    assert derived is None, "Expected None when 9M filing is missing"

    pit_q4 = reader.get_as_of_fundamental(cik, concept, as_of, fiscal_period="Q4")
    assert pit_q4 is None, "Expected None when 9M filing is missing"

    growth = calculate_quarterly_revenue_growth_yoy(reader, cik, as_of)
    assert growth is None, "Expected None when 9M filing is missing"


def test_exact_sec_acceptance_datetime_preserved(real_duckdb):
    """
    Assert accepted_datetime comes from the actual SEC acceptance timestamp,
    e.g. 2023-11-02T22:08:27Z for Form 10-K (accession 0000320193-23-000106).
    """
    reader = EquityPITReader(real_duckdb)
    as_of = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    q4_23 = reader.get_derived_q4_fundamental(
        "0000320193", "RevenueFromContractWithCustomerExcludingAssessedTax", 2023, as_of
    )
    assert q4_23 is not None
    acc_dt = q4_23["accepted_datetime"]
    assert acc_dt.hour != 0 or acc_dt.minute != 0 or acc_dt.second != 0
    acc_utc = acc_dt.astimezone(timezone.utc)
    assert acc_utc.hour == 22 and acc_utc.minute == 8 and acc_utc.second == 27

