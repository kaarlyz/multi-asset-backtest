from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pytest

from src.chronos.db import ChronosDB
from src.chronos.equity_pit import EquityPITReader
from src.features.revenue_growth import calculate_quarterly_revenue_growth_yoy
from src.ingestion.sec_edgar import SecEdgarIngestion, COMPANY_REGISTRY, REVENUE_CONCEPTS


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


def test_raw_provenance_all_five_companies():
    """Verify raw facts and submissions JSON and SHA256 manifests exist and match for all 5 companies."""
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "equities" / "sec_edgar"
    ingestion = SecEdgarIngestion(raw_data_dir=raw_dir)

    for sym, info in COMPANY_REGISTRY.items():
        cik = info["cik"]
        # 1. Company facts JSON and manifest
        facts_json = raw_dir / f"CIK{cik}.json"
        facts_manifest = raw_dir / f"CIK{cik}_manifest.json"

        assert facts_json.exists(), f"Facts JSON missing for {sym} (CIK {cik})"
        assert facts_manifest.exists(), f"Facts manifest missing for {sym} (CIK {cik})"

        with open(facts_manifest, "r", encoding="utf-8") as f:
            f_mf = json.load(f)
        for key in ["cik", "source_url", "retrieval_timestamp", "byte_size", "sha256"]:
            assert key in f_mf, f"Manifest missing {key} for {sym} facts"

        raw_facts = facts_json.read_bytes()
        assert len(raw_facts) == f_mf["byte_size"], f"Size mismatch for {sym} facts"
        assert hashlib.sha256(raw_facts).hexdigest() == f_mf["sha256"], f"SHA256 mismatch for {sym} facts"

        # 2. Submissions JSON and manifest
        sub_json = raw_dir / f"CIK{cik}_submissions.json"
        sub_manifest = raw_dir / f"CIK{cik}_submissions_manifest.json"

        assert sub_json.exists(), f"Submissions JSON missing for {sym} (CIK {cik})"
        assert sub_manifest.exists(), f"Submissions manifest missing for {sym} (CIK {cik})"

        with open(sub_manifest, "r", encoding="utf-8") as f:
            s_mf = json.load(f)
        for key in ["cik", "source_url", "retrieval_timestamp", "byte_size", "sha256"]:
            assert key in s_mf, f"Submissions manifest missing {key} for {sym}"

        raw_sub = sub_json.read_bytes()
        assert len(raw_sub) == s_mf["byte_size"], f"Size mismatch for {sym} submissions"
        assert hashlib.sha256(raw_sub).hexdigest() == s_mf["sha256"], f"SHA256 mismatch for {sym} submissions"

        # 3. Fact parsing canonical attributes
        records = ingestion.parse_facts(facts_json)
        assert len(records) > 0, f"No facts parsed for {sym}"

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
        for r in records[:20]:
            assert required_attrs.issubset(r.keys()), f"Missing canonical attributes in {r}"
            assert r["concept"] in REVENUE_CONCEPTS


def test_pit_query_and_revenue_growth_all_five_companies_at_2024_03_01(real_duckdb):
    """
    Test PIT query and YoY revenue growth for all 5 companies as of 2024-03-01:
    - AAPL (September FY end): returns Q1 FY2024 (period end 2023-12-30)
    - MSFT (June FY end): returns Q2 FY2024 (period end 2023-12-31)
    - WMT (January FY end, 52/53-week): returns Q3 FY2024 (period end 2023-10-31)
    - AMZN (December FY end): returns derived Q4 FY2023 (period end 2023-12-31)
    - NVDA (January FY end, 52/53-week): returns derived Q4 FY2024 (period end 2024-01-28)
    """
    reader = EquityPITReader(real_duckdb)
    as_of = datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc)

    # AAPL
    aapl_res = calculate_quarterly_revenue_growth_yoy(reader, "0000320193", as_of)
    assert aapl_res is not None, "AAPL revenue growth failed"
    assert aapl_res["current_period_end"] == "2023-12-30"
    assert aapl_res["prev_period_end"] == "2022-12-31"
    assert abs(aapl_res["growth_yoy"] - 0.020665) < 1e-4

    # MSFT
    msft_res = calculate_quarterly_revenue_growth_yoy(reader, "0000789019", as_of)
    assert msft_res is not None, "MSFT revenue growth failed"
    assert msft_res["current_period_end"] == "2023-12-31"
    assert msft_res["prev_period_end"] == "2022-12-31"
    assert abs(msft_res["growth_yoy"] - 0.175801) < 1e-4

    # WMT
    wmt_res = calculate_quarterly_revenue_growth_yoy(reader, "0000104169", as_of)
    assert wmt_res is not None, "WMT revenue growth failed"
    assert wmt_res["current_period_end"] == "2023-10-31"
    assert wmt_res["prev_period_end"] == "2022-10-31"
    assert abs(wmt_res["growth_yoy"] - 0.052292) < 1e-4

    # AMZN
    amzn_res = calculate_quarterly_revenue_growth_yoy(reader, "0001018724", as_of)
    assert amzn_res is not None, "AMZN revenue growth failed"
    assert amzn_res["current_period_end"] == "2023-12-31"
    assert amzn_res["prev_period_end"] == "2022-12-31"
    assert abs(amzn_res["growth_yoy"] - 0.139118) < 1e-4
    assert len(amzn_res["source_accessions"]) == 4

    # NVDA
    nvda_res = calculate_quarterly_revenue_growth_yoy(reader, "0001045810", as_of)
    assert nvda_res is not None, "NVDA revenue growth failed"
    assert nvda_res["current_period_end"] == "2024-01-28"
    assert nvda_res["prev_period_end"] == "2023-01-29"
    assert abs(nvda_res["growth_yoy"] - 2.652784) < 1e-4
    assert len(nvda_res["source_accessions"]) == 4


def test_q4_standalone_derivation_across_fiscal_calendars(real_duckdb):
    """
    Test Q4 standalone derivation across different fiscal calendars:
    - AAPL (FY end: September, 52/53-week)
    - MSFT (FY end: June)
    - WMT (FY end: January, 52/53-week)
    - AMZN (FY end: December)
    - NVDA (FY end: January, 52/53-week)
    """
    reader = EquityPITReader(real_duckdb)

    test_cases = [
        ("AAPL", "0000320193", "RevenueFromContractWithCustomerExcludingAssessedTax", 2023, datetime(2023, 11, 5, tzinfo=timezone.utc), "2023-09-30", 89498000000.0),
        ("MSFT", "0000789019", "RevenueFromContractWithCustomerExcludingAssessedTax", 2023, datetime(2023, 8, 1, tzinfo=timezone.utc), "2023-06-30", 56189000000.0),
        ("WMT", "0000104169", "RevenueFromContractWithCustomerExcludingAssessedTax", 2024, datetime(2024, 3, 20, tzinfo=timezone.utc), "2024-01-31", 171914000000.0),
        ("AMZN", "0001018724", "RevenueFromContractWithCustomerExcludingAssessedTax", 2023, datetime(2024, 2, 5, tzinfo=timezone.utc), "2023-12-31", 169961000000.0),
        ("NVDA", "0001045810", "Revenues", 2024, datetime(2024, 2, 25, tzinfo=timezone.utc), "2024-01-28", 22103000000.0),
    ]

    for sym, cik, concept, fy, as_of, exp_period_end, exp_val in test_cases:
        derived = reader.get_derived_q4_fundamental(cik, concept, fy, as_of)
        assert derived is not None, f"Failed to derive Q4 for {sym} FY{fy}"
        assert derived["cik"] == cik
        assert derived["period_end"] == exp_period_end
        assert derived["fiscal_year"] == fy
        assert derived["fiscal_period"] == "Q4"
        assert derived["val"] == exp_val
        assert derived["is_derived"] is True
        assert "+" in derived["accession_number"]
        parts = derived["accession_number"].split("+")
        assert len(parts) == 2 and parts[0] and parts[1]


def test_publication_time_leakage_and_temporal_boundary_msft(real_duckdb):
    """
    Test publication time leakage for MSFT (June FY end):
    MSFT Q2 FY2024 (period ending 2023-12-31) was accepted on 2024-01-30 21:05:46 UTC.
    Query at 2024-01-30 20:00:00 UTC (1 hour before acceptance) -> must return previous period (2023-09-30).
    Query at 2024-01-30 22:00:00 UTC (after acceptance) -> must return the newly accepted Q2 (2023-12-31).
    """
    reader = EquityPITReader(real_duckdb)
    cik = "0000789019"
    concept = "RevenueFromContractWithCustomerExcludingAssessedTax"

    t_before = datetime(2024, 1, 30, 20, 0, 0, tzinfo=timezone.utc)
    t_after = datetime(2024, 1, 30, 22, 0, 0, tzinfo=timezone.utc)

    # Before acceptance
    res_before_specific = reader.get_as_of_fundamental(cik, concept, t_before, period_end="2023-12-31")
    assert res_before_specific is None, "Temporal leakage: returned unaccepted filing"

    res_before_general = reader.get_as_of_fundamental(cik, concept, t_before)
    assert res_before_general is not None
    assert res_before_general["period_end"] == "2023-09-30"

    # After acceptance
    res_after_specific = reader.get_as_of_fundamental(cik, concept, t_after, period_end="2023-12-31")
    assert res_after_specific is not None
    assert res_after_specific["period_end"] == "2023-12-31"
    assert res_after_specific["fiscal_period"] == "Q2"


def test_publication_time_leakage_and_temporal_boundary_wmt(real_duckdb):
    """
    Test publication time leakage for WMT (January FY end, 52/53-week):
    WMT FY2024 10-K (period ending 2024-01-31) was accepted on 2024-03-15 20:54:48 UTC.
    Query before acceptance (2024-03-15 12:00:00 UTC) -> derived Q4 returns None.
    Query after acceptance (2024-03-16 00:00:00 UTC) -> derived Q4 returns 171,914,000,000.
    """
    reader = EquityPITReader(real_duckdb)
    cik = "0000104169"
    concept = "RevenueFromContractWithCustomerExcludingAssessedTax"

    t_before = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    t_after = datetime(2024, 3, 16, 0, 0, 0, tzinfo=timezone.utc)

    q4_before = reader.get_derived_q4_fundamental(cik, concept, 2024, t_before)
    assert q4_before is None, "Temporal leakage: derived Q4 returned before 10-K accepted"

    q4_after = reader.get_derived_q4_fundamental(cik, concept, 2024, t_after)
    assert q4_after is not None
    assert q4_after["period_end"] == "2024-01-31"
    assert q4_after["val"] == 171914000000.0


def test_provenance_completeness_tracks_all_accessions(real_duckdb):
    """Verify that derived Q4 YoY calculations preserve and trace all source accessions (up to 4)."""
    reader = EquityPITReader(real_duckdb)

    # NVDA as of 2024-03-01 derives both curr Q4 FY24 and prev Q4 FY23
    nvda_res = calculate_quarterly_revenue_growth_yoy(
        reader, "0001045810", datetime(2024, 3, 1, tzinfo=timezone.utc)
    )
    assert nvda_res is not None
    accessions = nvda_res["source_accessions"]
    assert len(accessions) == 4
    for acc in accessions:
        assert isinstance(acc, str)
        assert len(acc) >= 18
        assert acc.startswith("0001045810")

    # AMZN as of 2024-03-01 derives both curr Q4 FY23 and prev Q4 FY22
    amzn_res = calculate_quarterly_revenue_growth_yoy(
        reader, "0001018724", datetime(2024, 3, 1, tzinfo=timezone.utc)
    )
    assert amzn_res is not None
    amzn_accs = amzn_res["source_accessions"]
    assert len(amzn_accs) == 4
    for acc in amzn_accs:
        assert isinstance(acc, str)
        assert len(acc) >= 18
        assert acc.startswith("0001018724")


def test_missing_or_incompatible_data_returns_none_without_crash(test_db):
    """Ensure missing records, unknown CIKs, or future queries return None without crashing or zero-imputing."""
    reader = EquityPITReader(test_db)
    as_of = datetime(2024, 3, 1, tzinfo=timezone.utc)

    # 1. Non-existent CIK in empty database
    res_empty = calculate_quarterly_revenue_growth_yoy(reader, "0000999999", as_of)
    assert res_empty is None

    # 2. Non-existent derived Q4
    res_q4 = reader.get_derived_q4_fundamental("0000999999", "Revenues", 2024, as_of)
    assert res_q4 is None

    # 3. Query before any filing existed (e.g. 1980)
    res_ancient = reader.get_as_of_fundamental(
        "0000320193", "Revenues", datetime(1980, 1, 1, tzinfo=timezone.utc)
    )
    assert res_ancient is None

    # 4. Incompatible concept returns None
    res_concept = calculate_quarterly_revenue_growth_yoy(
        reader, "0000320193", as_of, concept="NonExistentConceptXYZ"
    )
    assert res_concept is None
