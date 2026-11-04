from datetime import datetime, timezone
from pathlib import Path
import pytest

from src.chronos.db import ChronosDB
from src.chronos.equity_pit import EquityPITReader
from src.features.financial_metrics import (
    FinancialMetricsEngine,
    calculate_quarterly_financial_metrics,
    compute_debt_to_equity,
    compute_free_cash_flow,
    compute_growth_yoy,
    compute_margin,
)
from src.features.period_matcher import (
    derive_q4_from_annual_and_nine_month,
    find_matching_prior_period,
    normalize_quarterly_records,
)
from src.ingestion.sec_edgar import (
    CANONICAL_CONCEPT_MAP,
    COMPANY_REGISTRY,
    SecEdgarIngestion,
)


@pytest.fixture
def real_duckdb():
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "data" / "chronos.duckdb"
    db = ChronosDB(str(db_path))
    yield db
    db.close()


@pytest.fixture
def test_db():
    db = ChronosDB(":memory:")
    yield db
    db.close()


def test_multi_statement_extraction_all_five_companies():
    """Verify parse_facts extracts multi-statement canonical metrics for all 5 companies."""
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "equities" / "sec_edgar"
    ingestion = SecEdgarIngestion(raw_data_dir=raw_dir)

    expected_metrics_by_sym = {
        "AAPL": {"revenue", "gross_profit", "operating_income", "net_income", "total_assets", "total_liabilities", "stockholders_equity", "operating_cash_flow", "capex"},
        "MSFT": {"revenue", "gross_profit", "operating_income", "net_income", "total_assets", "total_liabilities", "stockholders_equity", "operating_cash_flow", "capex"},
        "WMT": {"revenue", "operating_income", "net_income", "total_assets", "stockholders_equity", "operating_cash_flow", "capex"},
        "AMZN": {"revenue", "gross_profit", "operating_income", "net_income", "total_assets", "total_liabilities", "stockholders_equity", "operating_cash_flow", "capex"},
        "NVDA": {"revenue", "gross_profit", "operating_income", "net_income", "total_assets", "total_liabilities", "stockholders_equity", "operating_cash_flow", "capex"},
    }

    required_attrs = {
        "cik",
        "entity_name",
        "canonical_metric",
        "original_concept",
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

    for sym, info in COMPANY_REGISTRY.items():
        cik = info["cik"]
        json_path = raw_dir / f"CIK{cik}.json"
        assert json_path.exists(), f"Raw JSON missing for {sym}"

        records = ingestion.parse_facts(json_path, all_statements=True)
        assert len(records) > 0, f"No multi-statement records parsed for {sym}"

        found_metrics = {r["canonical_metric"] for r in records}
        expected = expected_metrics_by_sym[sym]
        assert expected.issubset(found_metrics), f"Missing canonical metrics for {sym}: {expected - found_metrics}"

        for r in records[:50]:
            assert required_attrs.issubset(r.keys()), f"Missing canonical attribute in record: {r}"
            assert r["canonical_metric"] in CANONICAL_CONCEPT_MAP


def test_derived_features_real_as_of_date_2024_03_01(real_duckdb):
    """
    Test Point-in-Time derived features on real DuckDB for all 5 companies as of 2024-03-01.
    - AAPL: Q1 FY2024 (period end 2023-12-30)
    - MSFT: Q2 FY2024 (period end 2023-12-31)
    - WMT: Q3 FY2024 (period end 2023-10-31)
    - AMZN: Q4 FY2023 (period end 2023-12-31, derived Q4)
    - NVDA: Q4 FY2024 (period end 2024-01-28, derived Q4)
    """
    reader = EquityPITReader(real_duckdb)
    engine = FinancialMetricsEngine(reader)
    as_of = datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc)

    # 1. AAPL
    aapl = engine.calculate_company_metrics("0000320193", as_of)
    assert aapl is not None, "AAPL metrics calculation returned None"
    assert aapl["current_period_end"] == "2023-12-30"
    assert aapl["fiscal_year"] == 2024
    assert aapl["fiscal_period"] == "Q1"

    a_feat = aapl["derived_features"]
    assert abs(a_feat["revenue_growth_yoy"] - 0.020665) < 1e-4
    assert abs(a_feat["gross_margin"] - 0.458750) < 1e-4
    assert abs(a_feat["operating_margin"] - 0.337637) < 1e-4
    assert abs(a_feat["net_margin"] - 0.283638) < 1e-4
    assert a_feat["free_cash_flow"] == 37503000000.0
    assert abs(a_feat["debt_to_equity"] - 3.770769) < 1e-4
    assert a_feat["equity_distress"] is False

    # 2. MSFT
    msft = engine.calculate_company_metrics("0000789019", as_of)
    assert msft is not None, "MSFT metrics calculation returned None"
    assert msft["current_period_end"] == "2023-12-31"
    assert msft["fiscal_year"] == 2024
    assert msft["fiscal_period"] == "Q2"

    m_feat = msft["derived_features"]
    assert abs(m_feat["revenue_growth_yoy"] - 0.175801) < 1e-4
    assert abs(m_feat["gross_margin"] - 0.683602) < 1e-4
    assert abs(m_feat["debt_to_equity"] - 0.974911) < 1e-4
    assert m_feat["free_cash_flow"] == 9118000000.0

    # 3. WMT (missing gross_profit and total_liabilities in US-GAAP XBRL -> returns None, no zero imputation)
    wmt = engine.calculate_company_metrics("0000104169", as_of)
    assert wmt is not None, "WMT metrics calculation returned None"
    assert wmt["current_period_end"] == "2023-10-31"
    w_feat = wmt["derived_features"]
    assert abs(w_feat["revenue_growth_yoy"] - 0.052293) < 1e-4
    assert w_feat["gross_margin"] is None, "WMT gross_margin must be None (no imputation)"
    assert w_feat["debt_to_equity"] is None, "WMT debt_to_equity must be None (no imputation)"
    assert w_feat["equity_distress"] is False

    # 4. AMZN (derived Q4)
    amzn = engine.calculate_company_metrics("0001018724", as_of)
    assert amzn is not None, "AMZN metrics calculation returned None"
    assert amzn["current_period_end"] == "2023-12-31"
    assert amzn["fiscal_year"] == 2023
    assert amzn["fiscal_period"] == "Q4"
    az_feat = amzn["derived_features"]
    assert abs(az_feat["revenue_growth_yoy"] - 0.139118) < 1e-4
    assert az_feat["free_cash_flow"] == 27877000000.0
    assert az_feat["gross_margin"] is None, "AMZN recent gross_margin must be None (unreported tag)"

    # 5. NVDA (derived Q4)
    nvda = engine.calculate_company_metrics("0001045810", as_of)
    assert nvda is not None, "NVDA metrics calculation returned None"
    assert nvda["current_period_end"] == "2024-01-28"
    assert nvda["fiscal_year"] == 2024
    assert nvda["fiscal_period"] == "Q4"
    n_feat = nvda["derived_features"]
    assert abs(n_feat["revenue_growth_yoy"] - 2.652785) < 1e-4
    assert abs(n_feat["gross_margin"] - 0.759671) < 1e-4
    assert abs(n_feat["operating_margin"] - 0.615934) < 1e-4
    assert abs(n_feat["debt_to_equity"] - 0.529341) < 1e-4
    assert n_feat["free_cash_flow"] == 11245000000.0


def test_handling_zero_negative_denominators_and_missing_facts():
    """Verify zero denominators, negative equity, and missing components return None without zero imputation."""
    # 1. Zero or negative revenue
    assert compute_margin(100.0, 0.0) is None
    assert compute_margin(100.0, -50.0) is None
    assert compute_margin(None, 100.0) is None
    assert compute_margin(100.0, None) is None

    # 2. Missing components in FCF
    assert compute_free_cash_flow(None, 50.0) is None
    assert compute_free_cash_flow(100.0, None) is None
    assert compute_free_cash_flow(None, None) is None
    # CapEx cash outflow treatment (both positive and negative values handled as outflow)
    assert compute_free_cash_flow(100.0, 30.0) == 70.0
    assert compute_free_cash_flow(100.0, -30.0) == 70.0

    # 3. Zero or missing prior in YoY growth
    assert compute_growth_yoy(120.0, 0.0) is None
    assert compute_growth_yoy(120.0, None) is None
    assert compute_growth_yoy(None, 100.0) is None
    g1 = compute_growth_yoy(120.0, 100.0)
    assert g1 is not None and abs(g1 - 0.20) < 1e-9

    g2 = compute_growth_yoy(50.0, -100.0)
    assert g2 is not None and abs(g2 - 1.50) < 1e-9

    # 4. Negative and zero equity in debt_to_equity
    ratio_neg, distress_neg = compute_debt_to_equity(1000.0, -200.0)
    assert ratio_neg is None
    assert distress_neg is True

    ratio_zero, distress_zero = compute_debt_to_equity(1000.0, 0.0)
    assert ratio_zero is None
    assert distress_zero is True

    ratio_missing, distress_missing = compute_debt_to_equity(None, 500.0)
    assert ratio_missing is None
    assert distress_missing is False

    ratio_valid, distress_valid = compute_debt_to_equity(1000.0, 500.0)
    assert ratio_valid == 2.0
    assert distress_valid is False


def test_restatements_and_pit_leakage_prevention(test_db):
    """
    Test restatement resolution and strict Point-in-Time leakage prevention across statements.
    Filing 1: Accepted 2024-01-10T10:00:00Z -> rev=100, net_inc=20
    Filing 2: Accepted 2024-02-15T10:00:00Z (Restatement) -> rev=110, net_inc=25
    Filing 3: Accepted 2024-03-20T10:00:00Z (Future quarter) -> rev=130, net_inc=30
    """
    cik = "0000999999"
    entity = "Test Corp"

    # Filing 1
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name=entity,
        concept="Revenues",
        canonical_metric="revenue",
        accession_number="0001-24-000001",
        form="10-Q",
        filed_date="2024-01-10",
        accepted_datetime=datetime(2024, 1, 10, 10, 0, tzinfo=timezone.utc),
        period_start="2023-10-01",
        period_end="2023-12-31",
        fiscal_year=2024,
        fiscal_period="Q1",
        val=100.0,
        unit="USD",
    )
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name=entity,
        concept="NetIncomeLoss",
        canonical_metric="net_income",
        accession_number="0001-24-000001",
        form="10-Q",
        filed_date="2024-01-10",
        accepted_datetime=datetime(2024, 1, 10, 10, 0, tzinfo=timezone.utc),
        period_start="2023-10-01",
        period_end="2023-12-31",
        fiscal_year=2024,
        fiscal_period="Q1",
        val=20.0,
        unit="USD",
    )

    # Filing 2: Restatement of Q1
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name=entity,
        concept="Revenues",
        canonical_metric="revenue",
        accession_number="0001-24-000002",
        form="10-Q/A",
        filed_date="2024-02-15",
        accepted_datetime=datetime(2024, 2, 15, 10, 0, tzinfo=timezone.utc),
        period_start="2023-10-01",
        period_end="2023-12-31",
        fiscal_year=2024,
        fiscal_period="Q1",
        val=110.0,
        unit="USD",
    )
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name=entity,
        concept="NetIncomeLoss",
        canonical_metric="net_income",
        accession_number="0001-24-000002",
        form="10-Q/A",
        filed_date="2024-02-15",
        accepted_datetime=datetime(2024, 2, 15, 10, 0, tzinfo=timezone.utc),
        period_start="2023-10-01",
        period_end="2023-12-31",
        fiscal_year=2024,
        fiscal_period="Q1",
        val=25.0,
        unit="USD",
    )

    # Filing 3: Future quarter Q2
    test_db.insert_sec_fundamental_pit(
        cik=cik,
        entity_name=entity,
        concept="Revenues",
        canonical_metric="revenue",
        accession_number="0001-24-000003",
        form="10-Q",
        filed_date="2024-03-20",
        accepted_datetime=datetime(2024, 3, 20, 10, 0, tzinfo=timezone.utc),
        period_start="2024-01-01",
        period_end="2024-03-31",
        fiscal_year=2024,
        fiscal_period="Q2",
        val=130.0,
        unit="USD",
    )

    reader = EquityPITReader(test_db)
    engine = FinancialMetricsEngine(reader)

    # 1. As of 2024-01-20 (before restatement): must return original val=100 and val=20
    as_of_pre = datetime(2024, 1, 20, 0, 0, tzinfo=timezone.utc)
    res_pre = engine.calculate_company_metrics(cik, as_of_pre)
    assert res_pre is not None
    assert res_pre["canonical_values"]["revenue"] == 100.0
    assert res_pre["canonical_values"]["net_income"] == 20.0
    assert abs(res_pre["derived_features"]["net_margin"] - 0.20) < 1e-9

    # 2. As of 2024-03-01 (after restatement, before Q2): must return restated val=110 and val=25
    as_of_mid = datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc)
    res_mid = engine.calculate_company_metrics(cik, as_of_mid)
    assert res_mid is not None
    assert res_mid["canonical_values"]["revenue"] == 110.0
    assert res_mid["canonical_values"]["net_income"] == 25.0
    assert abs(res_mid["derived_features"]["net_margin"] - (25.0 / 110.0)) < 1e-9
    assert res_mid["current_period_end"] == "2023-12-31"  # Q2 not leaked!

    # 3. As of 2024-04-01 (after Q2): returns Q2
    as_of_post = datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc)
    res_post = engine.calculate_company_metrics(cik, as_of_post)
    assert res_post is not None
    assert res_post["canonical_values"]["revenue"] == 130.0
    assert res_post["current_period_end"] == "2024-03-31"
