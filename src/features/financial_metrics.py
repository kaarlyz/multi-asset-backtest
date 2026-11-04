from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chronos.equity_pit import EquityPITReader
from src.features.period_matcher import (
    find_matching_prior_period,
    normalize_datetime,
    normalize_quarterly_records,
)
from src.ingestion.sec_edgar import COMPANY_REGISTRY, get_company_by_cik

CANONICAL_METRICS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "operating_cash_flow",
    "capex",
]


def compute_margin(
    numerator: Optional[float], revenue: Optional[float]
) -> Optional[float]:
    """
    Computes profit margin (numerator / revenue).
    Returns None if revenue <= 0, or if either numerator or revenue is missing.
    """
    if numerator is None or revenue is None:
        return None
    if revenue <= 0.0:
        return None
    return float(numerator) / float(revenue)


def compute_free_cash_flow(
    operating_cash_flow: Optional[float], capex: Optional[float]
) -> Optional[float]:
    """
    Computes Free Cash Flow (FCF) = Operating Cash Flow - CapEx.
    CapEx is treated as a cash outflow (abs(capex)).
    Returns None if either component is missing. Never imputes zero.
    """
    if operating_cash_flow is None or capex is None:
        return None
    return float(operating_cash_flow) - abs(float(capex))


def compute_growth_yoy(
    curr_val: Optional[float], prior_val: Optional[float]
) -> Optional[float]:
    """
    Computes Year-over-Year growth: (curr - prior) / abs(prior).
    Returns None if either value is missing or if prior == 0.
    """
    if curr_val is None or prior_val is None:
        return None
    if float(prior_val) == 0.0:
        return None
    c = float(curr_val)
    p = float(prior_val)
    return (c - p) / abs(p)


def compute_debt_to_equity(
    total_liabilities: Optional[float], stockholders_equity: Optional[float]
) -> Tuple[Optional[float], bool]:
    """
    Computes Debt-to-Equity = total_liabilities / stockholders_equity.
    If equity <= 0, returns (None, True) marking equity distress.
    If either component is missing, returns (None, False).
    """
    if total_liabilities is None or stockholders_equity is None:
        return None, False
    eq = float(stockholders_equity)
    if eq <= 0.0:
        return None, True
    return float(total_liabilities) / eq, False


class FinancialMetricsEngine:
    """Point-in-Time financial metrics calculation engine across multi-statement facts."""

    def __init__(self, reader: EquityPITReader):
        self.reader = reader

    def get_quarterly_series_for_metric(
        self, cik: str, canonical_metric: str, as_of_utc: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch all available filings for canonical metric and normalize into clean quarterly series."""
        filings = self.reader.get_canonical_history(cik, canonical_metric, as_of_utc)
        return normalize_quarterly_records(filings, canonical_metric=canonical_metric)

    def calculate_company_metrics(
        self,
        cik: str,
        as_of_datetime: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Computes all canonical quarterly metrics and derived features for given company as of as_of_datetime.
        Guarantees strictly PIT information boundaries and reproducible provenance.
        """
        as_of_utc = normalize_datetime(as_of_datetime)
        cik_str = str(cik).zfill(10)
        comp_info = get_company_by_cik(cik_str)
        symbol = comp_info.get("symbol", cik_str) if comp_info else cik_str
        entity_name = comp_info.get("name", "") if comp_info else ""

        # Gather quarterly series for each canonical metric
        series_by_metric: Dict[str, List[Dict[str, Any]]] = {}
        for m in CANONICAL_METRICS:
            series_by_metric[m] = self.get_quarterly_series_for_metric(cik_str, m, as_of_utc)

        # Determine reference current quarter from revenue (or operating_income / net_income)
        ref_candidates = series_by_metric.get("revenue") or series_by_metric.get("net_income") or series_by_metric.get("operating_income")
        if not ref_candidates:
            return None

        # Latest available quarter as of as_of_utc
        curr_ref = ref_candidates[-1]
        curr_period_end = str(curr_ref["period_end"])
        curr_fy = int(curr_ref.get("fiscal_year") or 0)
        curr_fp = str(curr_ref.get("fiscal_period", "")).upper()

        current_records: Dict[str, Optional[Dict[str, Any]]] = {}
        accessions: set = set()

        for acc_field in (curr_ref.get("accession_number"),):
            if acc_field:
                for part in str(acc_field).split("+"):
                    if part.strip():
                        accessions.add(part.strip())

        # Match facts for each canonical metric for current quarter
        for m in CANONICAL_METRICS:
            series = series_by_metric.get(m, [])
            matched_rec = None

            # 1. Exact match on period_end
            exact = [r for r in series if str(r.get("period_end")) == curr_period_end]
            if exact:
                exact.sort(key=lambda x: str(x.get("accepted_datetime", "")))
                matched_rec = exact[-1]
            elif m in ("total_assets", "total_liabilities", "stockholders_equity"):
                # For balance sheet, find latest snapshot on or before curr_period_end
                prior_or_eq = [r for r in series if str(r.get("period_end")) <= curr_period_end]
                if prior_or_eq:
                    prior_or_eq.sort(key=lambda x: (str(x.get("period_end")), str(x.get("accepted_datetime"))))
                    # Allow up to 14 days difference for 52/53-week alignment
                    cand = prior_or_eq[-1]
                    try:
                        d1 = datetime.strptime(str(cand["period_end"]), "%Y-%m-%d")
                        d2 = datetime.strptime(curr_period_end, "%Y-%m-%d")
                        if abs((d2 - d1).days) <= 14:
                            matched_rec = cand
                    except Exception:
                        pass
            elif curr_fp and curr_fy:
                # Match by fiscal period and fiscal year
                fy_match = [
                    r for r in series
                    if int(r.get("fiscal_year") or 0) == curr_fy
                    and str(r.get("fiscal_period", "")).upper() == curr_fp
                ]
                if fy_match:
                    fy_match.sort(key=lambda x: (str(x.get("period_end")), str(x.get("accepted_datetime"))))
                    matched_rec = fy_match[-1]

            current_records[m] = matched_rec
            if matched_rec:
                acc = matched_rec.get("accession_number")
                if acc:
                    for part in str(acc).split("+"):
                        if part.strip():
                            accessions.add(part.strip())

        # Extract current canonical values
        canonical_vals: Dict[str, Optional[float]] = {}
        for m in CANONICAL_METRICS:
            rec = current_records.get(m)
            canonical_vals[m] = float(rec["val"]) if rec is not None and rec.get("val") is not None else None

        # Find matching prior periods for YoY calculations
        prior_records: Dict[str, Optional[Dict[str, Any]]] = {}
        for m in ["revenue", "net_income", "operating_cash_flow", "capex"]:
            curr_m_rec = current_records.get(m)
            if curr_m_rec:
                prior_cand = [
                    r for r in series_by_metric.get(m, [])
                    if str(r.get("period_end")) < str(curr_m_rec.get("period_end"))
                ]
                p_rec = find_matching_prior_period(curr_m_rec, prior_cand)
                prior_records[m] = p_rec
                if p_rec:
                    acc = p_rec.get("accession_number")
                    if acc:
                        for part in str(acc).split("+"):
                            if part.strip():
                                accessions.add(part.strip())
            else:
                prior_records[m] = None

        p_rev = prior_records.get("revenue")
        prior_revenue = float(p_rev["val"]) if p_rev is not None and p_rev.get("val") is not None else None

        p_ni = prior_records.get("net_income")
        prior_net_income = float(p_ni["val"]) if p_ni is not None and p_ni.get("val") is not None else None

        p_ocf = prior_records.get("operating_cash_flow")
        prior_ocf = float(p_ocf["val"]) if p_ocf is not None and p_ocf.get("val") is not None else None

        p_cpx = prior_records.get("capex")
        prior_capex = float(p_cpx["val"]) if p_cpx is not None and p_cpx.get("val") is not None else None

        # Derived calculations
        gross_margin = compute_margin(canonical_vals.get("gross_profit"), canonical_vals.get("revenue"))
        operating_margin = compute_margin(canonical_vals.get("operating_income"), canonical_vals.get("revenue"))
        net_margin = compute_margin(canonical_vals.get("net_income"), canonical_vals.get("revenue"))

        free_cash_flow = compute_free_cash_flow(canonical_vals.get("operating_cash_flow"), canonical_vals.get("capex"))
        prior_fcf = compute_free_cash_flow(prior_ocf, prior_capex)

        revenue_growth_yoy = compute_growth_yoy(canonical_vals.get("revenue"), prior_revenue)
        earnings_growth_yoy = compute_growth_yoy(canonical_vals.get("net_income"), prior_net_income)
        fcf_growth_yoy = compute_growth_yoy(free_cash_flow, prior_fcf)

        debt_to_equity, equity_distress = compute_debt_to_equity(
            canonical_vals.get("total_liabilities"), canonical_vals.get("stockholders_equity")
        )

        return {
            "cik": cik_str,
            "symbol": symbol,
            "entity_name": entity_name,
            "as_of_datetime": as_of_utc,
            "current_period_end": curr_period_end,
            "fiscal_year": curr_fy,
            "fiscal_period": curr_fp,
            "canonical_values": canonical_vals,
            "prior_values": {
                "revenue_prior": prior_revenue,
                "net_income_prior": prior_net_income,
                "free_cash_flow_prior": prior_fcf,
            },
            "derived_features": {
                "gross_margin": gross_margin,
                "operating_margin": operating_margin,
                "net_margin": net_margin,
                "free_cash_flow": free_cash_flow,
                "revenue_growth_yoy": revenue_growth_yoy,
                "earnings_growth_yoy": earnings_growth_yoy,
                "fcf_growth_yoy": fcf_growth_yoy,
                "debt_to_equity": debt_to_equity,
                "equity_distress": equity_distress,
            },
            "source_accessions": sorted(list(accessions)),
        }


def calculate_quarterly_financial_metrics(
    reader: EquityPITReader,
    cik: str,
    as_of_datetime: datetime,
) -> Optional[Dict[str, Any]]:
    """Convenience function to compute financial metrics using FinancialMetricsEngine."""
    engine = FinancialMetricsEngine(reader)
    return engine.calculate_company_metrics(cik, as_of_datetime)
