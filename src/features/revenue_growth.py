from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chronos.equity_pit import EquityPITReader

PREFERRED_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]


def calculate_quarterly_revenue_growth_yoy(
    reader: EquityPITReader,
    cik: str,
    as_of_datetime: datetime,
    concept: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Deterministically computes Year-over-Year (YoY) quarterly revenue growth strictly as of as_of_datetime.
    - Queries latest available quarterly filing (form='10-Q' or '10-K' Q4).
    - Finds matching quarter from previous fiscal year available as of as_of_datetime.
    - Computes growth_yoy = (current_rev - prev_rev) / abs(prev_rev).
    - If either period is missing or data unavailable, returns None (no zero imputation).
    """
    if as_of_datetime.tzinfo is None:
        as_of_utc = as_of_datetime.replace(tzinfo=timezone.utc)
    else:
        as_of_utc = as_of_datetime.astimezone(timezone.utc)

    if concept:
        concepts_to_try = [concept]
    else:
        cik_zfill = str(cik).zfill(10)
        from src.ingestion.sec_edgar import get_company_by_cik
        comp = get_company_by_cik(cik_zfill)
        if comp and "primary_concept" in comp:
            primary = comp["primary_concept"]
            concepts_to_try = [primary] + [c for c in PREFERRED_REVENUE_CONCEPTS if c != primary]
        else:
            concepts_to_try = PREFERRED_REVENUE_CONCEPTS

    results = []
    for test_concept in concepts_to_try:
        res = _compute_yoy_for_concept(reader, cik, as_of_utc, test_concept)
        if res is not None:
            if concept is not None:
                return res
            results.append(res)

    if not results:
        return None

    # Pick result with the latest current_period_end
    results.sort(key=lambda x: (str(x["current_period_end"]), str(x.get("as_of_datetime", ""))))
    return results[-1]


def _compute_yoy_for_concept(
    reader: EquityPITReader,
    cik: str,
    as_of_datetime: datetime,
    concept: str,
) -> Optional[Dict[str, Any]]:
    history = reader.get_available_filings_history(cik, concept, as_of_datetime)
    if not history:
        return None

    # Filter for quarterly facts (approx 3-month duration) and annual/9-month facts for Q4 derivation
    quarterly_records: List[Dict[str, Any]] = []
    annual_10k_by_fy: Dict[int, List[Dict[str, Any]]] = {}
    nine_month_records: List[Dict[str, Any]] = []

    for h in history:
        try:
            d1 = datetime.strptime(str(h["period_start"]), "%Y-%m-%d")
            d2 = datetime.strptime(str(h["period_end"]), "%Y-%m-%d")
            duration_days = (d2 - d1).days
        except (ValueError, TypeError):
            continue

        form = str(h.get("form", "")).upper()
        fp = str(h.get("fiscal_period", "")).upper()
        fy = int(h.get("fiscal_year") or 0)

        # 10-Q quarters are ~90 days (60-120 days)
        # 10-K Q4 is also ~90 days
        is_quarterly_duration = 60 <= duration_days <= 120
        if is_quarterly_duration and (
            form == "10-Q" or (form == "10-K" and fp in ("Q4", "FY", ""))
        ):
            quarterly_records.append(h)

        if duration_days >= 330 and form == "10-K" and fp in ("FY", "Q4", "") and fy > 0:
            annual_10k_by_fy.setdefault(fy, []).append(h)

        if 240 <= duration_days <= 305 and form == "10-Q" and fp == "Q3":
            nine_month_records.append(h)

    # Derive standalone Q4 records from 10-K and 10-Q Q3 9-month facts
    for fy, ten_ks in sorted(annual_10k_by_fy.items()):
        ten_ks.sort(key=lambda x: (str(x["period_end"]), str(x["accepted_datetime"])))
        ten_k = ten_ks[-1]

        # Check if standard 3-month Q4 already exists for this period_end
        has_standalone = any(
            str(q.get("period_end")) == str(ten_k["period_end"])
            for q in quarterly_records
        )
        if has_standalone:
            continue

        # Match 9M fact for this CIK:
        # Handles 52/53-week calendar where period_start can shift by up to 7 days
        candidate_9ms = []
        tk_start_dt = None
        try:
            tk_start_dt = datetime.strptime(str(ten_k["period_start"]), "%Y-%m-%d")
        except (ValueError, TypeError):
            pass

        for m in nine_month_records:
            if (
                m.get("unit") != ten_k.get("unit")
                or str(m.get("cik")) != str(ten_k.get("cik"))
                or m.get("concept") != ten_k.get("concept")
            ):
                continue
            if str(m.get("period_end", "")) >= str(ten_k.get("period_end", "")):
                continue

            m_start_str = str(m.get("period_start", ""))
            tk_start_str = str(ten_k.get("period_start", ""))
            diff_days = 999
            if m_start_str == tk_start_str:
                diff_days = 0
            elif tk_start_dt is not None:
                try:
                    m_start_dt = datetime.strptime(m_start_str, "%Y-%m-%d")
                    diff_days = abs((m_start_dt - tk_start_dt).days)
                except (ValueError, TypeError):
                    diff_days = 999

            if diff_days <= 7:
                candidate_9ms.append((diff_days, m))

        if not candidate_9ms:
            continue

        candidate_9ms.sort(key=lambda x: x[0])
        matching_9m = [x[1] for x in candidate_9ms]

        # Match the 9M filing specifically filed for that fiscal year:
        # e.g. for FY2022 10-K, prefer the 10-Q filed in fiscal year 2022 (e.g. 2022-07-29)
        # over comparative periods re-reported in FY2023 10-Q.
        same_fy_9m = [
            m for m in matching_9m
            if int(m.get("fiscal_year") or 0) == int(ten_k.get("fiscal_year") or fy)
            or str(m.get("filed_date", ""))[:4] == str(ten_k.get("filed_date", ""))[:4]
        ]
        if same_fy_9m:
            same_fy_9m.sort(key=lambda x: str(x.get("filed_date", "")))
            q3_9m = same_fy_9m[0]
        else:
            matching_9m.sort(key=lambda x: str(x.get("filed_date", "")))
            q3_9m = matching_9m[0]

        q4_val = float(ten_k["val"]) - float(q3_9m["val"])
        d_q3_end = datetime.strptime(str(q3_9m["period_end"]), "%Y-%m-%d")
        q4_period_start = (d_q3_end + timedelta(days=1)).strftime("%Y-%m-%d")
        q4_period_end = str(ten_k["period_end"])

        ten_k_acc = ten_k["accepted_datetime"]
        if hasattr(ten_k_acc, "tzinfo") and ten_k_acc.tzinfo is None:
            ten_k_acc = ten_k_acc.replace(tzinfo=timezone.utc)
        q3_acc = q3_9m["accepted_datetime"]
        if hasattr(q3_acc, "tzinfo") and q3_acc.tzinfo is None:
            q3_acc = q3_acc.replace(tzinfo=timezone.utc)
        q4_accepted_datetime = max(ten_k_acc, q3_acc)

        q4_provenance = f"{ten_k['accession_number']}+{q3_9m['accession_number']}"

        derived_rec = {
            "cik": ten_k["cik"],
            "entity_name": ten_k.get("entity_name", ""),
            "concept": ten_k["concept"],
            "accession_number": q4_provenance,
            "form": "10-K (derived Q4)",
            "filed_date": ten_k.get("filed_date", ""),
            "accepted_datetime": q4_accepted_datetime,
            "period_start": q4_period_start,
            "period_end": q4_period_end,
            "fiscal_year": int(ten_k.get("fiscal_year") or fy),
            "fiscal_period": "Q4",
            "val": q4_val,
            "unit": ten_k["unit"],
            "is_derived": True,
        }
        quarterly_records.append(derived_rec)

    if not quarterly_records:
        return None

    # Sort quarterly records by period_end ASC, then accepted_datetime ASC
    quarterly_records.sort(
        key=lambda x: (str(x["period_end"]), str(x["accepted_datetime"]))
    )

    # Latest quarterly filing available as of as_of_datetime
    curr = quarterly_records[-1]
    curr_val = float(curr["val"])
    curr_end_str = str(curr["period_end"])
    curr_end_dt = datetime.strptime(curr_end_str, "%Y-%m-%d")
    curr_fy = int(curr.get("fiscal_year") or 0)
    curr_fp = str(curr.get("fiscal_period", "")).upper()

    # Find matching quarter from previous fiscal year
    # Candidate must have period_end strictly prior to current period_end
    # and either match (fiscal_year == curr_fy - 1 and fiscal_period == curr_fp)
    # or period_end approx 1 year prior (340 to 390 days diff)
    candidates: List[Dict[str, Any]] = []
    for q in quarterly_records:
        q_end_str = str(q["period_end"])
        if q_end_str >= curr_end_str:
            continue

        q_end_dt = datetime.strptime(q_end_str, "%Y-%m-%d")
        day_diff = (curr_end_dt - q_end_dt).days
        q_fy = int(q.get("fiscal_year") or 0)
        q_fp = str(q.get("fiscal_period", "")).upper()

        matched = False
        if curr_fp and q_fp and curr_fp == q_fp and curr_fy > 0 and q_fy == curr_fy - 1:
            matched = True
        elif 340 <= day_diff <= 395:
            matched = True

        if matched:
            candidates.append(q)

    if not candidates:
        return None

    # If multiple candidates exist for the prior quarter (e.g. restatements or amended filings),
    # pick the latest by period_end then accepted_datetime
    candidates.sort(
        key=lambda x: (str(x["period_end"]), str(x["accepted_datetime"]))
    )
    prev = candidates[-1]
    prev_val = float(prev["val"])

    if prev_val == 0.0:
        return None

    growth_yoy = (curr_val - prev_val) / abs(prev_val)
    accessions_set = set()
    for item in (curr, prev):
        acc = item.get("accession_number", "")
        if acc:
            for part in str(acc).split("+"):
                if part.strip():
                    accessions_set.add(part.strip())
    accessions = sorted(list(accessions_set))

    return {
        "as_of_datetime": as_of_datetime,
        "current_period_end": curr_end_str,
        "current_val": curr_val,
        "prev_period_end": str(prev["period_end"]),
        "prev_val": prev_val,
        "growth_yoy": growth_yoy,
        "source_accessions": accessions,
    }
