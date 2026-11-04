from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

INSTANT_METRICS = {
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
}

FLOW_METRICS = {
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "operating_cash_flow",
    "capex",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "ProfitLoss",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
}


def parse_date(date_val: Union[str, datetime]) -> datetime:
    if isinstance(date_val, datetime):
        return date_val
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(date_val)[:19], fmt)
        except ValueError:
            pass
    return datetime.strptime(str(date_val)[:10], "%Y-%m-%d")


def normalize_datetime(dt_val: Union[str, datetime]) -> datetime:
    if isinstance(dt_val, str):
        # Remove trailing Z or parse ISO
        cleaned = dt_val.replace("Z", "+00:00")
        try:
            res = datetime.fromisoformat(cleaned)
        except ValueError:
            res = parse_date(dt_val)
    else:
        res = dt_val
    if res.tzinfo is None:
        return res.replace(tzinfo=timezone.utc)
    return res.astimezone(timezone.utc)


def get_duration_days(period_start: str, period_end: str) -> int:
    try:
        d1 = datetime.strptime(str(period_start), "%Y-%m-%d")
        d2 = datetime.strptime(str(period_end), "%Y-%m-%d")
        return (d2 - d1).days
    except (ValueError, TypeError):
        return 0


def is_instant_metric(metric: str) -> bool:
    return metric in INSTANT_METRICS


def is_flow_metric(metric: str) -> bool:
    return metric in FLOW_METRICS


def derive_q4_from_annual_and_nine_month(
    ten_k: Dict[str, Any], nine_m: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Derives standalone Q4 fundamental from Form 10-K and 10-Q Q3 9-month facts.
    Strictly verifies entity, unit, and start dates (tolerance <= 7 days for 52/53-week calendars).
    """
    if str(ten_k.get("cik")) != str(nine_m.get("cik")):
        return None
    if ten_k.get("unit") != nine_m.get("unit"):
        return None

    try:
        tk_start = datetime.strptime(str(ten_k["period_start"]), "%Y-%m-%d")
        nm_start = datetime.strptime(str(nine_m["period_start"]), "%Y-%m-%d")
        if abs((tk_start - nm_start).days) > 7:
            return None
    except (ValueError, TypeError):
        return None

    q4_val = float(ten_k["val"]) - float(nine_m["val"])
    d_q3_end = datetime.strptime(str(nine_m["period_end"]), "%Y-%m-%d")
    q4_period_start = (d_q3_end + timedelta(days=1)).strftime("%Y-%m-%d")
    q4_period_end = str(ten_k["period_end"])

    ten_k_acc = normalize_datetime(ten_k["accepted_datetime"])
    nine_m_acc = normalize_datetime(nine_m["accepted_datetime"])
    q4_accepted = max(ten_k_acc, nine_m_acc)

    q4_provenance = f"{ten_k['accession_number']}+{nine_m['accession_number']}"
    concept_str = ten_k.get("concept") or ten_k.get("original_concept") or ""

    return {
        "cik": ten_k["cik"],
        "entity_name": ten_k.get("entity_name", ""),
        "canonical_metric": ten_k.get("canonical_metric"),
        "original_concept": ten_k.get("original_concept", concept_str),
        "concept": concept_str,
        "accession_number": q4_provenance,
        "form": "10-K (derived Q4)",
        "filed_date": str(ten_k.get("filed_date", "")),
        "accepted_datetime": q4_accepted,
        "period_start": q4_period_start,
        "period_end": q4_period_end,
        "fiscal_year": int(ten_k.get("fiscal_year") or 0),
        "fiscal_period": "Q4",
        "val": q4_val,
        "unit": ten_k["unit"],
        "is_derived": True,
    }


def normalize_quarterly_records(
    filings: List[Dict[str, Any]],
    canonical_metric: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Normalizes a list of raw PIT filings into clean quarterly records.
    - For balance sheet (instant metrics): filters for snapshot records (duration <= 1 day).
    - For income/cash flow (flow metrics): filters for quarterly records (duration 60-125 days)
      and derives Q4 from 10-K and 9M 10-Q when standalone Q4 does not exist.
      Also derives standalone Q2/Q3 from cumulative cash flow statements if direct 3M is absent.
    """
    if not filings:
        return []

    is_balance_sheet = False
    if canonical_metric and is_instant_metric(canonical_metric):
        is_balance_sheet = True
    elif not canonical_metric:
        c = filings[0].get("canonical_metric") or filings[0].get("concept")
        if c and is_instant_metric(c):
            is_balance_sheet = True

    if is_balance_sheet:
        instant_records = []
        for f in filings:
            dur = get_duration_days(f.get("period_start", ""), f.get("period_end", ""))
            # Balance sheet instant facts have start == end (dur == 0) or <= 1 day
            if dur <= 1:
                instant_records.append(f)
        instant_records.sort(key=lambda x: (str(x["period_end"]), str(x["accepted_datetime"])))
        return instant_records

    # Flow metrics processing
    quarterly_records: List[Dict[str, Any]] = []
    annual_10k_by_fy: Dict[int, List[Dict[str, Any]]] = {}
    nine_month_records: List[Dict[str, Any]] = []
    six_month_records: List[Dict[str, Any]] = []
    three_month_q1_records: List[Dict[str, Any]] = []

    for h in filings:
        dur = get_duration_days(h.get("period_start", ""), h.get("period_end", ""))
        form = str(h.get("form", "")).upper()
        fp = str(h.get("fiscal_period", "")).upper()
        fy = int(h.get("fiscal_year") or 0)

        # Standalone quarterly fact (60 to 125 days)
        if 60 <= dur <= 125 and (form.startswith("10-Q") or (form.startswith("10-K") and fp in ("Q4", "FY", ""))):
            quarterly_records.append(h)
            if fp == "Q1":
                three_month_q1_records.append(h)

        if dur >= 330 and form.startswith("10-K") and fp in ("FY", "Q4", "") and fy > 0:
            annual_10k_by_fy.setdefault(fy, []).append(h)

        if 240 <= dur <= 305 and form.startswith("10-Q") and fp == "Q3":
            nine_month_records.append(h)

        if 150 <= dur <= 210 and form.startswith("10-Q") and fp == "Q2":
            six_month_records.append(h)

    # Derive Q4 for missing fiscal years
    for fy, ten_ks in sorted(annual_10k_by_fy.items()):
        ten_ks.sort(key=lambda x: (str(x["period_end"]), str(x["accepted_datetime"])))
        ten_k = ten_ks[-1]

        has_standalone = any(
            str(q.get("period_end")) == str(ten_k["period_end"]) for q in quarterly_records
        )
        if has_standalone:
            continue

        candidate_9ms = []
        for m in nine_month_records:
            if (
                m.get("unit") != ten_k.get("unit")
                or str(m.get("cik")) != str(ten_k.get("cik"))
                or str(m.get("period_end", "")) >= str(ten_k.get("period_end", ""))
            ):
                continue
            diff = abs(get_duration_days(m.get("period_start", ""), ten_k.get("period_start", "")))
            if diff <= 7:
                candidate_9ms.append((diff, m))

        if not candidate_9ms:
            continue

        candidate_9ms.sort(key=lambda x: x[0])
        matching_9m = [x[1] for x in candidate_9ms]
        same_fy_9m = [
            m
            for m in matching_9m
            if int(m.get("fiscal_year") or 0) == fy
            or str(m.get("filed_date", ""))[:4] == str(ten_k.get("filed_date", ""))[:4]
        ]
        chosen_9m = same_fy_9m[0] if same_fy_9m else matching_9m[0]

        derived_q4 = derive_q4_from_annual_and_nine_month(ten_k, chosen_9m)
        if derived_q4:
            quarterly_records.append(derived_q4)

    # For cumulative cash flows where Q2 or Q3 standalone are missing:
    # Q2 standalone = 6M - Q1
    # Q3 standalone = 9M - 6M
    has_q2 = any(str(q.get("fiscal_period", "")).upper() == "Q2" for q in quarterly_records)
    if not has_q2 and six_month_records and three_month_q1_records:
        for s in six_month_records:
            s_fy = int(s.get("fiscal_year") or 0)
            matching_q1 = [
                q for q in three_month_q1_records
                if int(q.get("fiscal_year") or 0) == s_fy and q.get("unit") == s.get("unit")
            ]
            if matching_q1:
                q1 = matching_q1[-1]
                q2_val = float(s["val"]) - float(q1["val"])
                d_q1_end = datetime.strptime(str(q1["period_end"]), "%Y-%m-%d")
                q2_start = (d_q1_end + timedelta(days=1)).strftime("%Y-%m-%d")
                s_acc = normalize_datetime(s["accepted_datetime"])
                q1_acc = normalize_datetime(q1["accepted_datetime"])
                derived_q2 = {
                    "cik": s["cik"],
                    "entity_name": s.get("entity_name", ""),
                    "canonical_metric": s.get("canonical_metric"),
                    "original_concept": s.get("original_concept"),
                    "concept": s.get("concept"),
                    "accession_number": f"{s['accession_number']}+{q1['accession_number']}",
                    "form": "10-Q (derived Q2)",
                    "filed_date": str(s.get("filed_date", "")),
                    "accepted_datetime": max(s_acc, q1_acc),
                    "period_start": q2_start,
                    "period_end": str(s["period_end"]),
                    "fiscal_year": s_fy,
                    "fiscal_period": "Q2",
                    "val": q2_val,
                    "unit": s["unit"],
                    "is_derived": True,
                }
                quarterly_records.append(derived_q2)

    has_q3 = any(str(q.get("fiscal_period", "")).upper() == "Q3" for q in quarterly_records)
    if not has_q3 and nine_month_records and six_month_records:
        for n in nine_month_records:
            n_fy = int(n.get("fiscal_year") or 0)
            matching_6m = [
                s for s in six_month_records
                if int(s.get("fiscal_year") or 0) == n_fy and s.get("unit") == n.get("unit")
            ]
            if matching_6m:
                s6 = matching_6m[-1]
                q3_val = float(n["val"]) - float(s6["val"])
                d_s6_end = datetime.strptime(str(s6["period_end"]), "%Y-%m-%d")
                q3_start = (d_s6_end + timedelta(days=1)).strftime("%Y-%m-%d")
                n_acc = normalize_datetime(n["accepted_datetime"])
                s6_acc = normalize_datetime(s6["accepted_datetime"])
                derived_q3 = {
                    "cik": n["cik"],
                    "entity_name": n.get("entity_name", ""),
                    "canonical_metric": n.get("canonical_metric"),
                    "original_concept": n.get("original_concept"),
                    "concept": n.get("concept"),
                    "accession_number": f"{n['accession_number']}+{s6['accession_number']}",
                    "form": "10-Q (derived Q3)",
                    "filed_date": str(n.get("filed_date", "")),
                    "accepted_datetime": max(n_acc, s6_acc),
                    "period_start": q3_start,
                    "period_end": str(n["period_end"]),
                    "fiscal_year": n_fy,
                    "fiscal_period": "Q3",
                    "val": q3_val,
                    "unit": n["unit"],
                    "is_derived": True,
                }
                quarterly_records.append(derived_q3)

    quarterly_records.sort(key=lambda x: (str(x["period_end"]), str(x["accepted_datetime"])))
    return quarterly_records


def find_matching_prior_period(
    curr_rec: Dict[str, Any],
    candidate_records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Finds the Year-over-Year matching prior period for a given current quarter.
    - Prioritizes fiscal_year == curr_fy - 1 and fiscal_period == curr_fp.
    - Validates calendar delta tolerance (340 to 395 days) for 52/53-week fiscal companies.
    - If multiple match (restatements/amendments), picks latest by period_end then accepted_datetime.
    """
    curr_end_str = str(curr_rec["period_end"])
    curr_end_dt = datetime.strptime(curr_end_str, "%Y-%m-%d")
    curr_fy = int(curr_rec.get("fiscal_year") or 0)
    curr_fp = str(curr_rec.get("fiscal_period", "")).upper()

    matches: List[Dict[str, Any]] = []
    for cand in candidate_records:
        cand_end_str = str(cand["period_end"])
        if cand_end_str >= curr_end_str:
            continue

        cand_end_dt = datetime.strptime(cand_end_str, "%Y-%m-%d")
        day_diff = (curr_end_dt - cand_end_dt).days
        cand_fy = int(cand.get("fiscal_year") or 0)
        cand_fp = str(cand.get("fiscal_period", "")).upper()

        matched = False
        if curr_fp and cand_fp and curr_fp == cand_fp and curr_fy > 0 and cand_fy == curr_fy - 1:
            matched = True
        elif 340 <= day_diff <= 395:
            matched = True

        if matched:
            matches.append(cand)

    if not matches:
        return None

    matches.sort(key=lambda x: (str(x["period_end"]), str(x["accepted_datetime"])))
    return matches[-1]
