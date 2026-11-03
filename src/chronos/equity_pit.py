import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chronos.db import ChronosDB

logger = logging.getLogger("EquityPITReader")


class EquityPITReader:
    """Point-in-Time (PIT) query engine for SEC EDGAR fundamentals."""

    def __init__(self, db: ChronosDB):
        self.db = db

    def _normalize_datetime(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _get_raw_fundamental(
        self,
        cik_str: str,
        concept: str,
        as_of_utc: datetime,
        period_end: Optional[str] = None,
        form: Optional[str] = None,
        fiscal_period: Optional[str] = None,
        fiscal_year: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        query = """
            SELECT cik, entity_name, concept, accession_number, form, filed_date,
                   accepted_datetime, period_start, period_end, fiscal_year,
                   fiscal_period, val, unit
            FROM fact_sec_fundamentals_pit
            WHERE cik = ? AND concept = ? AND accepted_datetime <= ?
        """
        params: List[Any] = [cik_str, concept, as_of_utc]

        if period_end is not None:
            query += " AND period_end = ?"
            params.append(str(period_end))

        if form is not None and form != "10-K (derived Q4)":
            query += " AND form = ?"
            params.append(str(form))

        if fiscal_period is not None:
            query += " AND fiscal_period = ?"
            params.append(str(fiscal_period))

        if fiscal_year is not None:
            query += " AND fiscal_year = ?"
            params.append(fiscal_year)

        # Restatement resolution: order by period_end DESC, accepted_datetime DESC
        query += " ORDER BY period_end DESC, accepted_datetime DESC, filed_date DESC LIMIT 1"

        with self.db._lock:
            cursor = self.db.conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cursor.description]
            res = dict(zip(cols, row))

            # Additional temporal guard: ensure no future leak
            rec_accepted = res["accepted_datetime"]
            if hasattr(rec_accepted, "tzinfo") and rec_accepted.tzinfo is None:
                rec_accepted = rec_accepted.replace(tzinfo=timezone.utc)
            if rec_accepted > as_of_utc:
                raise ValueError(
                    f"Temporal leak detected! Record accepted at {rec_accepted} > as_of {as_of_utc}"
                )
            return res

    def get_derived_q4_fundamental(
        self, cik: str, concept: str, fiscal_year: int, as_of_datetime: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Derives standalone Q4 fundamental from Form 10-K and 10-Q Q3 9-month facts.
        Q4 = annual_10K - cumulative_9M.
        Returns record with full provenance or None if required filings are missing.
        """
        as_of_utc = self._normalize_datetime(as_of_datetime)
        cik_str = str(cik).zfill(10)
        history = self.get_available_filings_history(cik_str, concept, as_of_utc)
        if not history:
            return None

        annual_10ks = []
        nine_ms = []

        for h in history:
            try:
                d1 = datetime.strptime(str(h["period_start"]), "%Y-%m-%d")
                d2 = datetime.strptime(str(h["period_end"]), "%Y-%m-%d")
                dur = (d2 - d1).days
            except (ValueError, TypeError):
                continue

            form = str(h.get("form", "")).upper()
            fp = str(h.get("fiscal_period", "")).upper()
            fy = int(h.get("fiscal_year") or 0)

            if form == "10-K" and fp in ("FY", "Q4", "") and dur >= 330 and fy == fiscal_year:
                annual_10ks.append(h)
            elif form == "10-Q" and fp == "Q3" and 240 <= dur <= 300:
                nine_ms.append(h)

        if not annual_10ks:
            return None

        # Pick 10-K with latest period_end, then latest accepted_datetime
        annual_10ks.sort(key=lambda x: (str(x["period_end"]), str(x["accepted_datetime"])))
        ten_k = annual_10ks[-1]

        matching_9m = [
            m for m in nine_ms
            if str(m.get("period_start")) == str(ten_k["period_start"])
            and m.get("unit") == ten_k.get("unit")
            and m.get("cik") == ten_k.get("cik")
        ]
        if not matching_9m:
            return None

        matching_9m.sort(key=lambda x: str(x["accepted_datetime"]))
        nine_m = matching_9m[-1]

        q4_val = float(ten_k["val"]) - float(nine_m["val"])
        d_q3_end = datetime.strptime(str(nine_m["period_end"]), "%Y-%m-%d")
        q4_period_start = (d_q3_end + timedelta(days=1)).strftime("%Y-%m-%d")
        q4_period_end = str(ten_k["period_end"])

        ten_k_acc = self._normalize_datetime(ten_k["accepted_datetime"])
        nine_m_acc = self._normalize_datetime(nine_m["accepted_datetime"])
        q4_accepted = max(ten_k_acc, nine_m_acc)

        q4_provenance = f"{ten_k['accession_number']}+{nine_m['accession_number']}"

        return {
            "cik": ten_k["cik"],
            "entity_name": ten_k.get("entity_name", ""),
            "concept": ten_k["concept"],
            "accession_number": q4_provenance,
            "form": "10-K (derived Q4)",
            "filed_date": str(ten_k.get("filed_date", "")),
            "accepted_datetime": q4_accepted,
            "period_start": q4_period_start,
            "period_end": q4_period_end,
            "fiscal_year": int(ten_k["fiscal_year"]),
            "fiscal_period": "Q4",
            "val": q4_val,
            "unit": ten_k["unit"],
            "is_derived": True,
        }

    def get_as_of_fundamental(
        self,
        cik: str,
        concept: str,
        as_of_datetime: datetime,
        period_end: Optional[str] = None,
        form: Optional[str] = None,
        fiscal_period: Optional[str] = None,
        fiscal_year: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Point-in-time lookup of fundamental record.
        Strictly enforces: accepted_datetime <= as_of_datetime.
        Restatements: picks the latest filing where accepted_datetime <= as_of_datetime.
        """
        as_of_utc = self._normalize_datetime(as_of_datetime)
        cik_str = str(cik).zfill(10)

        # Allow querying fiscal_period='Q4' utilizing derived helper when standalone Q4 does not exist in raw table
        if fiscal_period is not None and str(fiscal_period).upper() == "Q4":
            raw_q4 = self._get_raw_fundamental(
                cik_str, concept, as_of_utc, period_end, form, "Q4", fiscal_year
            )
            if raw_q4 is not None:
                try:
                    d1 = datetime.strptime(str(raw_q4["period_start"]), "%Y-%m-%d")
                    d2 = datetime.strptime(str(raw_q4["period_end"]), "%Y-%m-%d")
                    dur = (d2 - d1).days
                except (ValueError, TypeError):
                    dur = 0
                if 60 <= dur <= 120:
                    return raw_q4

            if fiscal_year is not None:
                derived = self.get_derived_q4_fundamental(cik_str, concept, fiscal_year, as_of_utc)
                if derived is not None:
                    if period_end is not None and str(derived["period_end"]) != str(period_end):
                        return None
                    return derived
                return None

            if period_end is not None:
                query = """
                    SELECT fiscal_year FROM fact_sec_fundamentals_pit
                    WHERE cik = ? AND concept = ? AND period_end = ? AND form = '10-K'
                      AND accepted_datetime <= ?
                    ORDER BY accepted_datetime DESC LIMIT 1
                """
                with self.db._lock:
                    cur = self.db.conn.cursor()
                    cur.execute(query, [cik_str, concept, str(period_end), as_of_utc])
                    row = cur.fetchone()
                if row:
                    derived = self.get_derived_q4_fundamental(cik_str, concept, int(row[0]), as_of_utc)
                    if derived is not None and str(derived["period_end"]) == str(period_end):
                        return derived
                return None

            query = """
                SELECT DISTINCT fiscal_year FROM fact_sec_fundamentals_pit
                WHERE cik = ? AND concept = ? AND form = '10-K' AND accepted_datetime <= ?
                ORDER BY fiscal_year DESC
            """
            with self.db._lock:
                cur = self.db.conn.cursor()
                cur.execute(query, [cik_str, concept, as_of_utc])
                candidate_fys = [r[0] for r in cur.fetchall()]

            for fy in candidate_fys:
                derived = self.get_derived_q4_fundamental(cik_str, concept, int(fy), as_of_utc)
                if derived is not None:
                    return derived
            return None

        return self._get_raw_fundamental(
            cik_str, concept, as_of_utc, period_end, form, fiscal_period, fiscal_year
        )

    def get_available_filings_history(
        self, cik: str, concept: str, as_of_datetime: datetime
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all filings for (cik, concept) available as of as_of_datetime,
        strictly adhering to accepted_datetime <= as_of_datetime.
        """
        as_of_utc = self._normalize_datetime(as_of_datetime)
        cik_str = str(cik).zfill(10)

        query = """
            SELECT cik, entity_name, concept, accession_number, form, filed_date,
                   accepted_datetime, period_start, period_end, fiscal_year,
                   fiscal_period, val, unit
            FROM fact_sec_fundamentals_pit
            WHERE cik = ? AND concept = ? AND accepted_datetime <= ?
            ORDER BY period_end ASC, accepted_datetime ASC
        """
        with self.db._lock:
            cursor = self.db.conn.cursor()
            cursor.execute(query, [cik_str, concept, as_of_utc])
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        results = []
        for row in rows:
            rec = dict(zip(cols, row))
            # Verify no leak
            rec_accepted = rec["accepted_datetime"]
            if hasattr(rec_accepted, "tzinfo") and rec_accepted.tzinfo is None:
                rec_accepted = rec_accepted.replace(tzinfo=timezone.utc)
            if rec_accepted <= as_of_utc:
                results.append(rec)
        return results
