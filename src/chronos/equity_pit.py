import logging
from datetime import datetime, timezone
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

    def get_as_of_fundamental(
        self,
        cik: str,
        concept: str,
        as_of_datetime: datetime,
        period_end: Optional[str] = None,
        form: Optional[str] = None,
        fiscal_period: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Point-in-time lookup of fundamental record.
        Strictly enforces: accepted_datetime <= as_of_datetime.
        Restatements: picks the latest filing where accepted_datetime <= as_of_datetime.
        """
        as_of_utc = self._normalize_datetime(as_of_datetime)
        cik_str = str(cik).zfill(10)

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

        if form is not None:
            query += " AND form = ?"
            params.append(str(form))

        if fiscal_period is not None:
            query += " AND fiscal_period = ?"
            params.append(str(fiscal_period))

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
