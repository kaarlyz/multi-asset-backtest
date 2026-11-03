import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import urllib.request

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chronos.db import ChronosDB

logger = logging.getLogger("SecEdgarIngestion")

SEC_USER_AGENT = "KaarlyzResearch ekaresearch@kaarlyz.local"
REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]


class SecEdgarIngestion:
    """Ingests raw XBRL company facts from SEC EDGAR API with cryptographic provenance."""

    def __init__(
        self,
        raw_data_dir: Optional[Union[str, Path]] = None,
        db: Optional[ChronosDB] = None,
    ):
        if raw_data_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.raw_data_dir = project_root / "data" / "raw" / "equities" / "sec_edgar"
        else:
            self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.db = db

    def fetch_raw(
        self, cik: str = "0000320193", force: bool = False
    ) -> Tuple[Path, Path]:
        """
        Fetch companyfacts JSON payload for given CIK from SEC EDGAR.
        Saves raw JSON and SHA256 manifest.
        """
        cik_zfill = str(cik).zfill(10)
        json_path = self.raw_data_dir / f"CIK{cik_zfill}.json"
        manifest_path = self.raw_data_dir / f"CIK{cik_zfill}_manifest.json"

        if json_path.exists() and manifest_path.exists() and not force:
            logger.info("Using cached SEC EDGAR payload: %s", json_path)
            return json_path, manifest_path

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_zfill}.json"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": SEC_USER_AGENT},
        )

        logger.info("Fetching SEC EDGAR data from %s", url)
        retrieval_timestamp = datetime.now(timezone.utc).isoformat()
        with urllib.request.urlopen(req) as resp:
            raw_bytes = resp.read()

        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
        byte_size = len(raw_bytes)

        # Write raw payload
        with open(json_path, "wb") as f:
            f.write(raw_bytes)

        # Write metadata manifest
        manifest_data = {
            "cik": cik_zfill,
            "source_url": url,
            "retrieval_timestamp": retrieval_timestamp,
            "byte_size": byte_size,
            "sha256": sha256_hash,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(
            "Saved CIK%s raw data (%d bytes, sha256: %s)",
            cik_zfill,
            byte_size,
            sha256_hash[:12],
        )
        return json_path, manifest_path

    def parse_facts(
        self, raw_data_or_path: Union[str, Path, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse XBRL companyfacts JSON into canonical fundamental records.
        Filters for specified revenue concepts.
        """
        if isinstance(raw_data_or_path, (str, Path)):
            with open(raw_data_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = raw_data_or_path

        cik_val = str(data.get("cik", "")).zfill(10)
        entity_name = data.get("entityName", "")
        us_gaap = data.get("facts", {}).get("us-gaap", {})

        records: List[Dict[str, Any]] = []

        for concept in REVENUE_CONCEPTS:
            if concept not in us_gaap:
                continue

            concept_data = us_gaap[concept]
            units = concept_data.get("units", {})

            for unit, items in units.items():
                for it in items:
                    accn = it.get("accn", "")
                    form = it.get("form", "")
                    filed_date = it.get("filed", "")
                    val_raw = it.get("val")
                    val = float(val_raw) if val_raw is not None else 0.0
                    fy_raw = it.get("fy")
                    fy = int(fy_raw) if fy_raw is not None else 0
                    fp = str(it.get("fp") or "")
                    start = it.get("start")
                    end = it.get("end", "")

                    if not end:
                        continue

                    period_start = start if start else end

                    # Accepted datetime: extract from item, look up in submissions cache, or fallback to filed date
                    accepted_dt = (
                        it.get("accepted_datetime")
                        or it.get("accepted")
                        or it.get("acceptanceDateTime")
                    )
                    if not accepted_dt and accn:
                        # Attempt to resolve exact acceptanceDateTime from submissions cache
                        sub_file = self.raw_data_dir / f"CIK{cik_val}_submissions.json"
                        if sub_file.exists():
                            if not hasattr(self, "_acc_cache"):
                                self._acc_cache = {}
                            if cik_val not in self._acc_cache:
                                try:
                                    with open(sub_file, "r", encoding="utf-8") as sf:
                                        s_data = json.load(sf)
                                    rec_sub = s_data.get("filings", {}).get("recent", {})
                                    self._acc_cache[cik_val] = dict(zip(rec_sub.get("accessionNumber", []), rec_sub.get("acceptanceDateTime", [])))
                                except Exception:
                                    self._acc_cache[cik_val] = {}
                            accepted_dt = self._acc_cache[cik_val].get(accn)

                    if not accepted_dt:
                        accepted_dt = f"{filed_date}T00:00:00Z"

                    record = {
                        "cik": cik_val,
                        "entity_name": entity_name,
                        "concept": concept,
                        "accession_number": accn,
                        "form": form,
                        "filed_date": filed_date,
                        "accepted_datetime": accepted_dt,
                        "period_start": period_start,
                        "period_end": end,
                        "fiscal_year": fy,
                        "fiscal_period": fp,
                        "val": val,
                        "unit": unit,
                    }
                    records.append(record)

        logger.info(
            "Parsed %d canonical facts for CIK%s across %d revenue concepts",
            len(records),
            cik_val,
            len(REVENUE_CONCEPTS),
        )
        return records

    def ingest(
        self, cik: str = "0000320193", force_fetch: bool = False
    ) -> List[Dict[str, Any]]:
        """Run raw fetch, parse canonical facts, and store into database."""
        json_path, _ = self.fetch_raw(cik=cik, force=force_fetch)
        records = self.parse_facts(json_path)

        if self.db is not None:
            inserted = self.db.insert_sec_fundamentals_batch(records)
            logger.info("Inserted %d records into fact_sec_fundamentals_pit", inserted)

        return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db_path = Path(__file__).resolve().parent.parent.parent / "data" / "chronos.duckdb"
    db = ChronosDB(str(db_path))
    ingestion = SecEdgarIngestion(db=db)
    records = ingestion.ingest(cik="0000320193", force_fetch=True)
    print(f"Successfully ingested {len(records)} records for Apple Inc.")
