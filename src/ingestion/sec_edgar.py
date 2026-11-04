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

CANONICAL_CONCEPT_MAP: Dict[str, List[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
}

CONCEPT_TO_CANONICAL: Dict[str, str] = {}
for canonical, concepts in CANONICAL_CONCEPT_MAP.items():
    for c in concepts:
        CONCEPT_TO_CANONICAL[c] = canonical

REVENUE_CONCEPTS = CANONICAL_CONCEPT_MAP["revenue"]

COMPANY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "AAPL": {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "fiscal_year_end": "September",
        "primary_concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
    },
    "MSFT": {
        "cik": "0000789019",
        "name": "Microsoft Corp.",
        "fiscal_year_end": "June",
        "primary_concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
    },
    "WMT": {
        "cik": "0000104169",
        "name": "Walmart Inc.",
        "fiscal_year_end": "January",
        "primary_concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
    },
    "AMZN": {
        "cik": "0001018724",
        "name": "Amazon.com Inc.",
        "fiscal_year_end": "December",
        "primary_concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
    },
    "NVDA": {
        "cik": "0001045810",
        "name": "NVIDIA Corp.",
        "fiscal_year_end": "January",
        "primary_concept": "Revenues",
    },
}


def get_company_by_cik(cik: str) -> Optional[Dict[str, Any]]:
    """Lookup company information from COMPANY_REGISTRY by CIK."""
    cik_zfill = str(cik).zfill(10)
    for sym, info in COMPANY_REGISTRY.items():
        if info["cik"] == cik_zfill:
            return {**info, "symbol": sym}
    return None


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

    def fetch_submissions(
        self, cik: str = "0000320193", force: bool = False
    ) -> Tuple[Path, Path]:
        """
        Fetch submissions JSON metadata for given CIK from SEC EDGAR.
        Saves raw JSON and SHA256 manifest.
        """
        cik_zfill = str(cik).zfill(10)
        json_path = self.raw_data_dir / f"CIK{cik_zfill}_submissions.json"
        manifest_path = self.raw_data_dir / f"CIK{cik_zfill}_submissions_manifest.json"

        if json_path.exists() and manifest_path.exists() and not force:
            logger.info("Using cached SEC EDGAR submissions: %s", json_path)
            return json_path, manifest_path

        url = f"https://data.sec.gov/submissions/CIK{cik_zfill}.json"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": SEC_USER_AGENT},
        )

        logger.info("Fetching SEC EDGAR submissions from %s", url)
        retrieval_timestamp = datetime.now(timezone.utc).isoformat()
        with urllib.request.urlopen(req) as resp:
            raw_bytes = resp.read()

        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
        byte_size = len(raw_bytes)

        with open(json_path, "wb") as f:
            f.write(raw_bytes)

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
            "Saved CIK%s submissions raw data (%d bytes, sha256: %s)",
            cik_zfill,
            byte_size,
            sha256_hash[:12],
        )
        return json_path, manifest_path

    def parse_facts(
        self,
        raw_data_or_path: Union[str, Path, Dict[str, Any]],
        canonical_metrics: Optional[Union[str, List[str]]] = None,
        all_statements: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Parse XBRL companyfacts JSON into canonical fundamental records.
        Filters for specified canonical concepts.
        Excludes segmented facts with dimensions.
        Enforces deduplication on primary key.
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
        seen_keys = set()

        if isinstance(canonical_metrics, str):
            target_canonical = [canonical_metrics]
        elif canonical_metrics is not None:
            target_canonical = list(canonical_metrics)
        elif all_statements:
            target_canonical = list(CANONICAL_CONCEPT_MAP.keys())
        else:
            target_canonical = ["revenue"]

        concept_pairs: List[Tuple[str, str]] = []
        for c_metric in target_canonical:
            for concept in CANONICAL_CONCEPT_MAP.get(c_metric, []):
                concept_pairs.append((c_metric, concept))

        for canonical_metric, concept in concept_pairs:
            if concept not in us_gaap:
                continue

            concept_data = us_gaap[concept]
            units = concept_data.get("units", {})

            for unit, items in units.items():
                for it in items:
                    # Filter only consolidated facts (exclude segmented facts with dimensions)
                    if (
                        it.get("segment")
                        or it.get("dimensions")
                        or it.get("dimension")
                        or it.get("segments")
                    ):
                        continue

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

                    # Primary key enforcement
                    pk_key = (cik_val, concept, accn, end, form, period_start)
                    if pk_key in seen_keys:
                        continue
                    seen_keys.add(pk_key)

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
                                    acc_nums = rec_sub.get("accessionNumber", [])
                                    acc_dts = rec_sub.get("acceptanceDateTime", [])
                                    cache_map = {}
                                    for a_num, a_dt in zip(acc_nums, acc_dts):
                                        cache_map[a_num] = a_dt
                                        cache_map[a_num.replace("-", "")] = a_dt
                                    self._acc_cache[cik_val] = cache_map
                                except Exception:
                                    self._acc_cache[cik_val] = {}
                            accepted_dt = self._acc_cache[cik_val].get(accn) or self._acc_cache[cik_val].get(accn.replace("-", ""))

                    if not accepted_dt:
                        accepted_dt = f"{filed_date}T00:00:00Z"

                    record = {
                        "cik": cik_val,
                        "entity_name": entity_name,
                        "canonical_metric": canonical_metric,
                        "original_concept": concept,
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

        # Handling for total_liabilities synthesis: LiabilitiesCurrent + LiabilitiesNoncurrent
        if "total_liabilities" in target_canonical and "LiabilitiesCurrent" in us_gaap and "LiabilitiesNoncurrent" in us_gaap:
            lc_units = us_gaap["LiabilitiesCurrent"].get("units", {})
            lnc_units = us_gaap["LiabilitiesNoncurrent"].get("units", {})
            for u in lc_units:
                if u not in lnc_units:
                    continue
                lnc_clean = [
                    x for x in lnc_units[u]
                    if not (x.get("segment") or x.get("dimensions") or x.get("dimension") or x.get("segments"))
                ]
                lnc_map = {(x.get("end"), x.get("form"), x.get("accn")): x for x in lnc_clean}

                for lc in lc_units[u]:
                    if lc.get("segment") or lc.get("dimensions") or lc.get("dimension") or lc.get("segments"):
                        continue
                    end = lc.get("end", "")
                    form = lc.get("form", "")
                    accn = lc.get("accn", "")
                    start = lc.get("start")
                    p_start = start if start else end
                    k = (end, form, accn)
                    if k in lnc_map:
                        lnc = lnc_map[k]
                        synth_concept = "LiabilitiesCurrent+LiabilitiesNoncurrent"
                        pk_key = (cik_val, synth_concept, accn, end, form, p_start)
                        if pk_key in seen_keys:
                            continue
                        seen_keys.add(pk_key)

                        val = float(lc.get("val") or 0.0) + float(lnc.get("val") or 0.0)
                        fy = int(lc.get("fy") or lnc.get("fy") or 0)
                        fp = str(lc.get("fp") or lnc.get("fp") or "")
                        filed_date = lc.get("filed") or lnc.get("filed") or ""

                        accepted_dt = lc.get("accepted_datetime") or lc.get("accepted") or lc.get("acceptanceDateTime")
                        if not accepted_dt and hasattr(self, "_acc_cache") and cik_val in self._acc_cache:
                            accepted_dt = self._acc_cache[cik_val].get(accn) or self._acc_cache[cik_val].get(accn.replace("-", ""))
                        if not accepted_dt:
                            accepted_dt = f"{filed_date}T00:00:00Z"

                        records.append({
                            "cik": cik_val,
                            "entity_name": entity_name,
                            "canonical_metric": "total_liabilities",
                            "original_concept": synth_concept,
                            "concept": synth_concept,
                            "accession_number": accn,
                            "form": form,
                            "filed_date": filed_date,
                            "accepted_datetime": accepted_dt,
                            "period_start": p_start,
                            "period_end": end,
                            "fiscal_year": fy,
                            "fiscal_period": fp,
                            "val": val,
                            "unit": u,
                        })

        logger.info(
            "Parsed %d canonical facts for CIK%s across canonical metrics: %s",
            len(records),
            cik_val,
            target_canonical,
        )
        return records

    def ingest(
        self, cik: str = "0000320193", force_fetch: bool = False, all_statements: bool = True
    ) -> List[Dict[str, Any]]:
        """Run raw fetch for facts and submissions, parse canonical facts, and store into database."""
        cik_zfill = str(cik).zfill(10)
        sub_file = self.raw_data_dir / f"CIK{cik_zfill}_submissions.json"
        if force_fetch or not sub_file.exists():
            try:
                self.fetch_submissions(cik=cik_zfill, force=force_fetch)
            except Exception as e:
                logger.warning("Could not fetch submissions for CIK%s: %s", cik_zfill, e)

        json_path, _ = self.fetch_raw(cik=cik_zfill, force=force_fetch)
        records = self.parse_facts(json_path, all_statements=all_statements)

        if self.db is not None:
            inserted = self.db.insert_sec_fundamentals_batch(records)
            logger.info("Inserted %d records into fact_sec_fundamentals_pit", inserted)

        return records

    def ingest_all(self, force_fetch: bool = False, all_statements: bool = True) -> Dict[str, int]:
        """Run batch ingestion across all registered companies in COMPANY_REGISTRY."""
        results = {}
        for sym, info in COMPANY_REGISTRY.items():
            cik = info["cik"]
            logger.info("Ingesting %s (CIK %s)...", sym, cik)
            records = self.ingest(cik=cik, force_fetch=force_fetch, all_statements=all_statements)
            results[sym] = len(records)
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db_path = Path(__file__).resolve().parent.parent.parent / "data" / "chronos.duckdb"
    db = ChronosDB(str(db_path))
    ingestion = SecEdgarIngestion(db=db)
    batch_results = ingestion.ingest_all(force_fetch=False)
    for sym, count in batch_results.items():
        print(f"Successfully ingested {count} records for {sym} ({COMPANY_REGISTRY[sym]['name']})")
