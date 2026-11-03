import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import requests

# Ensure project root in sys.path when executed directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chronos.db import ChronosDB


DEFAULT_CHAIN_MAPPING = {
    "Bitcoin": {
        "asset_id": "CRYPTO_BTC",
        "symbol": "BTCUSDT",
        "base": "BTC",
        "quote": "USDT",
    },
    "Ethereum": {
        "asset_id": "CRYPTO_ETH",
        "symbol": "ETHUSDT",
        "base": "ETH",
        "quote": "USDT",
    },
}


class DefiLlamaIngestion:
    BASE_URL = "https://api.llama.fi"
    DEFAULT_LAG_MINUTES = 5

    def __init__(self, base_url: Optional[str] = None, lag_minutes: int = DEFAULT_LAG_MINUTES):
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.lag_minutes = lag_minutes

    def fetch_historical_chain_tvl(self, chain: str) -> List[Dict[str, Any]]:
        """Fetch historical TVL for a given blockchain (e.g. 'Bitcoin', 'Ethereum')."""
        url = f"{self.base_url}/v2/historicalChainTvl/{chain}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError(f"Unexpected response structure from DefiLlama: {type(data)}")
        return data

    def parse_tvl_records(
        self,
        raw_items: List[Dict[str, Any]],
        asset_id: str,
        metric_id: str = "TVL",
        source: str = "defillama",
    ) -> List[Dict[str, Any]]:
        """
        Convert raw DefiLlama points to PIT records.
        Strict publication lag: t_avail = t_pub + 5m (where t_pub is snapshot date).
        """
        ingest_time = datetime.now(timezone.utc)
        parsed = []

        for item in raw_items:
            date_ts = int(item["date"])
            val = float(item["tvl"])
            t_event = datetime.fromtimestamp(date_ts, tz=timezone.utc)
            t_pub = t_event
            t_avail = t_pub + timedelta(minutes=self.lag_minutes)

            raw_hash_input = f"{source}:{asset_id}:{metric_id}:{date_ts}:{val}".encode("utf-8")
            prov_hash = hashlib.sha256(raw_hash_input).hexdigest()

            parsed.append(
                {
                    "metric_id": metric_id,
                    "asset_id": asset_id,
                    "metric_value": val,
                    "t_event": t_event,
                    "t_pub": t_pub,
                    "t_ingest": ingest_time,
                    "t_avail": t_avail,
                    "source": source,
                    "provenance_hash": prov_hash,
                }
            )

        return parsed

    def ingest_chain_tvl_to_duckdb(
        self,
        db: ChronosDB,
        chain: str,
        asset_id: str,
        limit: Optional[int] = None,
    ) -> int:
        """Fetch and persist historical TVL records into DuckDB fact_fundamentals_pit."""
        raw_records = self.fetch_historical_chain_tvl(chain)
        if limit and limit > 0:
            raw_records = raw_records[-limit:]

        parsed_records = self.parse_tvl_records(raw_records, asset_id=asset_id)
        count = 0
        for rec in parsed_records:
            try:
                db.insert_fundamental_pit(**rec)
                count += 1
            except Exception as e:
                # Primary key duplicate / existing entry skipped
                pass
        return count

    def run_tvl_ingestion(
        self,
        db: Optional[ChronosDB] = None,
        db_path: str = "data/chronos.duckdb",
        chain_map: Optional[Dict[str, Dict[str, str]]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, int]:
        """Ingest TVL for configured chains into ChronosDB."""
        chains = chain_map or DEFAULT_CHAIN_MAPPING
        should_close = False
        if db is None:
            db = ChronosDB(db_path)
            should_close = True

        results = {}
        try:
            for chain_name, meta in chains.items():
                asset_id = meta["asset_id"]
                db.insert_asset(
                    asset_id=asset_id,
                    asset_class="CRYPTO",
                    symbol=meta["symbol"],
                    base_asset=meta["base"],
                    quote_asset=meta["quote"],
                    is_active=True,
                )
                cnt = self.ingest_chain_tvl_to_duckdb(
                    db=db,
                    chain=chain_name,
                    asset_id=asset_id,
                    limit=limit,
                )
                results[chain_name] = cnt
                print(f"[DefiLlama] Ingested {cnt} records for {chain_name} ({asset_id})")
        finally:
            if should_close:
                db.close()

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest DefiLlama historical chain TVL into DuckDB")
    parser.add_argument("--db-path", default="data/chronos.duckdb", help="Path to DuckDB database")
    parser.add_argument("--chains", nargs="+", default=["Bitcoin", "Ethereum"], help="Chains to ingest")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of most recent records per chain")
    args = parser.parse_args()

    active_map = {c: DEFAULT_CHAIN_MAPPING[c] for c in args.chains if c in DEFAULT_CHAIN_MAPPING}
    ingester = DefiLlamaIngestion()
    results = ingester.run_tvl_ingestion(db_path=args.db_path, chain_map=active_map, limit=args.limit)
    print("Ingestion summary:", results)


if __name__ == "__main__":
    main()
