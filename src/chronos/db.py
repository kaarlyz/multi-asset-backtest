import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import duckdb


class ChronosDB:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self.init_schema()

    def init_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()
        self.conn.execute(sql)

    def insert_asset(
        self,
        asset_id: str,
        asset_class: str,
        symbol: str,
        base_asset: str,
        quote_asset: str,
        is_active: bool = True,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO dim_assets 
            (asset_id, asset_class, symbol, base_asset, quote_asset, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [asset_id, asset_class, symbol, base_asset, quote_asset, is_active],
        )

    def insert_fundamental_pit(
        self,
        metric_id: str,
        asset_id: str,
        metric_value: float,
        t_event: datetime,
        t_pub: datetime,
        t_ingest: datetime,
        t_avail: datetime,
        source: str,
        provenance_hash: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO fact_fundamentals_pit
            (metric_id, asset_id, metric_value, t_event, t_pub, t_ingest, t_avail, source, provenance_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                metric_id,
                asset_id,
                metric_value,
                t_event,
                t_pub,
                t_ingest,
                t_avail,
                source,
                provenance_hash,
            ],
        )

    def insert_news_pit(
        self,
        news_id: str,
        title: str,
        source: str,
        url: Optional[str],
        raw_content: Optional[str],
        sentiment_score: Optional[float],
        tagged_symbols: Optional[List[str]],
        t_event: datetime,
        t_pub: datetime,
        t_ingest: datetime,
        t_avail: datetime,
        content_hash: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO fact_news_pit
            (news_id, title, source, url, raw_content, sentiment_score, tagged_symbols, t_event, t_pub, t_ingest, t_avail, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                news_id,
                title,
                source,
                url,
                raw_content,
                sentiment_score,
                tagged_symbols or [],
                t_event,
                t_pub,
                t_ingest,
                t_avail,
                content_hash,
            ],
        )

    def record_decision(
        self,
        decision_id: str,
        sim_run_id: str,
        sim_time: datetime,
        asset_id: str,
        interface: Optional[str],
        actor_id: Optional[str],
        action: str,
        agent_confidence: Optional[float] = None,
        evidence_snapshot: Optional[Union[Dict[str, Any], str]] = None,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_size: Optional[float] = None,
        status: Optional[str] = "PENDING",
    ) -> None:
        if isinstance(evidence_snapshot, dict):
            evidence_json = json.dumps(evidence_snapshot)
        else:
            evidence_json = evidence_snapshot

        # Strictly enforces primary key idempotency (fails on duplicate decision_id)
        self.conn.execute(
            """
            INSERT INTO decision_ledger
            (decision_id, sim_run_id, sim_time, asset_id, interface, actor_id, action,
             agent_confidence, evidence_snapshot, entry_price, stop_loss, take_profit, position_size, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                decision_id,
                sim_run_id,
                sim_time,
                asset_id,
                interface,
                actor_id,
                action,
                agent_confidence,
                evidence_json,
                entry_price,
                stop_loss,
                take_profit,
                position_size,
                status,
            ],
        )

    def get_crypto_snapshot(
        self,
        asset: str,
        sim_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        PIT snapshot query: NEVER returns records where t_avail > sim_time.
        Matches either asset_id or symbol in dim_assets.
        Returns the latest point-in-time fundamentals for each metric.
        """
        # Ensure sim_time has timezone if not present
        if sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)

        query = """
            WITH resolved_assets AS (
                SELECT asset_id FROM dim_assets
                WHERE asset_id = $1 OR symbol = $1
                UNION
                SELECT $1 AS asset_id
            ),
            pit_candidates AS (
                SELECT f.*,
                       ROW_NUMBER() OVER(PARTITION BY f.metric_id, f.asset_id ORDER BY f.t_avail DESC, f.t_event DESC) as rn
                FROM fact_fundamentals_pit f
                JOIN resolved_assets ra ON f.asset_id = ra.asset_id
                WHERE f.t_avail <= $2
            )
            SELECT metric_id, asset_id, metric_value, t_event, t_pub, t_ingest, t_avail, source, provenance_hash
            FROM pit_candidates
            WHERE rn = 1
            ORDER BY t_avail DESC
        """
        cursor = self.conn.execute(query, [asset, sim_time])
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def close(self) -> None:
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
