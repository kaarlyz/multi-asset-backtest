import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union
import duckdb


class ChronosDB:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self.init_schema()

    def init_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()
        with self._lock:
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
        with self._lock:
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
            INSERT OR REPLACE INTO fact_fundamentals_pit
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

    def insert_sec_fundamental_pit(
        self,
        cik: str,
        entity_name: str,
        concept: str,
        accession_number: str,
        form: str,
        filed_date: str,
        accepted_datetime: Union[datetime, str],
        period_start: str,
        period_end: str,
        fiscal_year: int,
        fiscal_period: str,
        val: float,
        unit: str,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO fact_sec_fundamentals_pit
                (cik, entity_name, concept, accession_number, form, filed_date, accepted_datetime,
                 period_start, period_end, fiscal_year, fiscal_period, val, unit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(cik).zfill(10),
                    entity_name,
                    concept,
                    accession_number,
                    form,
                    str(filed_date),
                    accepted_datetime,
                    str(period_start or period_end or ""),
                    str(period_end),
                    int(fiscal_year),
                    fiscal_period,
                    float(val),
                    unit,
                ],
            )

    def insert_sec_fundamentals_batch(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0
        rows = [
            (
                str(r["cik"]).zfill(10),
                r["entity_name"],
                r["concept"],
                r["accession_number"],
                r["form"],
                str(r["filed_date"]),
                r["accepted_datetime"],
                str(r.get("period_start") or r.get("period_end") or ""),
                str(r["period_end"]),
                int(r["fiscal_year"]),
                r["fiscal_period"],
                float(r["val"]),
                r["unit"],
            )
            for r in records
        ]
        with self._lock:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO fact_sec_fundamentals_pit
                (cik, entity_name, concept, accession_number, form, filed_date, accepted_datetime,
                 period_start, period_end, fiscal_year, fiscal_period, val, unit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

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
        with self._lock:
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

    def has_decision(self, decision_id: str) -> bool:
        """Check if decision_id already exists in decision_ledger (for idempotency checks)."""
        with self._lock:
            row = self.conn.execute(
                "SELECT count(*) FROM decision_ledger WHERE decision_id = ?",
                [decision_id],
            ).fetchone()
            return bool(row and row[0] > 0)

    def get_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a recorded decision by decision_id."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM decision_ledger WHERE decision_id = ?",
                [decision_id],
            )
            row = cursor.fetchone()
            if not row:
                return None
            cols = [desc[0] for desc in cursor.description]
            return dict(zip(cols, row))

    def update_decision(
        self,
        decision_id: str,
        action: str,
        actor_id: str,
        interface: str = "TELEGRAM",
        status: str = "RECORDED",
        entry_price: Optional[float] = None,
    ) -> None:
        """Update decision row upon callback or execution."""
        with self._lock:
            self.conn.execute(
                """
                UPDATE decision_ledger
                SET action = ?, actor_id = ?, interface = ?, status = ?, entry_price = COALESCE(?, entry_price)
                WHERE decision_id = ?
                """,
                [action, actor_id, interface, status, entry_price, decision_id],
            )

    def update_decision_status(self, decision_id: str, status: str) -> None:
        """Update decision status in ledger."""
        with self._lock:
            self.conn.execute(
                "UPDATE decision_ledger SET status = ? WHERE decision_id = ?",
                [status, decision_id],
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

    def get_fundamentals_history(
        self,
        asset: str,
        metric_id: str,
        sim_time: datetime,
        lookback_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        PIT fundamentals historical query: NEVER returns records where t_avail > sim_time.
        Matches either asset_id or symbol in dim_assets.
        """
        if sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)

        query = """
            WITH resolved_assets AS (
                SELECT asset_id FROM dim_assets
                WHERE asset_id = $1 OR symbol = $1
                UNION
                SELECT $1 AS asset_id
            )
            SELECT f.metric_id, f.asset_id, f.metric_value, f.t_event, f.t_pub, f.t_ingest, f.t_avail, f.source
            FROM fact_fundamentals_pit f
            JOIN resolved_assets ra ON f.asset_id = ra.asset_id
            WHERE f.metric_id = $2 AND f.t_avail <= $3
        """
        params: List[Any] = [asset, metric_id, sim_time]
        if lookback_days is not None:
            lookback_time = sim_time - timedelta(days=lookback_days)
            query += " AND f.t_avail >= $4"
            params.append(lookback_time)

        query += " ORDER BY f.t_avail ASC, f.t_event ASC"
        cursor = self.conn.execute(query, params)
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def get_news_pit(
        self,
        symbol_or_asset: Optional[str] = None,
        sim_time: Optional[datetime] = None,
        lookback_days: Optional[int] = 7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve news strictly adhering to t_avail <= sim_time.
        Optionally filter by symbol/asset and lookback days.
        """
        if sim_time is None:
            sim_time = datetime.now(timezone.utc)
        elif sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)

        params: List[Any] = [sim_time]
        query = "SELECT * FROM fact_news_pit WHERE t_avail <= ?"
        if lookback_days is not None:
            lookback_time = sim_time - timedelta(days=lookback_days)
            query += " AND t_avail >= ?"
            params.append(lookback_time)

        query += " ORDER BY t_avail DESC"
        cursor = self.conn.execute(query, params)
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        news = [dict(zip(cols, row)) for row in rows]

        if symbol_or_asset:
            sym = symbol_or_asset.upper()
            filtered = []
            for item in news:
                tags = [str(t).upper() for t in (item.get("tagged_symbols") or [])]
                if not tags or sym in tags or any(sym in t or t in sym for t in tags):
                    filtered.append(item)
            return filtered
        return news

    def close(self) -> None:
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
