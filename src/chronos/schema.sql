-- DuckDB Point-In-Time (PIT) Storage Schema

CREATE TABLE IF NOT EXISTS dim_assets (
    asset_id VARCHAR PRIMARY KEY,
    asset_class VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    base_asset VARCHAR NOT NULL,
    quote_asset VARCHAR NOT NULL,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS fact_fundamentals_pit (
    metric_id VARCHAR NOT NULL,
    asset_id VARCHAR NOT NULL,
    metric_value DOUBLE NOT NULL,
    t_event TIMESTAMPTZ NOT NULL,
    t_pub TIMESTAMPTZ NOT NULL,
    t_ingest TIMESTAMPTZ NOT NULL,
    t_avail TIMESTAMPTZ NOT NULL,
    source VARCHAR NOT NULL,
    provenance_hash VARCHAR,
    PRIMARY KEY (metric_id, asset_id, t_event, source)
);

CREATE TABLE IF NOT EXISTS fact_news_pit (
    news_id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    url VARCHAR,
    raw_content TEXT,
    sentiment_score DOUBLE,
    tagged_symbols VARCHAR[],
    t_event TIMESTAMPTZ NOT NULL,
    t_pub TIMESTAMPTZ NOT NULL,
    t_ingest TIMESTAMPTZ NOT NULL,
    t_avail TIMESTAMPTZ NOT NULL,
    content_hash VARCHAR
);

CREATE TABLE IF NOT EXISTS decision_ledger (
    decision_id VARCHAR PRIMARY KEY,
    sim_run_id VARCHAR NOT NULL,
    sim_time TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR NOT NULL,
    interface VARCHAR,
    actor_id VARCHAR,
    action VARCHAR NOT NULL,
    agent_confidence DOUBLE,
    evidence_snapshot JSON,
    entry_price DOUBLE,
    stop_loss DOUBLE,
    take_profit DOUBLE,
    position_size DOUBLE,
    status VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_pit_avail ON fact_fundamentals_pit(asset_id, t_avail);
CREATE INDEX IF NOT EXISTS idx_news_pit_avail ON fact_news_pit(t_avail);
CREATE INDEX IF NOT EXISTS idx_decision_ledger_sim ON decision_ledger(sim_run_id, sim_time);

CREATE TABLE IF NOT EXISTS fact_sec_fundamentals_pit (
    cik VARCHAR NOT NULL,
    entity_name VARCHAR NOT NULL,
    concept VARCHAR NOT NULL,
    canonical_metric VARCHAR,
    original_concept VARCHAR,
    accession_number VARCHAR NOT NULL,
    form VARCHAR NOT NULL,
    filed_date VARCHAR NOT NULL,
    accepted_datetime TIMESTAMPTZ NOT NULL,
    period_start VARCHAR NOT NULL,
    period_end VARCHAR NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_period VARCHAR NOT NULL,
    val DOUBLE NOT NULL,
    unit VARCHAR NOT NULL,
    PRIMARY KEY (cik, concept, accession_number, period_end, form, period_start)
);

CREATE INDEX IF NOT EXISTS idx_sec_fundamentals_pit_asof ON fact_sec_fundamentals_pit(cik, concept, accepted_datetime);
CREATE INDEX IF NOT EXISTS idx_sec_fundamentals_pit_period ON fact_sec_fundamentals_pit(cik, concept, period_end, accepted_datetime);
CREATE INDEX IF NOT EXISTS idx_sec_fundamentals_pit_canonical ON fact_sec_fundamentals_pit(cik, canonical_metric, accepted_datetime);
CREATE INDEX IF NOT EXISTS idx_sec_fundamentals_pit_canonical_period ON fact_sec_fundamentals_pit(cik, canonical_metric, period_end, accepted_datetime);

