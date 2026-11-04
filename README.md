# Multi-Asset Fundamental Intelligence & PIT Backtesting Engine 📊🏛️

> Institutional-grade Point-in-Time (PIT) backtesting and fundamental intelligence infrastructure built on **Python**, **DuckDB**, and **Apache Parquet**. Designed to rigorously evaluate multi-asset quantitative strategies and fundamental disclosures without look-ahead bias, survivorship bias, or post-facto data revisions.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.0+-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Apache Arrow](https://img.shields.io/badge/PyArrow-15.0+-E05A47?logo=apachearrow&logoColor=white)](https://arrow.apache.org/)
[![Pytest](https://img.shields.io/badge/Tests-47%20Passed-brightgreen?logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)

---

## 🏛️ System Philosophy & Invariants

Simulating quantitative performance on modern equities and crypto requires absolute temporal integrity. Standard backtesters suffer from **Look-Ahead Bias** (using revised statements or unannounced data) and **Survivorship Bias**. 

This engine enforces four immutable architectural invariants:
1. **Point-in-Time Availability (`t_avail <= simulation_timestamp`):** Financial disclosures (e.g. SEC 10-K / 10-Q) become visible to the model **strictly at the exact SEC filing acceptance timestamp**, not the fiscal period end date.
2. **Column-Oriented OLAP Storage:** DuckDB + Apache Parquet column storage provides sub-millisecond range scans across multi-year tick/macro series.
3. **Independent Risk Auditor:** 5-agent orchestration topology where the Risk Auditor maintains unconditional veto power over all proposed signals.
4. **Strict Zero Live-Money Execution:** Dedicated purely to paper simulation and scientific backtesting.

---

## 🏗️ Multi-Agent Architecture

```text
 ┌──────────────────────┐
 │ Chronos Timekeeper   │  Enforces monotonic PIT time steps (t_avail)
 └──────────┬───────────┘
            │
            ├──────────────────────────┐
            ▼                          ▼
 ┌──────────────────────┐   ┌──────────────────────┐
 │ Quant Specialist     │   │ Fundamental & News   │
 │ - Technical Factors  │   │ - SEC EDGAR (10-K/Q) │
 │ - Volatility & MFE   │   │ - Derived Metrics    │
 └──────────┬───────────┘   └──────────┬───────────┘
            │                          │
            └───────────┬──────────────┘
                        ▼
 ┌─────────────────────────────────────────────────┐
 │ Independent Risk Auditor (Veto Power)           │
 │ - Margin & Drawdown Ceiling (Max 2.0% Risk)     │
 └──────────────────────┬──────────────────────────┘
                        ▼
 ┌─────────────────────────────────────────────────┐
 │ Report Card Synthesizer & Telegram Bridge       │
 │ - Multi-statement derivation (standalone Q4)   │
 │ - DuckDB Decision Ledger                        │
 └─────────────────────────────────────────────────┘
```

---

## 📦 Ingestion Engines

- **US Equities (SEC EDGAR PIT):** Automatically ingests multi-company company-facts filings across diverse fiscal year calendars (Apple, Microsoft, Tesla). Resolves standalone Q4 revenue derivation from 10-K annual and 10-Q 9M filings.
- **Crypto & Macro:** Pre-configured connectors for Binance Public S3 (market data), DefiLlama (historical TVL & protocol revenue), and ALFRED (vintage macroeconomic series without future revisions).

---

## 🚀 Quick Start & Verification

### 1. Environment Setup
Requires Python 3.10+ (tested on Python 3.12):

```bash
git clone https://github.com/kaarlyz/multi-asset-backtest.git
cd multi-asset-backtest

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy sample configuration:
```bash
cp .env.example .env
```
*(Optionally provide `TELEGRAM_BOT_TOKEN` for the Telegram interface or leave as default for headless operation).*

### 3. Run the Automated Test Suite
Runs all 47 unit & integration tests covering PIT temporal invariants, EDGAR vertical slice parsing, and multi-agent coordination:

```bash
pytest
```
*Expected result: 47 passed in ~5 seconds.*

### 4. Running a Backtest Replay
```bash
python3 src/main.py --mode simulation --symbol AAPL --start 2023-01-01 --end 2024-01-01
```

---

## 📁 Repository Structure

```text
multi-asset-backtest/
├── src/
│   ├── agents/               # 5-agent cooperative orchestration system
│   │   ├── auditor.py        # Independent risk auditor with veto logic
│   │   ├── chronos.py        # Point-in-Time temporal clock
│   │   ├── quant.py          # Factor models & technical indicators
│   │   └── synthesizer.py    # Report card synthesis & decision ledger
│   ├── chronos/              # Temporal engine & DuckDB schema management
│   │   ├── db.py             # DuckDB connection & connection pool
│   │   └── schema.sql        # DDL for PIT facts and dimension tables
│   ├── ingestion/            # Raw data ingestion pipeline
│   │   ├── sec_edgar.py      # SEC EDGAR PIT company-facts parser
│   │   └── statements.py     # Multi-statement semantic normalization
│   └── telegram/             # Bot notification & interactive command interface
├── tests/                    # 47 automated Pytest test cases
├── config/                   # Risk models & asset metadata
├── report.md                 # Complete Master Engineering Specification
├── requirements.txt          # Python dependencies (DuckDB, PyArrow, Pandas)
└── pytest.ini
```

---

## 📄 License
MIT License. Created by [kaarlyz](https://github.com/kaarlyz).
