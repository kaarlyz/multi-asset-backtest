import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import requests

BINANCE_KLINES_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


class BinanceS3Ingestion:
    BASE_URL = "https://data.binance.vision/data/spot"

    def __init__(self, base_url: Optional[str] = None, lag_seconds: int = 10):
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.lag_seconds = lag_seconds

    def build_url(
        self,
        symbol: str,
        interval: str,
        date_str: str,
        market_type: str = "daily",
    ) -> str:
        """
        Build URL for binance vision klines archive.
        market_type: 'daily' (date_str: 'YYYY-MM-DD') or 'monthly' (date_str: 'YYYY-MM')
        """
        return f"{self.base_url}/{market_type}/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"

    def parse_klines_csv(
        self,
        csv_bytes: Union[bytes, str],
        symbol: str,
        interval: str,
        ingest_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Parse raw CSV bytes, apply schema and compute PIT timestamps."""
        if isinstance(csv_bytes, str):
            csv_bytes = csv_bytes.encode("utf-8")

        # Check for header
        first_line = csv_bytes.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
        has_header = "open_time" in first_line.lower()

        df = pd.read_csv(
            io.BytesIO(csv_bytes),
            header=0 if has_header else None,
            names=BINANCE_KLINES_COLUMNS if not has_header else None,
        )

        # Standardize column types
        for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")

        # PIT columns
        t_ingest = ingest_time or datetime.now(timezone.utc)
        if t_ingest.tzinfo is None:
            t_ingest = t_ingest.replace(tzinfo=timezone.utc)

        df["t_event"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["t_pub"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df["t_ingest"] = pd.to_datetime(t_ingest)
        df["t_avail"] = df["t_pub"] + pd.Timedelta(seconds=self.lag_seconds)

        df["symbol"] = symbol
        df["interval"] = interval
        if "ignore" in df.columns:
            df = df.drop(columns=["ignore"])
        return df

    def download_and_parse(
        self,
        symbol: str,
        interval: str,
        date_str: str,
        market_type: str = "daily",
    ) -> pd.DataFrame:
        """Download zip from data.binance.vision, unpack and parse into DataFrame."""
        url = self.build_url(symbol, interval, date_str, market_type)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found in zip archive from {url}")
            csv_content = z.read(csv_names[0])

        return self.parse_klines_csv(csv_content, symbol=symbol, interval=interval)

    def export_to_parquet(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path] = "data/parquet/market/CRYPTO/BTCUSDT",
        partition_by_interval: bool = True,
    ) -> Path:
        """Save DataFrame to partitioned parquet storage."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if partition_by_interval and "interval" in df.columns:
            # Partitioned parquet
            df.to_parquet(
                out_path,
                index=False,
                partition_cols=["interval"],
                engine="pyarrow",
            )
        else:
            file_path = out_path / "klines.parquet"
            df.to_parquet(file_path, index=False, engine="pyarrow")

        return out_path
