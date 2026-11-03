import argparse
import io
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
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

    def download_monthly_range(
        self,
        symbol: str,
        interval: str,
        start_year: int = 2023,
        end_year: int = 2024,
    ) -> pd.DataFrame:
        """Download and combine monthly klines across a year range."""
        frames: List[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                date_str = f"{year}-{month:02d}"
                try:
                    df_month = self.download_and_parse(
                        symbol=symbol,
                        interval=interval,
                        date_str=date_str,
                        market_type="monthly",
                    )
                    frames.append(df_month)
                    print(f"[{symbol}][{interval}] Ingested {date_str}: {len(df_month)} rows")
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        print(f"[{symbol}][{interval}] Month {date_str} not found on S3, skipping.")
                    else:
                        print(f"[{symbol}][{interval}] HTTP error on {date_str}: {e}")
                except Exception as e:
                    print(f"[{symbol}][{interval}] Error downloading {date_str}: {e}")

        if not frames:
            return pd.DataFrame(columns=BINANCE_KLINES_COLUMNS)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        return combined

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
            df.to_parquet(
                out_path,
                index=False,
                partition_cols=["interval"],
                existing_data_behavior="delete_matching",
                engine="pyarrow",
            )
        else:
            file_path = out_path / "klines.parquet"
            df.to_parquet(file_path, index=False, engine="pyarrow")

        return out_path

    def run_bulk_monthly_ingestion(
        self,
        symbol: str = "BTCUSDT",
        intervals: Optional[List[str]] = None,
        start_year: int = 2023,
        end_year: int = 2024,
        output_dir: Union[str, Path] = "data/parquet/market/CRYPTO/BTCUSDT",
    ) -> Dict[str, int]:
        """Download and save bulk monthly klines for specified intervals and year range."""
        if intervals is None:
            intervals = ["1d", "1h"]

        results = {}
        for interval in intervals:
            df_interval = self.download_monthly_range(
                symbol=symbol,
                interval=interval,
                start_year=start_year,
                end_year=end_year,
            )
            if not df_interval.empty:
                self.export_to_parquet(df_interval, output_dir=output_dir, partition_by_interval=True)
            results[interval] = len(df_interval)

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk ingestion for Binance monthly klines")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)")
    parser.add_argument("--intervals", nargs="+", default=["1d", "1h"], help="Intervals (default: 1d 1h)")
    parser.add_argument("--start-year", type=int, default=2023, help="Start year (default: 2023)")
    parser.add_argument("--end-year", type=int, default=2024, help="End year (default: 2024)")
    parser.add_argument("--output-dir", default="data/parquet/market/CRYPTO/BTCUSDT", help="Output directory")
    args = parser.parse_args()

    ingester = BinanceS3Ingestion(lag_seconds=10)
    results = ingester.run_bulk_monthly_ingestion(
        symbol=args.symbol,
        intervals=args.intervals,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
    )
    print("Ingestion results:", results)


if __name__ == "__main__":
    main()
