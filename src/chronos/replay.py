from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional, Union
import pandas as pd
import pyarrow.parquet as pq

from src.chronos.db import ChronosDB
from src.chronos.timekeeper import TimeKeeper


class TemporalLeakageError(RuntimeError):
    """Raised when future data is accessed (t_avail > current_sim_time)."""
    pass


def parse_step_size(step: Union[timedelta, str, int]) -> timedelta:
    """Parse step size specification into timedelta."""
    if isinstance(step, timedelta):
        return step
    if isinstance(step, (int, float)):
        return timedelta(seconds=int(step))
    if isinstance(step, str):
        s = step.strip().lower()
        m = re.match(r"^(\d+)\s*([a-z]+)$", s)
        if not m:
            raise ValueError(f"Invalid step size string format: '{step}'")
        val, unit = int(m.group(1)), m.group(2)
        if unit in ("s", "sec", "second", "seconds"):
            return timedelta(seconds=val)
        elif unit in ("m", "min", "minute", "minutes"):
            return timedelta(minutes=val)
        elif unit in ("h", "hr", "hour", "hours"):
            return timedelta(hours=val)
        elif unit in ("d", "day", "days"):
            return timedelta(days=val)
        elif unit in ("w", "wk", "week", "weeks"):
            return timedelta(weeks=val)
        else:
            raise ValueError(f"Unknown time unit in step size: '{unit}'")
    raise TypeError(f"Unsupported type for step size: {type(step)}")


def ensure_utc(dt: Union[datetime, str]) -> datetime:
    """Ensure datetime is timezone-aware UTC."""
    if isinstance(dt, str):
        # Convert iso format or date string
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class HistoricalReplayEngine:
    """
    Chronos Historical Replay Engine.
    Steps through historical time and provides market data + fundamentals
    enforcing strict Point-In-Time (PIT) invariants without future leakage.
    """

    def __init__(
        self,
        db: ChronosDB,
        symbol: str = "BTCUSDT",
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None,
        step_size: Union[timedelta, str, int] = "1h",
        market_data: Optional[Union[str, Path, pd.DataFrame]] = None,
        interval: Optional[str] = None,
        sim_run_id: Optional[str] = None,
    ):
        self.db = db
        self.symbol = symbol
        self.sim_run_id = sim_run_id or f"SIM_{symbol}_{int(datetime.now(timezone.utc).timestamp())}"
        self.step_size = parse_step_size(step_size)

        # Parse interval
        self.interval = interval or self._infer_interval_from_step(self.step_size)

        # Initialize market data
        self.market_df: pd.DataFrame = self._load_market_data(market_data)

        # Time bounds
        if start_time is not None:
            self.start_time = ensure_utc(start_time)
        elif not self.market_df.empty and "t_event" in self.market_df.columns:
            self.start_time = self.market_df["t_event"].iloc[0].to_pydatetime()
        else:
            self.start_time = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)

        if end_time is not None:
            self.end_time = ensure_utc(end_time)
        elif not self.market_df.empty and "t_event" in self.market_df.columns:
            self.end_time = self.market_df["t_event"].iloc[-1].to_pydatetime()
        else:
            self.end_time = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        self._current_sim_time = self.start_time
        self.timekeeper = TimeKeeper(start_time=self.start_time)
        self._is_paused = False

    def _infer_interval_from_step(self, step: timedelta) -> str:
        if step >= timedelta(days=1):
            return "1d"
        return "1h"

    def _load_market_data(
        self, market_data: Optional[Union[str, Path, pd.DataFrame]]
    ) -> pd.DataFrame:
        if isinstance(market_data, pd.DataFrame):
            df = market_data.copy()
        else:
            path = (
                Path(market_data)
                if market_data is not None
                else Path(f"data/parquet/market/CRYPTO/{self.symbol}")
            )
            if not path.exists():
                return pd.DataFrame()

            try:
                if path.is_file():
                    table = pq.read_table(path)
                else:
                    # Directory with potential interval partitions
                    filters = [("interval", "=", self.interval)] if self.interval else None
                    try:
                        table = pq.read_table(path, filters=filters)
                    except Exception:
                        table = pq.read_table(path)
                df = table.to_pandas()
            except Exception as e:
                print(f"Warning: could not load parquet market data from {path}: {e}")
                return pd.DataFrame()

        if df.empty:
            return df

        # Ensure datetime columns are timezone-aware UTC
        for col in ["t_event", "t_pub", "t_avail", "t_ingest"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

        if "t_avail" in df.columns:
            df = df.sort_values("t_avail").reset_index(drop=True)
        return df

    @property
    def current_sim_time(self) -> datetime:
        return self._current_sim_time

    @property
    def active_asset(self) -> str:
        return self.symbol

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def pause(self) -> None:
        """Pause replay progression."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume replay progression."""
        self._is_paused = False

    def is_finished(self) -> bool:
        """Check if simulation has reached or exceeded end_time."""
        return self._current_sim_time >= self.end_time

    def get_current_bar(self, sim_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest market bar whose t_avail <= target_sim_time.
        Strictly enforces PIT invariant: no future bars (t_avail > sim_time) can be returned.
        """
        target_time = sim_time or self._current_sim_time
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)

        if self.market_df.empty or "t_avail" not in self.market_df.columns:
            return None

        # Binary search for greatest t_avail <= target_time
        idx = int(self.market_df["t_avail"].searchsorted(pd.to_datetime(target_time), side="right")) - 1  # type: ignore
        if idx < 0:
            return None

        bar = self.market_df.iloc[idx].to_dict()

        # Invariant check
        bar_avail = bar.get("t_avail")
        if bar_avail is not None:
            if isinstance(bar_avail, pd.Timestamp):
                bar_avail = bar_avail.to_pydatetime()
            if bar_avail.tzinfo is None:
                bar_avail = bar_avail.replace(tzinfo=timezone.utc)
            if bar_avail > target_time:
                raise TemporalLeakageError(
                    f"Market bar temporal leakage: t_avail ({bar_avail}) > target_sim_time ({target_time})"
                )

        return bar

    def get_current_fundamentals(
        self, sim_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get point-in-time fundamentals snapshot.
        Strictly enforces PIT invariant: no record with t_avail > sim_time can be returned.
        """
        target_time = sim_time or self._current_sim_time
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)

        snapshot = self.db.get_crypto_snapshot(self.symbol, target_time)

        # Invariant check
        for item in snapshot:
            avail = item.get("t_avail")
            if avail is not None:
                if isinstance(avail, pd.Timestamp):
                    avail = avail.to_pydatetime()
                if avail.tzinfo is None:
                    avail = avail.replace(tzinfo=timezone.utc)
                if avail > target_time:
                    raise TemporalLeakageError(
                        f"Fundamental temporal leakage: t_avail ({avail}) > target_sim_time ({target_time})"
                    )

        return snapshot

    @property
    def current_bar(self) -> Optional[Dict[str, Any]]:
        return self.get_current_bar()

    @property
    def current_fundamentals(self) -> List[Dict[str, Any]]:
        return self.get_current_fundamentals()

    def step(self, advance: bool = True) -> Dict[str, Any]:
        """
        Advance simulation by step_size and return the current state.
        Manual step advances even if engine is paused.
        """
        if advance and not self.is_finished():
            new_time = self._current_sim_time + self.step_size
            self._current_sim_time = min(new_time, self.end_time)
            self.timekeeper.set_time(self._current_sim_time)

        bar = self.get_current_bar()
        fundamentals = self.get_current_fundamentals()

        return {
            "sim_run_id": self.sim_run_id,
            "sim_time": self._current_sim_time,
            "symbol": self.symbol,
            "bar": bar,
            "fundamentals": fundamentals,
            "is_paused": self.is_paused,
            "is_finished": self.is_finished(),
        }

    def reset(self) -> None:
        """Reset engine back to start_time."""
        self._current_sim_time = self.start_time
        self.timekeeper.set_time(self.start_time)
        self._is_paused = False

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.is_finished():
            raise StopIteration
        return self.step()
