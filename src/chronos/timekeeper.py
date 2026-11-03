from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


class TimeKeeper:
    def __init__(self, start_time: Optional[datetime] = None):
        if start_time is None:
            self._current_time = datetime.now(timezone.utc)
        elif start_time.tzinfo is None:
            self._current_time = start_time.replace(tzinfo=timezone.utc)
        else:
            self._current_time = start_time

    @property
    def current_time(self) -> datetime:
        return self._current_time

    def step(self, seconds: int = 0, minutes: int = 0, hours: int = 0, days: int = 0) -> datetime:
        delta = timedelta(seconds=seconds, minutes=minutes, hours=hours, days=days)
        self._current_time += delta
        return self._current_time

    def set_time(self, new_time: datetime) -> None:
        if new_time.tzinfo is None:
            self._current_time = new_time.replace(tzinfo=timezone.utc)
        else:
            self._current_time = new_time

    @staticmethod
    def compute_avail_time(t_pub: datetime, lag_seconds: int = 0) -> datetime:
        if t_pub.tzinfo is None:
            t_pub = t_pub.replace(tzinfo=timezone.utc)
        return t_pub + timedelta(seconds=lag_seconds)

    def is_available(self, t_avail: datetime) -> bool:
        if t_avail.tzinfo is None:
            t_avail = t_avail.replace(tzinfo=timezone.utc)
        return t_avail <= self._current_time

    def filter_pit_records(self, records: List[Dict[str, Any]], avail_key: str = "t_avail") -> List[Dict[str, Any]]:
        """Filter out records whose t_avail > current_sim_time."""
        return [
            rec for rec in records
            if self.is_available(rec[avail_key])
        ]
