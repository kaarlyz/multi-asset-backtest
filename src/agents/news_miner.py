from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional
import pandas as pd

from src.agents.base import BaseAgent
from src.chronos.replay import TemporalLeakageError


@dataclass
class NewsSentimentAnalysis:
    asset_id: str
    sim_time: datetime
    score: float
    rolling_sentiment_7d: float
    news_count: int
    event_breakdown: Dict[str, int] = field(default_factory=dict)
    source_weights: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = {
    "bloomberg": 1.0,
    "reuters": 1.0,
    "wsj": 1.0,
    "coindesk": 0.85,
    "theblock": 0.85,
    "defillama": 0.90,
    "binance": 0.80,
    "cointelegraph": 0.75,
    "decrypt": 0.75,
}

CATEGORY_KEYWORDS = {
    "REGULATORY": [
        "sec", "cftc", "regulation", "regulatory", "compliance", "lawsuit",
        "legal", "court", "ban", "license", "subpoena", "judge", "doj"
    ],
    "HACK": [
        "hack", "exploit", "breach", "stolen", "vulnerability", "drain",
        "rug", "attack", "compromise", "phishing", "heist"
    ],
    "MACRO": [
        "fed", "inflation", "rate", "interest", "cpi", "fomc", "powell",
        "gdp", "recession", "treasury", "macro", "unemployment"
    ],
    "ADOPTION": [
        "adopt", "etf", "institutional", "partner", "partnership", "launch",
        "integrate", "payment", "approval", "approved", "mainstream", "custody"
    ],
}


class NewsMiner(BaseAgent):
    """
    Agent 3: News Miner.
    Retrieves news strictly adhering to t_avail <= current_sim_time.
    Computes 7-day rolling sentiment aggregate, categorizes events,
    and weights by source reliability.
    """

    def __init__(
        self,
        agent_id: str = "agent_news_miner",
        name: str = "News Miner",
        source_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(agent_id=agent_id, name=name)
        self.source_weights = {**DEFAULT_SOURCE_WEIGHTS, **(source_weights or {})}

    def evaluate(self, context: Dict[str, Any]) -> NewsSentimentAnalysis:
        sim_time = context.get("sim_time") or context.get("current_sim_time")
        if sim_time is None:
            sim_time = datetime.now(timezone.utc)
        elif sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)

        asset_id = context.get("asset_id") or context.get("symbol") or "CRYPTO_BTC"
        replay_engine = context.get("replay_engine")
        db = context.get("db")

        # 1. Retrieve news
        news_items = self._get_news_items(context, replay_engine, db, asset_id, sim_time)

        # 2. Strict PIT verification
        for item in news_items:
            t_avail = item.get("t_avail")
            if t_avail is not None:
                if isinstance(t_avail, pd.Timestamp):
                    t_avail = t_avail.to_pydatetime()
                if t_avail.tzinfo is None:
                    t_avail = t_avail.replace(tzinfo=timezone.utc)
                if t_avail > sim_time:
                    raise TemporalLeakageError(
                        f"NewsMiner temporal leakage: news t_avail ({t_avail}) > sim_time ({sim_time})"
                    )

        # 3. Categorize events and aggregate 7d sentiment
        cutoff_7d = sim_time - timedelta(days=7)
        event_breakdown = {"REGULATORY": 0, "HACK": 0, "MACRO": 0, "ADOPTION": 0, "OTHER": 0}
        processed_events = []

        weighted_sentiment_sum = 0.0
        weight_sum = 0.0

        for item in news_items:
            # Check 7-day rolling window
            t_avail = item.get("t_avail")
            if t_avail is not None:
                if isinstance(t_avail, pd.Timestamp):
                    t_avail = t_avail.to_pydatetime()
                if t_avail.tzinfo is None:
                    t_avail = t_avail.replace(tzinfo=timezone.utc)
                if t_avail < cutoff_7d:
                    continue

            title = str(item.get("title", ""))
            content = str(item.get("raw_content", ""))
            category = self._categorize_news(title, content)
            event_breakdown[category] = event_breakdown.get(category, 0) + 1

            source = str(item.get("source", "")).lower()
            weight = self.source_weights.get(source, 0.70)

            raw_sentiment = item.get("sentiment_score")
            sentiment = float(raw_sentiment) if raw_sentiment is not None else 0.0

            # Domain invariant adjustment: unaddressed HACK is strongly negative
            if category == "HACK" and sentiment > -0.5:
                sentiment = -0.8

            weighted_sentiment_sum += sentiment * weight
            weight_sum += weight

            processed_events.append({
                "news_id": item.get("news_id"),
                "title": title,
                "category": category,
                "source": source,
                "weight": weight,
                "sentiment_score": sentiment,
                "t_avail": t_avail.isoformat() if t_avail else None,
            })

        if weight_sum > 0:
            rolling_7d = weighted_sentiment_sum / weight_sum
        else:
            rolling_7d = 0.0

        final_score = max(-1.0, min(1.0, rolling_7d))

        return NewsSentimentAnalysis(
            asset_id=asset_id,
            sim_time=sim_time,
            score=round(final_score, 4),
            rolling_sentiment_7d=round(rolling_7d, 4),
            news_count=len(processed_events),
            event_breakdown=event_breakdown,
            source_weights=self.source_weights,
            events=processed_events,
        )

    def _get_news_items(
        self,
        context: Dict[str, Any],
        replay_engine: Any,
        db: Any,
        asset_id: str,
        sim_time: datetime,
    ) -> List[Dict[str, Any]]:
        if "news" in context and context["news"] is not None:
            return list(context["news"])
        if replay_engine is not None and hasattr(replay_engine, "get_news"):
            return replay_engine.get_news(lookback_days=7, sim_time=sim_time)
        if db is not None and hasattr(db, "get_news_pit"):
            return db.get_news_pit(symbol_or_asset=asset_id, sim_time=sim_time, lookback_days=7)
        return []

    def _categorize_news(self, title: str, content: str) -> str:
        title_lower = title.lower()
        content_lower = content.lower()
        best_cat = "OTHER"
        max_score = 0
        for category, keywords in CATEGORY_KEYWORDS.items():
            cat_score = 0
            for kw in keywords:
                cat_score += len(re.findall(r"\b" + re.escape(kw) + r"\b", title_lower)) * 3
                cat_score += len(re.findall(r"\b" + re.escape(kw) + r"\b", content_lower))
            if cat_score > max_score:
                max_score = cat_score
                best_cat = category
        return best_cat
