from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional
import pandas as pd

from src.agents.base import BaseAgent
from src.chronos.replay import TemporalLeakageError


@dataclass
class QuantitativeAnalysis:
    asset_id: str
    sim_time: datetime
    score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class QuantSpecialist(BaseAgent):
    """
    Agent 2: Quantitative Specialist.
    Consumes market bars and PIT fundamentals at current_sim_time.
    Computes TVL momentum, Price-to-TVL, realized volatility/ATR, and SMA trend divergence.
    """

    def __init__(self, agent_id: str = "agent_quant_specialist", name: str = "Quant Specialist"):
        super().__init__(agent_id=agent_id, name=name)

    def evaluate(self, context: Dict[str, Any]) -> QuantitativeAnalysis:
        sim_time = context.get("sim_time") or context.get("current_sim_time")
        if sim_time is None:
            sim_time = datetime.now(timezone.utc)
        elif sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)

        asset_id = context.get("asset_id") or context.get("symbol") or "CRYPTO_BTC"
        replay_engine = context.get("replay_engine")
        db = context.get("db")

        # 1. Fetch market bars up to sim_time
        bars = self._get_bars(context, replay_engine, sim_time)

        # 2. Fetch fundamentals up to sim_time
        fundamentals = self._get_fundamentals(context, replay_engine, db, asset_id, sim_time)

        # 3. Compute Metrics
        metrics: Dict[str, Any] = {}
        signals: List[str] = []

        current_bar = bars[-1] if bars else context.get("bar") or {}
        current_price = float(current_bar.get("close", 0.0)) if current_bar else 0.0
        metrics["current_price"] = current_price
        metrics["volume"] = float(current_bar.get("volume", 0.0)) if current_bar else 0.0
        metrics["quote_volume"] = float(current_bar.get("quote_volume", 0.0)) if current_bar else 0.0

        # TVL metrics
        tvl_now, tvl_7d_mom, tvl_30d_mom, pt_tvl = self._compute_tvl_metrics(
            fundamentals, current_price, sim_time, db, asset_id
        )
        metrics["tvl"] = tvl_now
        metrics["tvl_7d_momentum"] = tvl_7d_mom
        metrics["tvl_30d_momentum"] = tvl_30d_mom
        metrics["price_to_tvl"] = pt_tvl

        if tvl_7d_mom is not None:
            if tvl_7d_mom > 0.05:
                signals.append("TVL_7D_STRONG_INFLOW")
            elif tvl_7d_mom < -0.05:
                signals.append("TVL_7D_OUTFLOW")

        # Volatility and ATR (14-period)
        realized_vol, atr = self._compute_volatility_and_atr(bars, period=14)
        metrics["realized_volatility"] = realized_vol
        metrics["atr"] = atr

        if realized_vol > 0.08:
            signals.append("HIGH_VOLATILITY_REGIME")
        elif realized_vol < 0.02:
            signals.append("LOW_VOLATILITY_COMPRESSION")

        # SMA trend divergence (20/50/200)
        sma_20, sma_50, sma_200, trend_div = self._compute_sma_divergence(bars, current_price)
        metrics["sma_20"] = sma_20
        metrics["sma_50"] = sma_50
        metrics["sma_200"] = sma_200
        metrics["trend_divergence"] = trend_div

        if sma_20 is not None and current_price > sma_20:
            signals.append("PRICE_ABOVE_SMA20")
        elif sma_20 is not None and current_price < sma_20:
            signals.append("PRICE_BELOW_SMA20")

        if sma_20 is not None and sma_50 is not None:
            if sma_20 > sma_50:
                signals.append("BULLISH_TREND_ALIGNMENT")
            else:
                signals.append("BEARISH_TREND_ALIGNMENT")

        # 4. Composite Quantitative Score [-1.0 to 1.0]
        score = self._compute_composite_score(
            current_price=current_price,
            sma_20=sma_20,
            sma_50=sma_50,
            tvl_7d_mom=tvl_7d_mom,
            realized_vol=realized_vol,
        )

        return QuantitativeAnalysis(
            asset_id=asset_id,
            sim_time=sim_time,
            score=round(score, 4),
            metrics=metrics,
            signals=signals,
            details={"bar_count": len(bars), "current_bar": current_bar},
        )

    def _get_bars(
        self, context: Dict[str, Any], replay_engine: Any, sim_time: datetime
    ) -> List[Dict[str, Any]]:
        raw_bars = context.get("bars") or context.get("historical_bars")
        if raw_bars is not None:
            if isinstance(raw_bars, pd.DataFrame):
                bars = raw_bars.to_dict(orient="records")
            else:
                bars = list(raw_bars)
        elif replay_engine is not None and hasattr(replay_engine, "get_historical_bars"):
            bars = replay_engine.get_historical_bars(limit=250, sim_time=sim_time)
        elif replay_engine is not None and hasattr(replay_engine, "get_current_bar"):
            bar = replay_engine.get_current_bar(sim_time=sim_time)
            bars = [bar] if bar else []
        elif "bar" in context and context["bar"]:
            bars = [context["bar"]]
        else:
            bars = []

        # Strict PIT check on bars
        for b in bars:
            avail = b.get("t_avail")
            if avail is not None:
                if isinstance(avail, pd.Timestamp):
                    avail = avail.to_pydatetime()
                if avail.tzinfo is None:
                    avail = avail.replace(tzinfo=timezone.utc)
                if avail > sim_time:
                    raise TemporalLeakageError(
                        f"QuantSpecialist temporal leakage: bar t_avail ({avail}) > sim_time ({sim_time})"
                    )
        return bars

    def _get_fundamentals(
        self,
        context: Dict[str, Any],
        replay_engine: Any,
        db: Any,
        asset_id: str,
        sim_time: datetime,
    ) -> List[Dict[str, Any]]:
        fundamentals = context.get("fundamentals")
        if fundamentals is not None:
            fund_list = list(fundamentals)
        elif replay_engine is not None and hasattr(replay_engine, "get_current_fundamentals"):
            fund_list = replay_engine.get_current_fundamentals(sim_time=sim_time)
        elif db is not None and hasattr(db, "get_crypto_snapshot"):
            fund_list = db.get_crypto_snapshot(asset_id, sim_time)
        else:
            fund_list = []

        # Strict PIT check on fundamentals
        for item in fund_list:
            avail = item.get("t_avail")
            if avail is not None:
                if isinstance(avail, pd.Timestamp):
                    avail = avail.to_pydatetime()
                if avail.tzinfo is None:
                    avail = avail.replace(tzinfo=timezone.utc)
                if avail > sim_time:
                    raise TemporalLeakageError(
                        f"QuantSpecialist temporal leakage: fundamental t_avail ({avail}) > sim_time ({sim_time})"
                    )
        return fund_list

    def _compute_tvl_metrics(
        self,
        fundamentals: List[Dict[str, Any]],
        current_price: float,
        sim_time: datetime,
        db: Any,
        asset_id: str,
    ):
        tvl_now: Optional[float] = None
        for item in fundamentals:
            if str(item.get("metric_id", "")).upper() == "TVL":
                tvl_now = float(item.get("metric_value", 0.0))
                break

        history = []
        if db is not None and hasattr(db, "get_fundamentals_history"):
            history = db.get_fundamentals_history(asset_id, "TVL", sim_time, lookback_days=35)

        tvl_7d_mom: Optional[float] = None
        tvl_30d_mom: Optional[float] = None

        if tvl_now is not None and history:
            t_7d = sim_time - timedelta(days=7)
            t_30d = sim_time - timedelta(days=30)

            past_7d = [h for h in history if h["t_avail"] <= t_7d]
            if past_7d:
                tvl_7d_val = float(past_7d[-1]["metric_value"])
                if tvl_7d_val > 0:
                    tvl_7d_mom = (tvl_now - tvl_7d_val) / tvl_7d_val

            past_30d = [h for h in history if h["t_avail"] <= t_30d]
            if past_30d:
                tvl_30d_val = float(past_30d[-1]["metric_value"])
                if tvl_30d_val > 0:
                    tvl_30d_mom = (tvl_now - tvl_30d_val) / tvl_30d_val

        price_to_tvl = (current_price / tvl_now) if (tvl_now and tvl_now > 0 and current_price > 0) else None

        return tvl_now, tvl_7d_mom, tvl_30d_mom, price_to_tvl

    def _compute_volatility_and_atr(
        self, bars: List[Dict[str, Any]], period: int = 14
    ):
        if len(bars) < 2:
            return 0.0, 0.0

        sample_bars = bars[-period:] if len(bars) >= period else bars
        closes = [float(b.get("close", 0.0)) for b in sample_bars]

        # Realized Volatility: sample std dev of returns
        returns = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            if prev > 0:
                returns.append((closes[i] - prev) / prev)

        if len(returns) >= 2:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            realized_vol = math.sqrt(variance)
        elif len(returns) == 1:
            realized_vol = abs(returns[0])
        else:
            realized_vol = 0.0

        # ATR proxy over the window
        tr_list = []
        for i in range(len(sample_bars)):
            high = float(sample_bars[i].get("high", sample_bars[i].get("close", 0.0)))
            low = float(sample_bars[i].get("low", sample_bars[i].get("close", 0.0)))
            if i == 0:
                tr = high - low
            else:
                prev_close = float(sample_bars[i - 1].get("close", high))
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)

        atr = sum(tr_list) / len(tr_list) if tr_list else 0.0
        return round(realized_vol, 6), round(atr, 4)

    def _compute_sma_divergence(
        self, bars: List[Dict[str, Any]], current_price: float
    ):
        if not bars:
            return None, None, None, None

        closes = [float(b.get("close", 0.0)) for b in bars]

        sma_20 = sum(closes[-20:]) / len(closes[-20:]) if len(closes) >= 5 else None
        sma_50 = sum(closes[-50:]) / len(closes[-50:]) if len(closes) >= 20 else None
        sma_200 = sum(closes[-200:]) / len(closes[-200:]) if len(closes) >= 50 else None

        trend_div = None
        if sma_20 and sma_20 > 0:
            trend_div = round((current_price - sma_20) / sma_20, 6)

        return (
            round(sma_20, 4) if sma_20 else None,
            round(sma_50, 4) if sma_50 else None,
            round(sma_200, 4) if sma_200 else None,
            trend_div,
        )

    def _compute_composite_score(
        self,
        current_price: float,
        sma_20: Optional[float],
        sma_50: Optional[float],
        tvl_7d_mom: Optional[float],
        realized_vol: float,
    ) -> float:
        score = 0.0

        # Trend component (-0.4 to +0.4)
        if sma_20 is not None and sma_20 > 0:
            diff_pct = (current_price - sma_20) / sma_20
            score += max(-0.3, min(0.3, diff_pct * 3.0))

            if sma_50 is not None:
                if sma_20 > sma_50:
                    score += 0.1
                else:
                    score -= 0.1

        # TVL Momentum component (-0.4 to +0.4)
        if tvl_7d_mom is not None:
            tvl_contrib = max(-0.4, min(0.4, tvl_7d_mom * 2.0))
            score += tvl_contrib

        # Volatility dampening if extreme
        if realized_vol > 0.10:
            score *= 0.5

        return max(-1.0, min(1.0, score))
