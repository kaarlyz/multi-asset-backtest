from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from src.agents.base import BaseAgent
from src.agents.news_miner import NewsSentimentAnalysis
from src.agents.quant_specialist import QuantitativeAnalysis


@dataclass
class DecisionReport:
    decision_id: str
    asset_id: str
    sim_time: datetime
    proposed_action: str  # "BUY", "SHORT", "IGNORE"
    confidence: float     # 0.0 to 1.0
    quant_analysis: QuantitativeAnalysis
    news_analysis: NewsSentimentAnalysis
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    current_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Synthesizer(BaseAgent):
    """
    Agent 4: Synthesizer.
    Fuses QuantitativeAnalysis and NewsSentimentAnalysis into a synthesized DecisionReport.
    Determines proposed action (BUY, SHORT, IGNORE), bounded confidence,
    and supporting vs contradicting evidence.
    """

    def __init__(self, agent_id: str = "agent_synthesizer", name: str = "Synthesizer"):
        super().__init__(agent_id=agent_id, name=name)

    def evaluate(self, context: Dict[str, Any]) -> DecisionReport:
        sim_time = context.get("sim_time") or context.get("current_sim_time")
        if sim_time is None:
            sim_time = datetime.now(timezone.utc)
        elif sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)

        asset_id = context.get("asset_id") or context.get("symbol") or "CRYPTO_BTC"
        decision_id = context.get("decision_id") or f"DEC_{asset_id}_{int(sim_time.timestamp())}_{uuid.uuid4().hex[:6]}"

        quant_analysis: Optional[QuantitativeAnalysis] = context.get("quant_analysis")
        news_analysis: Optional[NewsSentimentAnalysis] = context.get("news_analysis")

        if quant_analysis is None:
            raise ValueError("Synthesizer requires quant_analysis in context")
        if news_analysis is None:
            raise ValueError("Synthesizer requires news_analysis in context")

        q_score = quant_analysis.score
        n_score = news_analysis.score
        current_price = quant_analysis.metrics.get("current_price")

        supporting: List[str] = []
        contradicting: List[str] = []

        # Convergence Analysis
        is_both_bullish = q_score > 0.15 and n_score >= 0.0
        is_both_bearish = q_score < -0.15 and n_score <= 0.0
        is_divergent = (q_score > 0.15 and n_score < -0.15) or (q_score < -0.15 and n_score > 0.15)

        if is_both_bullish:
            proposed_action = "BUY"
            # Bounded high confidence on convergence
            base_conf = 0.50 + 0.30 * q_score + 0.20 * max(0.0, n_score)
            confidence = min(0.95, max(0.60, base_conf))

            supporting.append(f"Quant score bullish ({q_score:+.2f})")
            supporting.extend(quant_analysis.signals)
            if n_score > 0:
                supporting.append(f"News sentiment positive ({n_score:+.2f}, {news_analysis.news_count} items)")
            for ev in news_analysis.events:
                if ev.get("sentiment_score", 0) > 0.2:
                    supporting.append(f"News: {ev.get('title')}")

            # Contradicting (if any)
            for ev in news_analysis.events:
                if ev.get("sentiment_score", 0) < -0.2:
                    contradicting.append(f"Negative news outlier: {ev.get('title')}")

        elif is_both_bearish:
            proposed_action = "SHORT"
            base_conf = 0.50 + 0.30 * abs(q_score) + 0.20 * abs(min(0.0, n_score))
            confidence = min(0.95, max(0.60, base_conf))

            supporting.append(f"Quant score bearish ({q_score:+.2f})")
            supporting.extend(quant_analysis.signals)
            if n_score < 0:
                supporting.append(f"News sentiment negative ({n_score:+.2f}, {news_analysis.news_count} items)")
            for ev in news_analysis.events:
                if ev.get("sentiment_score", 0) < -0.2:
                    supporting.append(f"News: {ev.get('title')}")

            for ev in news_analysis.events:
                if ev.get("sentiment_score", 0) > 0.2:
                    contradicting.append(f"Positive news outlier: {ev.get('title')}")

        elif is_divergent:
            # Significant divergence forces IGNORE and lowers confidence
            proposed_action = "IGNORE"
            confidence = round(max(0.10, min(0.40, 1.0 - abs(q_score - n_score) / 2.0)), 4)

            contradicting.append(
                f"Severe signal divergence: Quant ({q_score:+.2f}) contradicts News ({n_score:+.2f})"
            )
            if q_score > 0:
                supporting.append("Quant shows bullish indicators")
                contradicting.append("News sentiment is bearish")
            else:
                supporting.append("News sentiment is bullish")
                contradicting.append("Quant shows bearish technical breakdown")

        else:
            # Neutral / Low conviction
            proposed_action = "IGNORE"
            confidence = round(max(0.0, min(0.35, (abs(q_score) + abs(n_score)) / 2.0)), 4)
            contradicting.append("Insufficient momentum or signal conviction to initiate position")
            if quant_analysis.signals:
                supporting.extend(quant_analysis.signals[:2])

        return DecisionReport(
            decision_id=decision_id,
            asset_id=asset_id,
            sim_time=sim_time,
            proposed_action=proposed_action,
            confidence=round(confidence, 4),
            quant_analysis=quant_analysis,
            news_analysis=news_analysis,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            current_price=current_price,
            metadata={
                "q_score": q_score,
                "n_score": n_score,
                "is_divergent": is_divergent,
            },
        )
