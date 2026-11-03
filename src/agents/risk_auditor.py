from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from src.agents.base import BaseAgent
from src.agents.synthesizer import DecisionReport


@dataclass
class AuditVerdict:
    decision_id: str
    status: str  # "APPROVED" or "VETOED"
    veto_reasons: List[str] = field(default_factory=list)
    audit_details: Dict[str, Any] = field(default_factory=dict)
    sim_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskAuditor(BaseAgent):
    """
    Agent 5: Risk Auditor.
    Independent auditor with absolute VETO power.
    Evaluates DecisionReport against strict risk invariants:
      1. Look-ahead bias audit (evidence t_avail <= current_sim_time)
      2. Liquidity & volume threshold
      3. Volatility guard (vetoes flash crashes / extreme black swan regimes)
      4. Cost & slippage feasibility check
    """

    def __init__(
        self,
        agent_id: str = "agent_risk_auditor",
        name: str = "Risk Auditor",
        min_volume: float = 1.0,
        min_quote_volume: float = 5000.0,
        max_realized_volatility: float = 0.08,
        max_cost_slippage_ratio: float = 0.03,
    ):
        super().__init__(agent_id=agent_id, name=name)
        self.min_volume = min_volume
        self.min_quote_volume = min_quote_volume
        self.max_realized_volatility = max_realized_volatility
        self.max_cost_slippage_ratio = max_cost_slippage_ratio

    def evaluate(self, context: Dict[str, Any]) -> AuditVerdict:
        sim_time = context.get("sim_time") or context.get("current_sim_time")
        if sim_time is None:
            sim_time = datetime.now(timezone.utc)
        elif sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)

        report: Optional[DecisionReport] = context.get("report") or context.get("decision_report")
        if report is None:
            raise ValueError("RiskAuditor requires DecisionReport in context")

        veto_reasons: List[str] = []
        details: Dict[str, Any] = {}

        # Invariant 1: Strict Look-Ahead Bias Audit
        lookahead_violations = self._audit_temporal_integrity(context, report, sim_time)
        if lookahead_violations:
            veto_reasons.extend(lookahead_violations)
        details["temporal_audit"] = "PASSED" if not lookahead_violations else "FAILED"

        # Invariant 2: Liquidity & Volume Threshold
        volume = float(report.quant_analysis.metrics.get("volume", 0.0))
        quote_volume = float(report.quant_analysis.metrics.get("quote_volume", 0.0))
        details["volume"] = volume
        details["quote_volume"] = quote_volume

        if report.proposed_action in ("BUY", "SHORT"):
            if volume < self.min_volume and quote_volume < self.min_quote_volume:
                veto_reasons.append(
                    f"INSUFFICIENT_LIQUIDITY: volume ({volume:.2f}) < min ({self.min_volume:.2f}) "
                    f"and quote_volume ({quote_volume:.2f}) < min ({self.min_quote_volume:.2f})"
                )

        # Invariant 3: Volatility Guard (flash-crash / black swan regime)
        realized_vol = float(report.quant_analysis.metrics.get("realized_volatility", 0.0))
        details["realized_volatility"] = realized_vol

        if realized_vol > self.max_realized_volatility:
            veto_reasons.append(
                f"VOLATILITY_GUARD_EXCEEDED: realized volatility ({realized_vol:.4f}) "
                f"> safe threshold ({self.max_realized_volatility:.4f})"
            )

        # Invariant 4: Cost & Slippage Feasibility Check
        atr = float(report.quant_analysis.metrics.get("atr", 0.0))
        price = float(report.quant_analysis.metrics.get("current_price", 0.0))
        cost_ratio = (atr / price) if price > 0 else 0.0
        details["cost_slippage_ratio"] = round(cost_ratio, 6)

        if report.proposed_action in ("BUY", "SHORT"):
            if cost_ratio > self.max_cost_slippage_ratio:
                veto_reasons.append(
                    f"COST_SLIPPAGE_INFEASIBLE: cost/slippage proxy ratio ({cost_ratio:.4f}) "
                    f"> max allowable ({self.max_cost_slippage_ratio:.4f})"
                )

        status = "VETOED" if veto_reasons else "APPROVED"

        return AuditVerdict(
            decision_id=report.decision_id,
            status=status,
            veto_reasons=veto_reasons,
            audit_details=details,
            sim_time=sim_time,
        )

    def _audit_temporal_integrity(
        self, context: Dict[str, Any], report: DecisionReport, sim_time: datetime
    ) -> List[str]:
        violations = []

        if report.sim_time > sim_time:
            violations.append(
                f"LOOKAHEAD_BIAS_DETECTED: report sim_time ({report.sim_time}) > current_sim_time ({sim_time})"
            )

        # Check news events referenced
        for ev in report.news_analysis.events:
            t_avail_str = ev.get("t_avail")
            if t_avail_str:
                try:
                    t_avail = datetime.fromisoformat(t_avail_str)
                    if t_avail.tzinfo is None:
                        t_avail = t_avail.replace(tzinfo=timezone.utc)
                    if t_avail > sim_time:
                        violations.append(
                            f"LOOKAHEAD_BIAS_DETECTED: news '{ev.get('title')}' t_avail ({t_avail}) > sim_time ({sim_time})"
                        )
                except Exception:
                    pass

        # Check raw news in context if present
        raw_news = context.get("news") or []
        for n in raw_news:
            avail = n.get("t_avail")
            if avail is not None:
                if isinstance(avail, pd.Timestamp):
                    avail = avail.to_pydatetime()
                if avail.tzinfo is None:
                    avail = avail.replace(tzinfo=timezone.utc)
                if avail > sim_time:
                    violations.append(
                        f"LOOKAHEAD_BIAS_DETECTED: context news t_avail ({avail}) > sim_time ({sim_time})"
                    )

        # Check current bar in context if present
        current_bar = context.get("bar") or {}
        b_avail = current_bar.get("t_avail")
        if b_avail is not None:
            if isinstance(b_avail, pd.Timestamp):
                b_avail = b_avail.to_pydatetime()
            if b_avail.tzinfo is None:
                b_avail = b_avail.replace(tzinfo=timezone.utc)
            if b_avail > sim_time:
                violations.append(
                    f"LOOKAHEAD_BIAS_DETECTED: market bar t_avail ({b_avail}) > sim_time ({sim_time})"
                )

        return violations
