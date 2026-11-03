from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import uuid

from src.agents.news_miner import NewsMiner, NewsSentimentAnalysis
from src.agents.quant_specialist import QuantSpecialist, QuantitativeAnalysis
from src.agents.risk_auditor import AuditVerdict, RiskAuditor
from src.agents.synthesizer import DecisionReport, Synthesizer
from src.bot.telegram_service import TelegramBotService
from src.chronos.db import ChronosDB
from src.chronos.replay import HistoricalReplayEngine


class MultiAgentOrchestrator:
    """
    Multi-Agent Intelligence Engine Orchestrator.
    Executes sequential/parallel pipeline:
      Chronos Replay Step -> Quant Specialist & News Miner -> Synthesizer -> Risk Auditor.
    Enforces risk audit verdict:
      - APPROVED: creates decision record and dispatches signal (via Telegram or autonomous ledger).
      - VETOED: logs veto reason immutably in decision ledger without order creation.
    """

    def __init__(
        self,
        replay_engine: HistoricalReplayEngine,
        quant_agent: Optional[QuantSpecialist] = None,
        news_miner: Optional[NewsMiner] = None,
        synthesizer: Optional[Synthesizer] = None,
        risk_auditor: Optional[RiskAuditor] = None,
        telegram_service: Optional[TelegramBotService] = None,
        db: Optional[ChronosDB] = None,
        actor_id: str = "MULTI_AGENT_ENGINE",
    ):
        self.replay_engine = replay_engine
        self.db = db or replay_engine.db
        self.telegram_service = telegram_service
        self.actor_id = actor_id

        self.quant_agent = quant_agent or QuantSpecialist()
        self.news_miner = news_miner or NewsMiner()
        self.synthesizer = synthesizer or Synthesizer()
        self.risk_auditor = risk_auditor or RiskAuditor()

    def step(self, advance: bool = True) -> Dict[str, Any]:
        """
        Execute one simulated time-step across the multi-agent pipeline.
        """
        # 1. Step replay engine to current sim state
        state = self.replay_engine.step(advance=advance)
        sim_time = state["sim_time"]
        if sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)

        asset_id = self.replay_engine.active_asset
        sim_run_id = self.replay_engine.sim_run_id
        current_bar = state.get("bar") or {}

        # Build execution context strictly at current_sim_time
        context: Dict[str, Any] = {
            "sim_time": sim_time,
            "current_sim_time": sim_time,
            "symbol": asset_id,
            "asset_id": asset_id,
            "bar": current_bar,
            "fundamentals": state.get("fundamentals", []),
            "replay_engine": self.replay_engine,
            "db": self.db,
        }

        # 2. Agent 2 & Agent 3: Quant Specialist and News Miner
        quant_result: QuantitativeAnalysis = self.quant_agent.evaluate(context)
        news_result: NewsSentimentAnalysis = self.news_miner.evaluate(context)

        # 3. Agent 4: Synthesizer
        context["quant_analysis"] = quant_result
        context["news_analysis"] = news_result
        decision_report: DecisionReport = self.synthesizer.evaluate(context)

        # 4. Agent 5: Risk Auditor (with absolute VETO power)
        context["report"] = decision_report
        context["decision_report"] = decision_report
        audit_verdict: AuditVerdict = self.risk_auditor.evaluate(context)

        # 5. Execute Outcome & Record to Decision Ledger
        dispatched = False
        decision_id = decision_report.decision_id
        entry_price = float(current_bar.get("close", 0.0)) if current_bar else None

        evidence_payload = {
            "proposed_action": decision_report.proposed_action,
            "confidence": decision_report.confidence,
            "supporting_evidence": decision_report.supporting_evidence,
            "contradicting_evidence": decision_report.contradicting_evidence,
            "quant_score": quant_result.score,
            "quant_metrics": quant_result.metrics,
            "news_score": news_result.score,
            "news_count": news_result.news_count,
            "veto_reasons": audit_verdict.veto_reasons,
            "audit_details": audit_verdict.audit_details,
        }

        if audit_verdict.status == "APPROVED":
            if decision_report.proposed_action in ("BUY", "SHORT"):
                # Dispatch signal via Telegram Bot if available
                if self.telegram_service is not None and entry_price is not None:
                    summary = (
                        f"{decision_report.proposed_action} signal (conf: {decision_report.confidence:.2f}). "
                        f"Quant: {quant_result.score:+.2f}, News: {news_result.score:+.2f}"
                    )
                    self.telegram_service.send_signal_prompt(
                        asset=asset_id,
                        current_price=entry_price,
                        report_summary=summary,
                        decision_id=decision_id,
                    )
                    dispatched = True
                    interface = "TELEGRAM_PROMPT"
                    status = "SIGNAL_DISPATCHED"
                else:
                    interface = "AUTONOMOUS_LEDGER"
                    status = "APPROVED"

                if self.db is not None:
                    self.db.record_decision(
                        decision_id=decision_id,
                        sim_run_id=sim_run_id,
                        sim_time=sim_time,
                        asset_id=asset_id,
                        interface=interface,
                        actor_id=self.actor_id,
                        action=decision_report.proposed_action,
                        agent_confidence=decision_report.confidence,
                        evidence_snapshot=evidence_payload,
                        entry_price=entry_price,
                        status=status,
                    )
            else:
                # Proposed action is IGNORE
                if self.db is not None:
                    self.db.record_decision(
                        decision_id=decision_id,
                        sim_run_id=sim_run_id,
                        sim_time=sim_time,
                        asset_id=asset_id,
                        interface="AUTONOMOUS_LEDGER",
                        actor_id=self.actor_id,
                        action="IGNORE",
                        agent_confidence=decision_report.confidence,
                        evidence_snapshot=evidence_payload,
                        entry_price=entry_price,
                        status="APPROVED",
                    )
        else:
            # Risk Auditor VETOED: log immutably without order creation
            if self.db is not None:
                self.db.record_decision(
                    decision_id=decision_id,
                    sim_run_id=sim_run_id,
                    sim_time=sim_time,
                    asset_id=asset_id,
                    interface="RISK_AUDITOR",
                    actor_id="agent_risk_auditor",
                    action=decision_report.proposed_action,
                    agent_confidence=decision_report.confidence,
                    evidence_snapshot=evidence_payload,
                    entry_price=entry_price,
                    status="VETOED",
                )

        return {
            "sim_time": sim_time,
            "asset_id": asset_id,
            "state": state,
            "quant_analysis": quant_result,
            "news_analysis": news_result,
            "decision_report": decision_report,
            "audit_verdict": audit_verdict,
            "dispatched": dispatched,
            "status": audit_verdict.status,
        }

    def run(self, max_steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run orchestration loop for max_steps or until replay is finished."""
        results = []
        step_count = 0
        while not self.replay_engine.is_finished():
            if max_steps is not None and step_count >= max_steps:
                break
            res = self.step(advance=True)
            results.append(res)
            step_count += 1
        return results
