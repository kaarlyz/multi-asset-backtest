from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict, List, Optional

from src.bot.telegram_service import TelegramBotService
from src.chronos.replay import HistoricalReplayEngine
from src.engine.portfolio import SimPortfolio
from src.orchestrator import MultiAgentOrchestrator


class SimulationRunner:
    """
    Simulation Runner & Orchestration Loop.
    Coordinates ChronosReplayEngine, MultiAgentOrchestrator, SimPortfolio,
    and TelegramBotService across AUTONOMOUS and INTERACTIVE modes.
    """

    def __init__(
        self,
        replay_engine: HistoricalReplayEngine,
        orchestrator: MultiAgentOrchestrator,
        portfolio: SimPortfolio,
        telegram_service: Optional[TelegramBotService] = None,
        mode: str = "AUTONOMOUS",
        step_delay: float = 0.0,
        decision_timeout: float = 300.0,
        poll_interval: float = 0.1,
    ):
        self.replay_engine = replay_engine
        self.orchestrator = orchestrator
        self.portfolio = portfolio
        self.telegram_service = telegram_service
        self.mode = mode.upper()
        self.step_delay = float(step_delay)
        self.decision_timeout = float(decision_timeout)
        self.poll_interval = float(poll_interval)

        self._waiting_for_decision = False
        self._stop_listener = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None

        # Synchronize telegram service state
        if self.telegram_service is not None:
            self.telegram_service.portfolio = self.portfolio  # type: ignore
            self.telegram_service.replay_engine = self.replay_engine
            self.telegram_service.db = self.replay_engine.db

        # In both modes, SimulationRunner orchestrates execution and prompts
        self.orchestrator.telegram_service = None

    def start_command_listener(self) -> None:
        """Start background daemon thread for Telegram command polling."""
        if not self.telegram_service or not self.telegram_service.token:
            return
        self._stop_listener.clear()
        self._listener_thread = threading.Thread(
            target=self._command_listener_worker,
            name="TelegramCommandListener",
            daemon=True,
        )
        self._listener_thread.start()

    def stop_command_listener(self) -> None:
        """Stop background command listener thread."""
        self._stop_listener.set()
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.5)

    def _command_listener_worker(self) -> None:
        offset = None
        while not self._stop_listener.is_set():
            try:
                if self.telegram_service:
                    updates = self.telegram_service.get_updates(offset=offset, timeout=1)
                    for upd in updates:
                        offset = upd.get("update_id", 0) + 1
                        self.telegram_service.handle_update(upd)
            except Exception:
                pass
            time.sleep(0.2)

    def step(self) -> Dict[str, Any]:
        """
        Execute one complete orchestration step:
        1. Wait if replay is paused by user command.
        2. Advance replay & execute agent pipeline via orchestrator.
        3. Update portfolio mark-to-market and SL/TP with current bar.
        4. In AUTONOMOUS mode: execute approved signal automatically.
        5. In INTERACTIVE mode: prompt Telegram, pause clock, wait for callback, execute chosen action, resume.
        """
        # 1. Handle user manual pause
        while self.replay_engine.is_paused and not self._waiting_for_decision and not self._stop_listener.is_set():
            time.sleep(0.2)

        # 2. Advance replay and run multi-agent pipeline
        step_out = self.orchestrator.step(advance=True)
        state = step_out.get("state", {})
        bar = state.get("bar") or self.replay_engine.get_current_bar()

        # 3. Update portfolio mark-to-market and check SL/TP triggers
        closed_in_bar = []
        if bar:
            closed_in_bar = self.portfolio.update_bar(bar)

        audit_verdict = step_out.get("audit_verdict")
        decision_report = step_out.get("decision_report")
        quant_analysis = step_out.get("quant_analysis")
        news_analysis = step_out.get("news_analysis")

        proposed_action = decision_report.proposed_action if decision_report else "IGNORE"
        decision_id = decision_report.decision_id if decision_report else None
        is_approved = bool(audit_verdict and audit_verdict.status == "APPROVED")

        executed_action = None

        # 4. Mode AUTONOMOUS
        if self.mode == "AUTONOMOUS":
            if is_approved and proposed_action in ("BUY", "SHORT") and bar and decision_id:
                self.portfolio.execute_decision(
                    action=proposed_action,
                    bar=bar,
                    decision_id=decision_id,
                    actor_id="AUTONOMOUS_RUNNER",
                )
                executed_action = proposed_action
                if self.replay_engine.db:
                    self.replay_engine.db.update_decision_status(decision_id, "EXECUTED")

        # 5. Mode INTERACTIVE
        elif self.mode == "INTERACTIVE":
            if is_approved and proposed_action in ("BUY", "SHORT") and bar and decision_id:
                price = float(bar.get("close", 0.0))
                quant_score = quant_analysis.score if quant_analysis else 0.0
                tvl_mom = 0.0
                if quant_analysis and quant_analysis.metrics:
                    tvl_mom = float(quant_analysis.metrics.get("tvl_momentum_30d", quant_analysis.metrics.get("tvl_momentum", 0.0)))
                news_score = news_analysis.score if news_analysis else 0.0
                news_cnt = news_analysis.news_count if news_analysis else 0
                conf = decision_report.confidence if decision_report else 0.0

                rich_card = (
                    f"Price: ${price:,.2f} | Quant Score: {quant_score:+.2f}\n"
                    f"TVL Momentum: {tvl_mom:+.2f}% | Sentiment: {news_score:+.2f} ({news_cnt} news items)\n"
                    f"Proposed Action: {proposed_action} (Confidence: {conf:.1%})"
                )

                # Send prompt via TelegramBotService
                if self.telegram_service is not None:
                    self.telegram_service.send_signal_prompt(
                        asset=self.replay_engine.active_asset,
                        current_price=price,
                        report_summary=rich_card,
                        decision_id=decision_id,
                    )

                # Pause replay clock and wait for human decision
                self.replay_engine.pause()
                self._waiting_for_decision = True
                start_wait = time.time()
                chosen_action = None
                chosen_actor = "TELEGRAM_USER"

                try:
                    while (time.time() - start_wait) < self.decision_timeout:
                        if self.replay_engine.db:
                            rec = self.replay_engine.db.get_decision(decision_id)
                            if rec and rec.get("status") == "RECORDED":
                                chosen_action = rec.get("action")
                                chosen_actor = rec.get("actor_id", "TELEGRAM_USER")
                                break
                        time.sleep(self.poll_interval)
                finally:
                    self._waiting_for_decision = False
                    self.replay_engine.resume()

                # Process decision outcome
                if chosen_action is None:
                    # Timeout reached
                    chosen_action = "IGNORE"
                    if self.replay_engine.db:
                        self.replay_engine.db.update_decision_status(decision_id, "TIMEOUT")
                elif chosen_action in ("BUY", "SHORT"):
                    self.portfolio.execute_decision(
                        action=chosen_action,
                        bar=bar,
                        decision_id=decision_id,
                        actor_id=chosen_actor,
                    )
                    if self.replay_engine.db:
                        self.replay_engine.db.update_decision_status(decision_id, "EXECUTED")

                executed_action = chosen_action

        return {
            "step_output": step_out,
            "bar": bar,
            "executed_action": executed_action,
            "closed_trades": closed_in_bar,
            "portfolio": self.portfolio.get_summary(),
        }

    def run(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run orchestration loop to completion or until max_steps.
        Returns final portfolio summary.
        """
        self.start_command_listener()
        steps = 0
        try:
            while not self.replay_engine.is_finished():
                if max_steps is not None and steps >= max_steps:
                    break
                self.step()
                steps += 1
                if self.step_delay > 0:
                    time.sleep(self.step_delay)
        finally:
            self.stop_command_listener()

        return self.portfolio.get_summary()
