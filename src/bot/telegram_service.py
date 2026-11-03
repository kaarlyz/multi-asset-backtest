from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set, Union
import httpx
from dotenv import load_dotenv

from src.chronos.db import ChronosDB
from src.chronos.replay import HistoricalReplayEngine


DEFAULT_BOT_USERNAME = "ekaresearchmulti_bot"


class TelegramBotService:
    """
    Central Telegram Bot Service for human-in-the-loop decisions and replay control.
    Enforces user allowlisting, idempotency on decision callbacks, and replay engine state control.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        allowed_user_ids: Optional[Union[List[Union[int, str]], Set[Union[int, str]], str]] = None,
        replay_engine: Optional[HistoricalReplayEngine] = None,
        db: Optional[ChronosDB] = None,
        portfolio: Optional[Union[Dict[str, Any], Any]] = None,
        http_client: Optional[httpx.Client] = None,
        env_path: Optional[Union[str, Path]] = None,
    ):
        # Load environment variables (.env)
        if env_path:
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()

        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.bot_username = os.getenv("TELEGRAM_BOT_USERNAME", DEFAULT_BOT_USERNAME)

        # Parse allowed user IDs
        self.allowed_user_ids: Set[int] = set()
        raw_allowed = allowed_user_ids if allowed_user_ids is not None else os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")
        self._parse_allowed_user_ids(raw_allowed)

        self.replay_engine = replay_engine
        self.db = db
        self.portfolio: Any = portfolio or {
            "cash": 100000.0,
            "equity": 100000.0,
            "open_positions": [],
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
        }
        self.http_client = http_client or httpx.Client(timeout=10.0)
        self.base_url = f"https://api.telegram.org/bot{self.token}"

        self.sim_runner: Optional[Any] = None
        self.sim_thread: Optional[threading.Thread] = None

    def is_sim_running(self) -> bool:
        return bool(self.sim_thread and self.sim_thread.is_alive())

    def _parse_date(self, date_str: str, is_end: bool = False) -> datetime:
        try:
            dt = datetime.fromisoformat(date_str)
        except Exception:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if is_end and dt.hour == 0 and dt.minute == 0 and dt.second == 0:
            dt = dt.replace(hour=23, minute=59, second=59)
        return dt

    def start_simulation(
        self,
        asset: str = "BTCUSDT",
        start_date: str = "2024-01-01",
        end_date: str = "2024-03-31",
        interval: str = "1d",
        chat_id: Optional[Union[int, str]] = None,
        step_delay: float = 2.0,
    ) -> Any:
        from src.engine.portfolio import SimPortfolio
        from src.engine.runner import SimulationRunner
        from src.orchestrator import MultiAgentOrchestrator

        project_root = Path(__file__).resolve().parent.parent.parent

        if self.db is None:
            default_db = project_root / "data" / "chronos.duckdb"
            self.db = ChronosDB(str(default_db))

        base_asset = asset.replace("USDT", "")
        try:
            self.db.insert_asset(
                asset_id=f"CRYPTO_{base_asset}",
                asset_class="CRYPTO",
                symbol=asset,
                base_asset=base_asset,
                quote_asset="USDT",
                is_active=True,
            )
        except Exception:
            pass

        market_dir = project_root / "data" / "parquet" / "market" / "CRYPTO" / asset
        market_path = market_dir / f"interval={interval}"
        if not market_path.exists():
            market_path = market_dir
        market_data = market_path if market_path.exists() else None

        start_dt = self._parse_date(start_date, is_end=False)
        end_dt = self._parse_date(end_date, is_end=True)

        replay_engine = HistoricalReplayEngine(
            db=self.db,
            symbol=asset,
            start_time=start_dt,
            end_time=end_dt,
            step_size=interval,
            interval=interval,
            market_data=market_data,
        )

        orchestrator = MultiAgentOrchestrator(
            replay_engine=replay_engine,
            db=self.db,
        )

        portfolio = SimPortfolio(initial_cash=10000.0)

        runner = SimulationRunner(
            replay_engine=replay_engine,
            orchestrator=orchestrator,
            portfolio=portfolio,
            telegram_service=self,
            mode="INTERACTIVE",
            step_delay=step_delay,
            auto_listener=False,
        )

        self.sim_runner = runner
        self.replay_engine = replay_engine
        self.portfolio = portfolio

        def _worker():
            try:
                runner.run()
                if chat_id and not runner.is_stopped:
                    self.send_message(
                        chat_id=chat_id,
                        text=f"🏁 Simulasi {asset} selesai!\n\n{self._build_portfolio_report()}",
                    )
            except Exception as e:
                if chat_id and not runner.is_stopped:
                    self.send_message(
                        chat_id=chat_id,
                        text=f"⚠️ Error dalam simulasi {asset}: {e}",
                    )

        self.sim_thread = threading.Thread(target=_worker, name=f"SimRunner-{asset}", daemon=True)
        self.sim_thread.start()
        return runner

    def stop_simulation(self) -> Dict[str, Any]:
        """Halt active simulation runner and worker thread."""
        if self.sim_runner is not None and hasattr(self.sim_runner, "stop"):
            self.sim_runner.stop()
        if self.sim_thread is not None and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=3.0)

        if hasattr(self.portfolio, "get_summary"):
            return self.portfolio.get_summary()
        return self.portfolio if isinstance(self.portfolio, dict) else {}

    def _parse_allowed_user_ids(
        self, allowed: Union[List[Union[int, str]], Set[Union[int, str]], str]
    ) -> None:
        if isinstance(allowed, (list, set)):
            for item in allowed:
                try:
                    self.allowed_user_ids.add(int(item))
                except (ValueError, TypeError):
                    pass
        elif isinstance(allowed, str) and allowed.strip():
            for part in allowed.split(","):
                part = part.strip()
                if part:
                    try:
                        self.allowed_user_ids.add(int(part))
                    except ValueError:
                        pass

    def is_user_authorized(self, user_id: Optional[Union[int, str]]) -> bool:
        """Security Gate: Verify if user_id is in allowed list."""
        if user_id is None:
            return False
        try:
            return int(user_id) in self.allowed_user_ids
        except (ValueError, TypeError):
            return False

    def send_message(
        self,
        chat_id: Union[int, str],
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send message via Telegram Bot API."""
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        if not self.token:
            # Running in mock/test mode without live token
            return {"ok": True, "result": payload}

        try:
            resp = self.http_client.post(f"{self.base_url}/sendMessage", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"ok": False, "error": str(e), "payload": payload}

    def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False,
    ) -> Dict[str, Any]:
        """Acknowledge or answer an inline callback query."""
        payload: Dict[str, Any] = {
            "callback_query_id": callback_query_id,
            "show_alert": show_alert,
        }
        if text:
            payload["text"] = text

        if not self.token:
            return {"ok": True, "result": payload}

        try:
            resp = self.http_client.post(
                f"{self.base_url}/answerCallbackQuery", json=payload
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"ok": False, "error": str(e), "payload": payload}

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming chat messages and dispatch commands."""
        from_user = message.get("from", {})
        user_id = from_user.get("id")
        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()

        # Security Gate
        if not self.is_user_authorized(user_id):
            return {
                "status": "DROPPED_UNAUTHORIZED",
                "user_id": user_id,
                "message": "User ID not in allowlist",
            }

        cmd = text.split()[0].lower() if text else ""
        # Strip bot username if present e.g. /status@ekaresearchmulti_bot
        if "@" in cmd:
            cmd = cmd.split("@")[0]

        reply_text = ""
        if cmd in ("/start", "/help"):
            reply_text = (
                f"🤖 Chronos Central Bot (@{self.bot_username})\n\n"
                "Available commands:\n"
                "/start_sim [ASSET] [START] [END] - Start interactive simulation\n"
                "/stop_sim - Stop active simulation and show portfolio\n"
                "/status - Current simulation and replay status\n"
                "/portfolio - Simulation cash, equity, open positions, PnL\n"
                "/report - Latest fundamental snapshot & quant metrics\n"
                "/pause - Pause replay loop\n"
                "/resume - Resume replay loop"
            )
        elif cmd == "/start_sim":
            if self.is_sim_running():
                reply_text = "⚠️ Simulasi sedang berjalan. Hentikan dulu dengan /stop_sim sebelum memulai yang baru."
            else:
                parts = text.split()
                asset = parts[1].upper() if len(parts) > 1 else "BTCUSDT"
                start_date = parts[2] if len(parts) > 2 else "2024-01-01"
                end_date = parts[3] if len(parts) > 3 else "2024-03-31"

                try:
                    self.start_simulation(
                        asset=asset,
                        start_date=start_date,
                        end_date=end_date,
                        chat_id=chat_id,
                        step_delay=2.0,
                    )
                    reply_text = f"🚀 Simulasi {asset} dimulai ({start_date} s/d {end_date}). Mode: INTERACTIVE."
                except Exception as e:
                    reply_text = f"❌ Gagal memulai simulasi: {e}"
        elif cmd == "/stop_sim":
            if not self.is_sim_running() and (self.sim_runner is None or getattr(self.sim_runner, "is_stopped", True)):
                reply_text = "⚠️ Tidak ada simulasi yang sedang berjalan."
            else:
                self.stop_simulation()
                reply_text = f"🛑 Simulasi dihentikan.\n\n{self._build_portfolio_report()}"
        elif cmd == "/status":
            reply_text = self._build_status_report()
        elif cmd == "/portfolio":
            reply_text = self._build_portfolio_report()
        elif cmd == "/report":
            reply_text = self._build_fundamental_report()
        elif cmd == "/pause":
            if self.replay_engine:
                self.replay_engine.pause()
                sim_time = self.replay_engine.current_sim_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                reply_text = f"⏸️ Replay PAUSED at sim_time: {sim_time}"
            else:
                reply_text = "⚠️ No active replay engine attached."
        elif cmd == "/resume":
            if self.replay_engine:
                self.replay_engine.resume()
                sim_time = self.replay_engine.current_sim_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                reply_text = f"▶️ Replay RESUMED at sim_time: {sim_time}"
            else:
                reply_text = "⚠️ No active replay engine attached."
        else:
            reply_text = f"Unknown command '{cmd}'. Use /help for available commands."

        if chat_id:
            self.send_message(chat_id=chat_id, text=reply_text)

        return {
            "status": "HANDLED",
            "command": cmd,
            "response": reply_text,
            "user_id": user_id,
        }

    def _build_status_report(self) -> str:
        if not self.replay_engine:
            return "⚠️ No active replay engine attached."

        eng = self.replay_engine
        if eng.is_paused:
            state_str = "PAUSED ⏸️"
        elif self.is_sim_running():
            state_str = "RUNNING ▶️"
        elif eng.is_finished():
            state_str = "FINISHED 🏁"
        elif self.sim_runner and getattr(self.sim_runner, "is_stopped", False):
            state_str = "STOPPED 🛑"
        else:
            state_str = "RUNNING ▶️"

        sim_time_str = eng.current_sim_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        bar = eng.get_current_bar()
        current_price = f"${bar['close']:,.2f}" if bar and "close" in bar else "N/A"

        return (
            f"📊 SIMULATION STATUS:\n"
            f"State: {state_str}\n"
            f"Sim Run ID: {eng.sim_run_id}\n"
            f"Sim Time: {sim_time_str}\n"
            f"Active Asset: {eng.active_asset}\n"
            f"Current Price: {current_price}\n"
            f"Step Size: {eng.step_size}"
        )

    def _build_portfolio_report(self) -> str:
        port = self.portfolio
        if hasattr(port, "cash"):
            cash = getattr(port, "cash", 0.0)
            equity = getattr(port, "equity", 0.0)
            unrealized_pnl = getattr(port, "unrealized_pnl", 0.0)
            realized_pnl = getattr(port, "realized_pnl", 0.0)
            positions = getattr(port, "open_positions", {})
        elif isinstance(port, dict):
            cash = port.get("cash", 0.0)
            equity = port.get("equity", 0.0)
            unrealized_pnl = port.get("unrealized_pnl", 0.0)
            realized_pnl = port.get("realized_pnl", 0.0)
            positions = port.get("open_positions", [])
        else:
            cash, equity, unrealized_pnl, realized_pnl, positions = 0.0, 0.0, 0.0, 0.0, []

        pos_str = "None"
        if positions:
            pos_items = list(positions.values()) if isinstance(positions, dict) else positions
            pos_lines = [
                f" - {p.get('symbol', 'UNKNOWN')}: {p.get('size', 0)} @ ${p.get('entry_price', 0):,.2f}"
                for p in pos_items
            ]
            pos_str = "\n".join(pos_lines)

        return (
            f"💼 PORTFOLIO SUMMARY:\n"
            f"Cash: ${cash:,.2f}\n"
            f"Equity: ${equity:,.2f}\n"
            f"Unrealized PnL: ${unrealized_pnl:,.2f}\n"
            f"Realized PnL: ${realized_pnl:,.2f}\n"
            f"Open Positions:\n{pos_str}"
        )

    def _build_fundamental_report(self) -> str:
        if not self.replay_engine:
            return "⚠️ No active replay engine attached."

        eng = self.replay_engine
        sim_time_str = eng.current_sim_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        fundamentals = eng.get_current_fundamentals()

        lines = [f"📈 PIT FUNDAMENTAL & QUANT REPORT ({eng.active_asset}):", f"Sim Time: {sim_time_str}"]

        bar = eng.get_current_bar()
        if bar:
            lines.append(f"Market Close: ${bar.get('close', 0):,.2f} (Vol: {bar.get('volume', 0):,.2f})")

        if fundamentals:
            lines.append("\nPoint-in-Time Metrics:")
            for item in fundamentals:
                m_id = item.get("metric_id", "METRIC")
                val = item.get("metric_value", 0.0)
                t_pub = item.get("t_pub")
                t_pub_str = t_pub.strftime("%Y-%m-%d %H:%M") if t_pub is not None and hasattr(t_pub, "strftime") else str(t_pub)
                lines.append(f" • {m_id}: {val:,.2f} (published: {t_pub_str}, source: {item.get('source')})")
        else:
            lines.append("\nNo fundamental data available at this timestamp.")

        return "\n".join(lines)

    def send_signal_prompt(
        self,
        asset: str,
        current_price: float,
        report_summary: str,
        chat_id: Optional[Union[int, str]] = None,
        decision_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send signal prompt with inline keyboard [BUY] [SHORT] [IGNORE].
        Callback data format: DECIDE:{decision_id}:{ACTION}
        """
        if decision_id is None:
            ts = int(datetime.now(timezone.utc).timestamp() * 1000)
            decision_id = f"DEC_{asset}_{ts}"

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "BUY 🟢", "callback_data": f"DECIDE:{decision_id}:BUY"},
                    {"text": "SHORT 🔴", "callback_data": f"DECIDE:{decision_id}:SHORT"},
                    {"text": "IGNORE ⚪", "callback_data": f"DECIDE:{decision_id}:IGNORE"},
                ]
            ]
        }

        text = (
            f"🚨 *SIGNAL PROMPT: {asset}*\n"
            f"Price: ${current_price:,.2f}\n"
            f"Summary: {report_summary}\n"
            f"Decision ID: `{decision_id}`\n\n"
            f"Select execution action:"
        )

        resp = None
        target_chat = chat_id or (list(self.allowed_user_ids)[0] if self.allowed_user_ids else None)
        if target_chat is not None:
            resp = self.send_message(chat_id=target_chat, text=text, reply_markup=keyboard)

        return {
            "decision_id": decision_id,
            "asset": asset,
            "price": current_price,
            "text": text,
            "reply_markup": keyboard,
            "response": resp,
        }

    def send_exit_notification(
        self,
        trade: Dict[str, Any],
        chat_id: Optional[Union[int, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send notification when a position is closed (SL, TP, or signal exit).
        """
        target_chat = chat_id or (list(self.allowed_user_ids)[0] if self.allowed_user_ids else None)
        if target_chat is None:
            return None

        sym = trade.get("symbol", "ASSET")
        side = trade.get("side", "")
        reason = trade.get("exit_reason", "EXIT")
        entry_p = float(trade.get("entry_price", 0.0))
        exit_p = float(trade.get("exit_price", 0.0))
        pnl = float(trade.get("realized_pnl", 0.0))
        ret = float(trade.get("return_pct", 0.0))
        emoji = "🟢" if pnl >= 0 else "🔴"

        text = (
            f"🔔 *POSITION CLOSED ({reason}): {sym}*\n"
            f"Side: {side}\n"
            f"Entry: ${entry_p:,.2f} | Exit: ${exit_p:,.2f}\n"
            f"Realized PnL: {emoji} ${pnl:+,.2f} ({ret:+.2f}%)"
        )
        return self.send_message(chat_id=target_chat, text=text)

    def handle_callback(self, callback_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate user allowlist, enforce idempotency against decision_ledger,
        and record decision into Chronos decision_ledger.
        """
        callback_id = callback_query.get("id", "")
        from_user = callback_query.get("from", {})
        user_id = from_user.get("id")
        data = (callback_query.get("data") or "").strip()

        # 1. Security Gate
        if not self.is_user_authorized(user_id):
            if callback_id:
                self.answer_callback_query(
                    callback_query_id=callback_id,
                    text="Unauthorized user.",
                    show_alert=True,
                )
            return {
                "status": "DROPPED_UNAUTHORIZED",
                "user_id": user_id,
                "message": "User not in allowed list",
            }

        # 2. Parse callback data
        if not data.startswith("DECIDE:"):
            return {
                "status": "INVALID_CALLBACK_DATA",
                "data": data,
            }

        parts = data.split(":")
        if len(parts) != 3:
            return {
                "status": "MALFORMED_CALLBACK_FORMAT",
                "data": data,
            }

        _, decision_id, action = parts
        action = action.upper()
        if action not in ("BUY", "SHORT", "IGNORE"):
            return {
                "status": "INVALID_ACTION",
                "action": action,
            }

        # 3. Idempotency Guard against decision_ledger
        existing_dec = self.db.get_decision(decision_id) if self.db is not None else None
        if existing_dec is not None and existing_dec.get("status") in ("RECORDED", "EXECUTED"):
            if callback_id:
                self.answer_callback_query(
                    callback_query_id=callback_id,
                    text=f"Decision {decision_id} was already executed.",
                    show_alert=True,
                )
            return {
                "status": "REJECTED_ALREADY_DECIDED",
                "decision_id": decision_id,
                "action": action,
                "message": "Duplicate decision rejected by idempotency guard.",
            }

        # 4. Record decision in decision_ledger
        sim_run_id = self.replay_engine.sim_run_id if self.replay_engine else "TELEGRAM_REPLAY"
        sim_time = (
            self.replay_engine.current_sim_time
            if self.replay_engine
            else datetime.now(timezone.utc)
        )
        asset_id = (
            self.replay_engine.active_asset
            if self.replay_engine
            else "CRYPTO_BTC"
        )
        bar = self.replay_engine.get_current_bar() if self.replay_engine else None
        entry_price = float(bar["close"]) if bar and "close" in bar else None

        if self.db is not None:
            if existing_dec is not None:
                self.db.update_decision(
                    decision_id=decision_id,
                    action=action,
                    actor_id=str(user_id),
                    interface="TELEGRAM",
                    status="RECORDED",
                    entry_price=entry_price,
                )
            else:
                try:
                    self.db.record_decision(
                        decision_id=decision_id,
                        sim_run_id=sim_run_id,
                        sim_time=sim_time,
                        asset_id=asset_id,
                        interface="TELEGRAM",
                        actor_id=str(user_id),
                        action=action,
                        entry_price=entry_price,
                        status="RECORDED",
                    )
                except Exception as e:
                    # Primary key collision caught by DB constraint
                    return {
                        "status": "REJECTED_ALREADY_DECIDED",
                        "decision_id": decision_id,
                        "error": str(e),
                    }

        # 5. Acknowledge callback query
        if callback_id:
            self.answer_callback_query(
                callback_query_id=callback_id,
                text=f"✅ Action recorded: {action} ({decision_id})",
            )

        return {
            "status": "RECORDED",
            "decision_id": decision_id,
            "action": action,
            "actor_id": str(user_id),
            "interface": "TELEGRAM",
            "entry_price": entry_price,
        }

    def handle_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch update to message or callback handler."""
        if "message" in update:
            return self.handle_message(update["message"])
        elif "callback_query" in update:
            return self.handle_callback(update["callback_query"])
        return {"status": "IGNORED", "update_id": update.get("update_id")}

    def get_updates(
        self, offset: Optional[int] = None, timeout: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetch updates via Telegram long-polling."""
        if not self.token:
            return []
        params: Dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        try:
            resp = self.http_client.get(
                f"{self.base_url}/getUpdates", params=params, timeout=timeout + 5.0
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    return data.get("result", [])
        except Exception:
            pass
        return []
