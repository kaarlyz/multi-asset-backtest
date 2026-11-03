#!/usr/bin/env python3
"""
Autonomous Telegram Bot Daemon for Chronos Backtesting System (@ekaresearchmulti_bot).
Runs continuous long-polling loop with exception recovery and graceful shutdown.
"""

import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.bot.telegram_service import TelegramBotService
from src.chronos.db import ChronosDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BotDaemon")


class BotDaemon:
    """Standalone Telegram Bot Daemon."""

    def __init__(self, env_path: Optional[Path] = None, db_path: Optional[Path] = None):
        self.project_root = project_root
        self.env_path = env_path or (self.project_root / ".env")
        load_dotenv(dotenv_path=self.env_path)

        self.db_path = db_path or (self.project_root / "data" / "chronos.duckdb")
        self.db = ChronosDB(str(self.db_path))

        self.bot_service = TelegramBotService(
            db=self.db,
            env_path=self.env_path,
        )

        self.stop_event = threading.Event()
        self._setup_signals()

    def _setup_signals(self) -> None:
        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info("Received signal %s (%d). Initiating graceful shutdown...", sig_name, signum)
            self.stop_event.set()

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, AttributeError):
            # In non-main thread or test environments where signal cannot be set
            pass

    def run(self, poll_timeout: int = 10) -> None:
        """Continuous long-polling loop with exception recovery."""
        logger.info("=" * 60)
        logger.info("🤖 Starting Chronos Telegram Bot Daemon")
        logger.info("Bot Username: @%s", self.bot_service.bot_username)
        logger.info("Allowed Users: %s", sorted(list(self.bot_service.allowed_user_ids)))
        logger.info("Database: %s", self.db_path)
        logger.info("Token status: %s", "Configured" if self.bot_service.token else "NOT CONFIGURED (Mock)")
        logger.info("=" * 60)

        offset: Optional[int] = None

        try:
            while not self.stop_event.is_set():
                try:
                    updates = self.bot_service.get_updates(offset=offset, timeout=poll_timeout)
                    for upd in updates:
                        offset = upd.get("update_id", 0) + 1
                        self.bot_service.handle_update(upd)
                except Exception as exc:
                    logger.error("Polling loop error: %s. Retrying in 2 seconds...", exc)
                    # Yield CPU / back off briefly before retrying
                    self.stop_event.wait(timeout=2.0)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Halt active simulation and clean up resources."""
        logger.info("Shutting down BotDaemon...")
        if self.bot_service.is_sim_running():
            logger.info("Active simulation detected. Stopping simulation thread...")
            try:
                self.bot_service.stop_simulation()
            except Exception as e:
                logger.error("Error stopping simulation during shutdown: %s", e)

        try:
            self.db.close()
        except Exception as e:
            logger.error("Error closing database connection: %s", e)

        logger.info("BotDaemon stopped cleanly.")


def main():
    daemon = BotDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
