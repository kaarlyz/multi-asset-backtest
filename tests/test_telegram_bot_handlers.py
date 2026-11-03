from datetime import datetime, timezone
import pytest
import httpx

from src.chronos.db import ChronosDB
from src.chronos.replay import HistoricalReplayEngine
from src.bot.telegram_service import TelegramBotService


@pytest.fixture
def test_db():
    database = ChronosDB(":memory:")
    database.insert_asset("CRYPTO_BTC", "CRYPTO", "BTCUSDT", "BTC", "USDT")
    yield database
    database.close()


@pytest.fixture
def mock_http_client():
    calls = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        calls.append({
            "url": str(request.url),
            "method": request.method,
            "content": request.read().decode("utf-8"),
        })
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 123}})

    transport = httpx.MockTransport(handle_request)
    client = httpx.Client(transport=transport)
    client.mock_calls = calls  # type: ignore
    return client


@pytest.fixture
def replay_engine(test_db):
    start = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc)
    return HistoricalReplayEngine(
        db=test_db,
        symbol="BTCUSDT",
        start_time=start,
        end_time=end,
        step_size="1h",
        market_data=None,
    )


@pytest.fixture
def bot_service(test_db, replay_engine, mock_http_client):
    service = TelegramBotService(
        token="MOCK_BOT_TOKEN_123456",
        allowed_user_ids=[6526833205],
        replay_engine=replay_engine,
        db=test_db,
        http_client=mock_http_client,
    )
    return service


def test_unauthorized_user_blocked(bot_service):
    """Verify security gate: unlisted user_id is blocked for both messages and callbacks."""
    unauthorized_id = 999999999

    # 1. Message from unauthorized user
    msg_update = {
        "from": {"id": unauthorized_id, "username": "attacker"},
        "chat": {"id": unauthorized_id},
        "text": "/status",
    }
    res_msg = bot_service.handle_message(msg_update)
    assert res_msg["status"] == "DROPPED_UNAUTHORIZED"

    # 2. Callback from unauthorized user
    cb_update = {
        "id": "cb_001",
        "from": {"id": unauthorized_id, "username": "attacker"},
        "data": "DECIDE:DEC_TEST_001:BUY",
    }
    res_cb = bot_service.handle_callback(cb_update)
    assert res_cb["status"] == "DROPPED_UNAUTHORIZED"

    # Decision should NOT be recorded in DB
    assert not bot_service.db.has_decision("DEC_TEST_001")


def test_authorized_commands(bot_service):
    """Verify authorized user can execute /status, /portfolio, /report, /pause, /resume."""
    auth_id = 6526833205

    # /status
    res_status = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/status",
    })
    assert res_status["status"] == "HANDLED"
    assert "SIMULATION STATUS:" in res_status["response"]

    # /portfolio
    res_port = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/portfolio",
    })
    assert res_port["status"] == "HANDLED"
    assert "PORTFOLIO SUMMARY:" in res_port["response"]
    assert "Cash: $100,000.00" in res_port["response"]

    # /pause & /resume
    assert bot_service.replay_engine.is_paused is False
    res_pause = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/pause",
    })
    assert res_pause["status"] == "HANDLED"
    assert bot_service.replay_engine.is_paused is True
    assert "PAUSED" in res_pause["response"]

    res_resume = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/resume",
    })
    assert res_resume["status"] == "HANDLED"
    assert bot_service.replay_engine.is_paused is False
    assert "RESUMED" in res_resume["response"]

    # /report
    res_rep = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/report",
    })
    assert res_rep["status"] == "HANDLED"
    assert "PIT FUNDAMENTAL & QUANT REPORT" in res_rep["response"]


def test_send_signal_prompt_structure(bot_service, mock_http_client):
    """Verify send_signal_prompt constructs inline keyboard with BUY, SHORT, IGNORE."""
    prompt = bot_service.send_signal_prompt(
        asset="BTCUSDT",
        current_price=43250.0,
        report_summary="Bullish RSI + TVL Spike",
        chat_id=6526833205,
        decision_id="DEC_BTC_TEST_99",
    )

    assert prompt["decision_id"] == "DEC_BTC_TEST_99"
    keyboard = prompt["reply_markup"]["inline_keyboard"][0]
    actions = [btn["text"] for btn in keyboard]
    callbacks = [btn["callback_data"] for btn in keyboard]

    assert any("BUY" in a for a in actions)
    assert any("SHORT" in a for a in actions)
    assert any("IGNORE" in a for a in actions)

    assert callbacks == [
        "DECIDE:DEC_BTC_TEST_99:BUY",
        "DECIDE:DEC_BTC_TEST_99:SHORT",
        "DECIDE:DEC_BTC_TEST_99:IGNORE",
    ]

    # Verify HTTP POST was dispatched
    assert len(mock_http_client.mock_calls) > 0
    last_call = mock_http_client.mock_calls[-1]
    assert "sendMessage" in last_call["url"]


def test_callbacks_write_to_decision_ledger(bot_service):
    """Verify BUY/SHORT/IGNORE callbacks write properly into decision_ledger."""
    auth_id = 6526833205

    # 1. Action: BUY
    res_buy = bot_service.handle_callback({
        "id": "cb_buy",
        "from": {"id": auth_id},
        "data": "DECIDE:DEC_ORDER_001:BUY",
    })
    assert res_buy["status"] == "RECORDED"
    assert res_buy["action"] == "BUY"

    db_rec_buy = bot_service.db.get_decision("DEC_ORDER_001")
    assert db_rec_buy is not None
    assert db_rec_buy["action"] == "BUY"
    assert db_rec_buy["interface"] == "TELEGRAM"
    assert db_rec_buy["actor_id"] == str(auth_id)

    # 2. Action: SHORT
    res_short = bot_service.handle_callback({
        "id": "cb_short",
        "from": {"id": auth_id},
        "data": "DECIDE:DEC_ORDER_002:SHORT",
    })
    assert res_short["status"] == "RECORDED"
    assert res_short["action"] == "SHORT"
    db_rec_short = bot_service.db.get_decision("DEC_ORDER_002")
    assert db_rec_short["action"] == "SHORT"

    # 3. Action: IGNORE
    res_ignore = bot_service.handle_callback({
        "id": "cb_ignore",
        "from": {"id": auth_id},
        "data": "DECIDE:DEC_ORDER_003:IGNORE",
    })
    assert res_ignore["status"] == "RECORDED"
    assert res_ignore["action"] == "IGNORE"
    db_rec_ignore = bot_service.db.get_decision("DEC_ORDER_003")
    assert db_rec_ignore["action"] == "IGNORE"


def test_duplicate_callback_rejected_by_idempotency_guard(bot_service):
    """Verify duplicate callback on the same decision_id is rejected by idempotency guard."""
    auth_id = 6526833205
    decision_id = "DEC_IDEMPOTENT_100"

    # First attempt: successfully records
    first_res = bot_service.handle_callback({
        "id": "cb_first",
        "from": {"id": auth_id},
        "data": f"DECIDE:{decision_id}:BUY",
    })
    assert first_res["status"] == "RECORDED"

    # Second attempt with same decision_id: must be rejected
    second_res = bot_service.handle_callback({
        "id": "cb_second",
        "from": {"id": auth_id},
        "data": f"DECIDE:{decision_id}:SHORT",  # Even if action differs
    })
    assert second_res["status"] == "REJECTED_ALREADY_DECIDED"

    # Third attempt: still rejected
    third_res = bot_service.handle_callback({
        "id": "cb_third",
        "from": {"id": auth_id},
        "data": f"DECIDE:{decision_id}:IGNORE",
    })
    assert third_res["status"] == "REJECTED_ALREADY_DECIDED"

    # Original decision remains intact
    record = bot_service.db.get_decision(decision_id)
    assert record["action"] == "BUY"


def test_start_sim_and_stop_sim_lifecycle(bot_service):
    """
    Verify /start_sim starts background interactive simulation thread,
    /status, /portfolio, /report reflect active run,
    concurrent /start_sim replies with warning,
    and /stop_sim halts thread cleanly returning portfolio summary.
    """
    auth_id = 6526833205

    # 1. Start simulation
    res_start = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/start_sim BTCUSDT 2024-01-01 2024-01-05",
    })
    assert res_start["status"] == "HANDLED"
    assert "🚀 Simulasi BTCUSDT dimulai (2024-01-01 s/d 2024-01-05). Mode: INTERACTIVE." in res_start["response"]

    # Verify simulation thread is alive and configured
    assert bot_service.is_sim_running() is True
    assert bot_service.sim_thread is not None
    assert bot_service.sim_thread.is_alive() is True
    assert bot_service.sim_runner is not None
    assert bot_service.sim_runner.mode == "INTERACTIVE"
    assert bot_service.sim_runner.step_delay == 2.0
    assert bot_service.replay_engine.active_asset == "BTCUSDT"

    # 2. Verify /status reflects active run
    res_status = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/status",
    })
    assert res_status["status"] == "HANDLED"
    assert "SIMULATION STATUS:" in res_status["response"]
    assert "Active Asset: BTCUSDT" in res_status["response"]
    assert "RUNNING ▶️" in res_status["response"]

    # 3. Verify /portfolio reflects active run
    res_port = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/portfolio",
    })
    assert res_port["status"] == "HANDLED"
    assert "PORTFOLIO SUMMARY:" in res_port["response"]
    assert "Cash: $10,000.00" in res_port["response"]

    # 4. Verify /report reflects active asset
    res_report = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/report",
    })
    assert res_report["status"] == "HANDLED"
    assert "PIT FUNDAMENTAL & QUANT REPORT (BTCUSDT):" in res_report["response"]

    # 5. Starting another simulation while one is active returns warning
    res_conflict = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/start_sim ETHUSDT 2024-01-01 2024-01-05",
    })
    assert res_conflict["status"] == "HANDLED"
    assert "⚠️ Simulasi sedang berjalan" in res_conflict["response"]

    # 6. Stop simulation via /stop_sim
    res_stop = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/stop_sim",
    })
    assert res_stop["status"] == "HANDLED"
    assert "🛑 Simulasi dihentikan." in res_stop["response"]
    assert "PORTFOLIO SUMMARY:" in res_stop["response"]

    # Thread and runner state checks
    assert bot_service.is_sim_running() is False
    assert bot_service.sim_thread.is_alive() is False
    assert bot_service.sim_runner.is_stopped is True

    # 7. Calling /stop_sim when no simulation is running
    res_stop_again = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/stop_sim",
    })
    assert res_stop_again["status"] == "HANDLED"
    assert "⚠️ Tidak ada simulasi yang sedang berjalan." in res_stop_again["response"]


def test_start_sim_default_arguments(bot_service):
    """Verify /start_sim applies default syntax: BTCUSDT 2024-01-01 2024-03-31."""
    auth_id = 6526833205

    res = bot_service.handle_message({
        "from": {"id": auth_id},
        "chat": {"id": auth_id},
        "text": "/start_sim",
    })
    assert res["status"] == "HANDLED"
    assert "🚀 Simulasi BTCUSDT dimulai (2024-01-01 s/d 2024-03-31). Mode: INTERACTIVE." in res["response"]
    assert bot_service.is_sim_running() is True

    # Clean up
    bot_service.stop_simulation()
    assert bot_service.is_sim_running() is False


def test_start_sim_and_stop_sim_unauthorized(bot_service):
    """Verify unauthorized users cannot execute /start_sim or /stop_sim."""
    unauthorized_id = 111222333

    res_start = bot_service.handle_message({
        "from": {"id": unauthorized_id},
        "chat": {"id": unauthorized_id},
        "text": "/start_sim BTCUSDT",
    })
    assert res_start["status"] == "DROPPED_UNAUTHORIZED"
    assert bot_service.is_sim_running() is False

    res_stop = bot_service.handle_message({
        "from": {"id": unauthorized_id},
        "chat": {"id": unauthorized_id},
        "text": "/stop_sim",
    })
    assert res_stop["status"] == "DROPPED_UNAUTHORIZED"


def test_bot_daemon_lifecycle_and_shutdown(test_db):
    """Verify BotDaemon initializes, polls safely, and handles shutdown cleanly."""
    from src.bot_daemon import BotDaemon

    daemon = BotDaemon()
    daemon.db = test_db
    daemon.bot_service.db = test_db
    # Signal daemon to stop immediately after start
    daemon.stop_event.set()
    daemon.run(poll_timeout=1)
    assert daemon.stop_event.is_set()


