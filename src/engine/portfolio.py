from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class SimPortfolio:
    """
    Simulated Portfolio & Execution Engine.
    Tracks mark-to-market equity, open positions, closed trades, SL/TP execution,
    and cumulative performance metrics.
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        default_risk_pct: float = 0.02,
        default_sl_pct: float = 0.05,
        default_tp_pct: float = 0.10,
        max_leverage: float = 5.0,
    ):
        self.initial_cash: float = float(initial_cash)
        self.cash: float = float(initial_cash)
        self.equity: float = float(initial_cash)
        self.peak_equity: float = float(initial_cash)
        self.max_drawdown_pct: float = 0.0

        self.default_risk_pct: float = float(default_risk_pct)
        self.default_sl_pct: float = float(default_sl_pct)
        self.default_tp_pct: float = float(default_tp_pct)
        self.max_leverage: float = float(max_leverage)

        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []

        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0

    def update_bar(self, bar: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Update mark-to-market equity, evaluate SL/TP triggers on bar high/low,
        close positions if SL/TP hit, and update drawdown metrics.
        Returns list of trades closed during this bar.
        """
        closed_in_bar: List[Dict[str, Any]] = []
        if not bar:
            return closed_in_bar

        symbol = bar.get("symbol")
        high = float(bar.get("high", bar.get("close", 0.0)))
        low = float(bar.get("low", bar.get("close", 0.0)))
        close = float(bar.get("close", 0.0))
        bar_time = bar.get("t_event") or bar.get("t_avail")

        # Check open positions matching symbol, or all positions if symbol not specified
        symbols_to_evaluate = [symbol] if symbol and symbol in self.open_positions else list(self.open_positions.keys())

        for sym in symbols_to_evaluate:
            pos = self.open_positions.get(sym)
            if not pos:
                continue

            side = pos["side"]
            sl = pos.get("sl")
            tp = pos.get("tp")

            if side == "LONG":
                hit_sl = sl is not None and low <= sl
                hit_tp = tp is not None and high >= tp
                if hit_sl and hit_tp:
                    # Conservative: stop loss triggers first
                    tr = self._close_position(sym, exit_price=float(sl) if sl is not None else low, exit_reason="SL", exit_time=bar_time)
                    if tr:
                        closed_in_bar.append(tr)
                elif hit_sl:
                    tr = self._close_position(sym, exit_price=float(sl) if sl is not None else low, exit_reason="SL", exit_time=bar_time)
                    if tr:
                        closed_in_bar.append(tr)
                elif hit_tp:
                    tr = self._close_position(sym, exit_price=float(tp) if tp is not None else high, exit_reason="TP", exit_time=bar_time)
                    if tr:
                        closed_in_bar.append(tr)

            elif side == "SHORT":
                hit_sl = sl is not None and high >= sl
                hit_tp = tp is not None and low <= tp
                if hit_sl and hit_tp:
                    tr = self._close_position(sym, exit_price=float(sl) if sl is not None else high, exit_reason="SL", exit_time=bar_time)
                    if tr:
                        closed_in_bar.append(tr)
                elif hit_sl:
                    tr = self._close_position(sym, exit_price=float(sl) if sl is not None else high, exit_reason="SL", exit_time=bar_time)
                    if tr:
                        closed_in_bar.append(tr)
                elif hit_tp:
                    tr = self._close_position(sym, exit_price=float(tp) if tp is not None else low, exit_reason="TP", exit_time=bar_time)
                    if tr:
                        closed_in_bar.append(tr)

        # Recalculate mark-to-market equity & drawdown
        self._recalculate_equity(current_price=close, symbol=symbol)
        return closed_in_bar

    def execute_decision(
        self,
        action: str,
        bar: Dict[str, Any],
        decision_id: str,
        actor_id: str,
        risk_pct: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute trading decision:
          - BUY: closes existing SHORT (if any), opens LONG with risk_pct sizing.
          - SHORT: closes existing LONG (if any), opens SHORT with risk_pct sizing.
          - IGNORE: logs pass, does nothing.
        """
        act = (action or "").upper()
        if act == "IGNORE":
            return None

        if not bar or "close" not in bar:
            return None

        symbol = bar.get("symbol", "BTCUSDT")
        entry_price = float(bar["close"])
        bar_time = bar.get("t_event") or bar.get("t_avail")
        risk = float(risk_pct) if risk_pct is not None else self.default_risk_pct
        stop_pct = float(sl_pct) if sl_pct is not None else self.default_sl_pct
        prof_pct = float(tp_pct) if tp_pct is not None else self.default_tp_pct

        if entry_price <= 0.0:
            return None

        if act == "BUY":
            # Close existing SHORT if any
            if symbol in self.open_positions:
                curr = self.open_positions[symbol]
                if curr["side"] == "SHORT":
                    self._close_position(
                        symbol,
                        exit_price=entry_price,
                        exit_reason="SIGNAL",
                        exit_time=bar_time,
                    )
                elif curr["side"] == "LONG":
                    # Already long, maintain position
                    return curr

            calc_sl = sl if sl is not None else entry_price * (1.0 - stop_pct)
            calc_tp = tp if tp is not None else entry_price * (1.0 + prof_pct)
            sl_dist = abs(entry_price - calc_sl)

            risk_amt = self.equity * risk
            size = (risk_amt / sl_dist) if sl_dist > 0 else (risk_amt / entry_price)
            # Leverage cap
            max_size = (self.equity * self.max_leverage) / entry_price if entry_price > 0 else 0.0
            size = min(size, max_size)
            size = round(size, 6)

            pos = {
                "symbol": symbol,
                "side": "LONG",
                "size": size,
                "entry_price": entry_price,
                "sl": round(calc_sl, 4),
                "tp": round(calc_tp, 4),
                "entry_time": bar_time,
                "decision_id": decision_id,
                "actor_id": actor_id,
            }
            self.open_positions[symbol] = pos
            self._recalculate_equity(current_price=entry_price, symbol=symbol)
            return pos

        elif act == "SHORT":
            # Close existing LONG if any
            if symbol in self.open_positions:
                curr = self.open_positions[symbol]
                if curr["side"] == "LONG":
                    self._close_position(
                        symbol,
                        exit_price=entry_price,
                        exit_reason="SIGNAL",
                        exit_time=bar_time,
                    )
                elif curr["side"] == "SHORT":
                    # Already short, maintain position
                    return curr

            calc_sl = sl if sl is not None else entry_price * (1.0 + stop_pct)
            calc_tp = tp if tp is not None else entry_price * (1.0 - prof_pct)
            sl_dist = abs(calc_sl - entry_price)

            risk_amt = self.equity * risk
            size = (risk_amt / sl_dist) if sl_dist > 0 else (risk_amt / entry_price)
            # Leverage cap
            max_size = (self.equity * self.max_leverage) / entry_price if entry_price > 0 else 0.0
            size = min(size, max_size)
            size = round(size, 6)

            pos = {
                "symbol": symbol,
                "side": "SHORT",
                "size": size,
                "entry_price": entry_price,
                "sl": round(calc_sl, 4),
                "tp": round(calc_tp, 4),
                "entry_time": bar_time,
                "decision_id": decision_id,
                "actor_id": actor_id,
            }
            self.open_positions[symbol] = pos
            self._recalculate_equity(current_price=entry_price, symbol=symbol)
            return pos

        return None

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        exit_time: Any,
    ) -> Optional[Dict[str, Any]]:
        """Close an active position and record realized PnL into cash balance."""
        pos = self.open_positions.pop(symbol, None)
        if not pos:
            return None

        side = pos["side"]
        size = pos["size"]
        entry_price = pos["entry_price"]

        if side == "LONG":
            realized_pnl = (exit_price - entry_price) * size
            return_pct = ((exit_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
        else:
            realized_pnl = (entry_price - exit_price) * size
            return_pct = ((entry_price - exit_price) / entry_price) * 100.0 if entry_price > 0 else 0.0

        realized_pnl = round(realized_pnl, 4)
        return_pct = round(return_pct, 4)

        self.cash = round(self.cash + realized_pnl, 4)
        self.realized_pnl = round(self.realized_pnl + realized_pnl, 4)

        trade = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": pos.get("entry_time"),
            "exit_time": exit_time,
            "realized_pnl": realized_pnl,
            "return_pct": return_pct,
            "exit_reason": exit_reason,
            "decision_id": pos.get("decision_id"),
            "actor_id": pos.get("actor_id"),
        }
        self.closed_trades.append(trade)
        return trade

    def _recalculate_equity(
        self,
        current_price: Optional[float] = None,
        symbol: Optional[str] = None,
    ) -> None:
        """Mark-to-market unrealized PnL, equity, peak equity, and max drawdown."""
        unrealized = 0.0
        for sym, pos in self.open_positions.items():
            price = current_price if (symbol is None or sym == symbol) and current_price is not None else pos["entry_price"]
            if pos["side"] == "LONG":
                unrealized += (price - pos["entry_price"]) * pos["size"]
            else:
                unrealized += (pos["entry_price"] - price) * pos["size"]

        self.unrealized_pnl = round(unrealized, 4)
        self.equity = round(self.cash + self.unrealized_pnl, 4)

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = round(dd, 4)

    def get_summary(self) -> Dict[str, Any]:
        """Return cumulative portfolio performance metrics."""
        total_trades = len(self.closed_trades)
        win_trades = len([t for t in self.closed_trades if t["realized_pnl"] > 0])
        win_rate = round(win_trades / total_trades, 4) if total_trades > 0 else 0.0
        total_return_pct = round(((self.equity - self.initial_cash) / self.initial_cash) * 100.0, 4) if self.initial_cash > 0 else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": round(self.max_drawdown_pct * 100.0, 4),
            "final_equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "open_positions": len(self.open_positions),
        }

    # Dict-like compatibility for TelegramBotService and logging
    def get(self, key: str, default: Any = None) -> Any:
        if key == "cash":
            return self.cash
        elif key == "equity":
            return self.equity
        elif key == "open_positions":
            return list(self.open_positions.values())
        elif key == "unrealized_pnl":
            return self.unrealized_pnl
        elif key == "realized_pnl":
            return self.realized_pnl
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)
