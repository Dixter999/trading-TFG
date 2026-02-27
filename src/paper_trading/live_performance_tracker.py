"""Live Performance Tracker for adaptive position sizing (Kelly v3).

Kelly v3 formula chain:
  1. Lifetime Kelly: max(0, W - (1-W)/R)
  2. EMA Kelly: max(0, ema_wr - (1-ema_wr)/R)
  3. Blend: min((1-b)*lifetime + b*ema, lifetime)  — EMA can only reduce
  4. Decay (HL=5) + Streak bonus (+15%/win, cap 5)
  5. Ramp: n<10 → unknown_weight; 10-20 → blend toward formula
  6. Clamp to [floor, ceiling]

Drawdown throttle is applied externally in main.py (portfolio-level).

Signal key format: "EURUSD:LONG:MACD_Stoch_long" (no timeframe — DB entry_model
doesn't store it, and merging across TFs is acceptable).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class KellyConfig:
    """Kelly v3 configuration parameters (loaded from trading_config_params)."""

    version: int = 3
    ramp_start: int = 10
    ramp_end: int = 20
    unknown_weight: float = 0.50
    min_weight: float = 0.15
    max_weight: float = 2.00
    decay_halflife: float = 5.0
    streak_bonus: float = 0.15
    streak_cap: int = 5
    ema_span: int = 30
    ema_blend: float = 0.50
    dd_throttle_start: float = 0.03
    dd_throttle_end: float = 0.10


@dataclass
class SignalStats:
    """Running statistics for a single signal."""

    wins: int = 0
    losses: int = 0
    total_pips: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    ema_wr: float = 0.5
    last_updated: datetime | None = None


class LivePerformanceTracker:
    """Track live trade outcomes per signal and compute sizing weights.

    Kelly v3 formula (see docs/kelly_v3.md for full spec):
        W  = wins / total              (lifetime win rate)
        R  = avg_win / |avg_loss|      (reward/risk ratio)

        kelly_lifetime = max(0, W - (1-W)/R)
        kelly_ema      = max(0, ema_wr - (1-ema_wr)/R)
        blended = (1-b)*kelly_lifetime + b*kelly_ema
        kelly_raw = min(blended, kelly_lifetime)

        decay = 0.5 ^ (cl / decay_halflife)
        boost = 1 + streak_bonus * min(cw, streak_cap)
        weight = kelly_raw * decay * boost

    Ramp schedule:
        < ramp_start:  weight = unknown_weight (0.50)
        ramp_start–ramp_end: linear blend from unknown_weight → computed
        >= ramp_end: weight = computed
        Final: clamp(weight, min_weight, max_weight)
    """

    def __init__(self, config: KellyConfig | None = None) -> None:
        self._stats: dict[str, SignalStats] = {}
        self._db_pool = None  # Set by bootstrap_from_db for persistence
        self.config = config or KellyConfig()

    async def _load_config_from_db(self, db_pool) -> None:
        """Load Kelly v3 config parameters from trading_config_params table."""
        query = """
            SELECT param_key, param_value
            FROM trading_config_params
            WHERE category = 'kelly'
        """
        try:
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(query)

            param_map = {row["param_key"]: row["param_value"] for row in rows}

            cfg = self.config
            cfg.version = int(float(param_map.get("kelly:version", cfg.version)))
            cfg.ramp_start = int(float(param_map.get("kelly:ramp_start", cfg.ramp_start)))
            cfg.ramp_end = int(float(param_map.get("kelly:ramp_end", cfg.ramp_end)))
            cfg.unknown_weight = float(param_map.get("kelly:unknown_weight", cfg.unknown_weight))
            cfg.min_weight = float(param_map.get("kelly:min_weight", cfg.min_weight))
            cfg.max_weight = float(param_map.get("kelly:max_weight", cfg.max_weight))
            cfg.decay_halflife = float(param_map.get("kelly:decay_halflife", cfg.decay_halflife))
            cfg.streak_bonus = float(param_map.get("kelly:streak_bonus", cfg.streak_bonus))
            cfg.streak_cap = int(float(param_map.get("kelly:streak_cap", cfg.streak_cap)))
            cfg.ema_span = int(float(param_map.get("kelly:ema_span", cfg.ema_span)))
            cfg.ema_blend = float(param_map.get("kelly:ema_blend", cfg.ema_blend))
            cfg.dd_throttle_start = float(param_map.get("kelly:dd_throttle_start", cfg.dd_throttle_start))
            cfg.dd_throttle_end = float(param_map.get("kelly:dd_throttle_end", cfg.dd_throttle_end))

            logger.info(
                f"Kelly config loaded from DB: version={cfg.version}, "
                f"ema_span={cfg.ema_span}, ema_blend={cfg.ema_blend}, "
                f"floor={cfg.min_weight}, ceiling={cfg.max_weight}, "
                f"decay_hl={cfg.decay_halflife}, streak={cfg.streak_bonus}x{cfg.streak_cap}, "
                f"dd_throttle={cfg.dd_throttle_start}-{cfg.dd_throttle_end}"
            )
        except Exception as e:
            logger.warning(f"Failed to load Kelly config from DB, using defaults: {e}")

    async def bootstrap_from_db(self, db_pool) -> None:
        """Load historical paper_trades and rebuild stats.

        Processes trades chronologically so consecutive_losses/wins and
        ema_wr are accurate. Persists all Kelly weights to
        signal_kelly_weights table for TypeScript/frontend consumption.
        """
        self._db_pool = db_pool

        # Load config params first
        await self._load_config_from_db(db_pool)

        query = """
            SELECT symbol, direction, entry_model, pnl_pips, exit_time
            FROM paper_trades
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time ASC
        """
        try:
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(query)

            count = 0
            for row in rows:
                key = self.normalize_signal_key(
                    symbol=row["symbol"],
                    direction=row["direction"],
                    entry_model=row["entry_model"],
                )
                if key is None:
                    continue
                pnl = float(row["pnl_pips"]) if row["pnl_pips"] is not None else 0.0
                self.record_trade(key, pnl, persist=False)
                count += 1

            logger.info(
                f"LivePerformanceTracker bootstrapped: {count} trades, "
                f"{len(self._stats)} signals tracked (Kelly v{self.config.version})"
            )

            # Log weights for all signals with enough trades
            cfg = self.config
            for sig_key, stats in sorted(self._stats.items()):
                total = stats.wins + stats.losses
                if total >= cfg.ramp_start:
                    w = self.get_weight(sig_key)
                    logger.info(
                        f"  {sig_key}: W={stats.wins}/{total} "
                        f"ema_wr={stats.ema_wr:.3f} "
                        f"avg_win={stats.avg_win_pips:.1f} avg_loss={stats.avg_loss_pips:.1f} "
                        f"cw={stats.consecutive_wins} cl={stats.consecutive_losses} "
                        f"weight={w:.3f}"
                    )

            # Bulk persist all Kelly weights to DB
            await self._persist_all_kelly_to_db()

        except Exception as e:
            logger.error(f"LivePerformanceTracker bootstrap failed: {e}")

    def record_trade(self, signal_key: str, pnl_pips: float, *, persist: bool = True) -> None:
        """Update stats after a trade closes.

        Args:
            signal_key: Normalized signal key (e.g. "EURUSD:LONG:MACD_Stoch_long")
            pnl_pips: Trade P&L in pips (positive = win, negative/zero = loss)
            persist: If True and db_pool available, persist Kelly weight to DB.
                     Set False during bulk bootstrap to avoid per-row writes.
        """
        if signal_key not in self._stats:
            self._stats[signal_key] = SignalStats()

        s = self._stats[signal_key]
        is_win = pnl_pips > 0

        if is_win:
            s.wins += 1
            s.avg_win_pips = (
                (s.avg_win_pips * (s.wins - 1) + pnl_pips) / s.wins
            )
            s.consecutive_losses = 0
            s.consecutive_wins += 1
        else:
            s.losses += 1
            s.avg_loss_pips = (
                (s.avg_loss_pips * (s.losses - 1) + pnl_pips) / s.losses
            )
            s.consecutive_losses += 1
            s.consecutive_wins = 0

        # v3: EMA win rate update (always runs, cost-free warm-up)
        alpha = 2.0 / (self.config.ema_span + 1)
        s.ema_wr = s.ema_wr * (1.0 - alpha) + (1.0 if is_win else 0.0) * alpha

        s.total_pips += pnl_pips
        s.last_updated = datetime.now(timezone.utc)

        if persist and self._db_pool is not None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist_kelly_to_db(signal_key))
            except RuntimeError:
                pass  # No event loop — skip async persist

    def get_weight(self, signal_key: str) -> float:
        """Return sizing weight for a signal using Kelly v3.

        Returns unknown_weight (0.50) if fewer than ramp_start trades.
        Between ramp_start and ramp_end, linearly blends from
        unknown_weight toward the computed Kelly weight.
        """
        cfg = self.config
        stats = self._stats.get(signal_key)
        if not stats:
            return cfg.unknown_weight

        total = stats.wins + stats.losses
        if total < cfg.ramp_start:
            return cfg.unknown_weight

        W = stats.wins / total
        R = stats.avg_win_pips / max(abs(stats.avg_loss_pips), 0.1)

        if R <= 0:
            # No wins or no meaningful reward → floor
            if total < cfg.ramp_end:
                t = (total - cfg.ramp_start) / (cfg.ramp_end - cfg.ramp_start)
                return max((1.0 - t) * cfg.unknown_weight + t * cfg.min_weight, cfg.min_weight)
            return cfg.min_weight

        # Step 1: Lifetime Kelly (full, not half)
        kelly_lifetime = max(0.0, W - (1.0 - W) / R)

        # Step 2: EMA Kelly
        kelly_ema = max(0.0, stats.ema_wr - (1.0 - stats.ema_wr) / R)

        # Step 3: Blend (EMA can only reduce, never increase)
        blended = (1.0 - cfg.ema_blend) * kelly_lifetime + cfg.ema_blend * kelly_ema
        kelly_raw = min(blended, kelly_lifetime)

        # Step 4: Decay + streak boost
        decay = 0.5 ** (stats.consecutive_losses / cfg.decay_halflife)
        boost = 1.0 + cfg.streak_bonus * min(stats.consecutive_wins, cfg.streak_cap)
        weight = kelly_raw * decay * boost

        # Step 5: Ramp
        if total < cfg.ramp_end:
            t = (total - cfg.ramp_start) / (cfg.ramp_end - cfg.ramp_start)
            weight = (1.0 - t) * cfg.unknown_weight + t * weight

        # Step 6: Clamp
        return max(min(weight, cfg.max_weight), cfg.min_weight)

    def get_all_weights(self) -> dict[str, float]:
        """Return weights for all tracked signals."""
        return {key: self.get_weight(key) for key in self._stats}

    def _get_kelly_breakdown(self, signal_key: str) -> dict:
        """Return full Kelly weight breakdown for a signal (for DB persistence)."""
        cfg = self.config
        no_data = {
            "raw_kelly": 0.0, "decay_factor": 1.0,
            "blend_phase": "No Trades", "final_weight": cfg.unknown_weight,
            "win_rate": None, "reward_ratio": None,
            "ema_wr": 0.5, "ema_kelly": 0.0,
            "consecutive_wins": 0, "streak_bonus": 1.0,
            "kelly_version": cfg.version,
        }

        stats = self._stats.get(signal_key)
        if not stats:
            return no_data

        total = stats.wins + stats.losses
        if total == 0:
            return no_data

        W = stats.wins / total
        R = stats.avg_win_pips / max(abs(stats.avg_loss_pips), 0.1)

        # Step 1-3: Kelly computation
        kelly_lifetime = max(0.0, W - (1.0 - W) / R) if R > 0 else 0.0
        kelly_ema = max(0.0, stats.ema_wr - (1.0 - stats.ema_wr) / R) if R > 0 else 0.0
        blended = (1.0 - cfg.ema_blend) * kelly_lifetime + cfg.ema_blend * kelly_ema
        kelly_raw = min(blended, kelly_lifetime)

        # Step 4: Decay + boost
        decay = 0.5 ** (stats.consecutive_losses / cfg.decay_halflife)
        boost = 1.0 + cfg.streak_bonus * min(stats.consecutive_wins, cfg.streak_cap)

        # Step 5: Phase-based weight
        if total < cfg.ramp_start:
            blend_phase = f"Flat {cfg.unknown_weight}"
            final_weight = cfg.unknown_weight
        elif total >= cfg.ramp_end:
            blend_phase = "Full Kelly"
            final_weight = kelly_raw * decay * boost
        else:
            pct = round(((total - cfg.ramp_start) / (cfg.ramp_end - cfg.ramp_start)) * 100)
            blend_phase = f"Ramp {pct}%"
            t = (total - cfg.ramp_start) / (cfg.ramp_end - cfg.ramp_start)
            computed = kelly_raw * decay * boost
            final_weight = (1.0 - t) * cfg.unknown_weight + t * computed

        # Step 6: Clamp
        final_weight = max(min(final_weight, cfg.max_weight), cfg.min_weight)

        return {
            "raw_kelly": kelly_lifetime,
            "decay_factor": decay,
            "blend_phase": blend_phase,
            "final_weight": final_weight,
            "win_rate": W,
            "reward_ratio": R,
            "ema_wr": stats.ema_wr,
            "ema_kelly": kelly_ema,
            "consecutive_wins": stats.consecutive_wins,
            "streak_bonus": boost,
            "kelly_version": cfg.version,
        }

    async def _persist_kelly_to_db(self, signal_key: str) -> None:
        """UPSERT a single signal's Kelly weight to signal_kelly_weights."""
        if self._db_pool is None:
            return
        stats = self._stats.get(signal_key)
        if stats is None:
            return

        parts = signal_key.split(":")
        if len(parts) < 3:
            return
        symbol, direction, signal_name = parts[0], parts[1], ":".join(parts[2:])

        breakdown = self._get_kelly_breakdown(signal_key)

        upsert = """
            INSERT INTO signal_kelly_weights
                (signal_key, symbol, direction, signal_name,
                 wins, losses, total_pips, avg_win_pips, avg_loss_pips,
                 consecutive_losses, raw_kelly, decay_factor, blend_phase,
                 final_weight, win_rate, reward_ratio,
                 ema_wr, consecutive_wins, streak_bonus, ema_kelly, kelly_version,
                 updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,
                    $17,$18,$19,$20,$21,NOW())
            ON CONFLICT (signal_key) DO UPDATE SET
                wins=$5, losses=$6, total_pips=$7, avg_win_pips=$8, avg_loss_pips=$9,
                consecutive_losses=$10, raw_kelly=$11, decay_factor=$12, blend_phase=$13,
                final_weight=$14, win_rate=$15, reward_ratio=$16,
                ema_wr=$17, consecutive_wins=$18, streak_bonus=$19,
                ema_kelly=$20, kelly_version=$21, updated_at=NOW()
        """
        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    upsert,
                    signal_key, symbol, direction, signal_name,
                    stats.wins, stats.losses, stats.total_pips,
                    stats.avg_win_pips, stats.avg_loss_pips,
                    stats.consecutive_losses,
                    breakdown["raw_kelly"], breakdown["decay_factor"],
                    breakdown["blend_phase"], breakdown["final_weight"],
                    breakdown["win_rate"], breakdown["reward_ratio"],
                    breakdown["ema_wr"], breakdown["consecutive_wins"],
                    breakdown["streak_bonus"], breakdown["ema_kelly"],
                    breakdown["kelly_version"],
                )
        except Exception as e:
            logger.warning(f"Failed to persist Kelly weight for {signal_key}: {e}")

    async def _persist_all_kelly_to_db(self) -> None:
        """Bulk UPSERT all Kelly weights — called after bootstrap."""
        if self._db_pool is None:
            return
        count = 0
        for signal_key in self._stats:
            await self._persist_kelly_to_db(signal_key)
            count += 1
        logger.info(f"Persisted {count} Kelly weights to signal_kelly_weights table")

    @staticmethod
    def normalize_signal_key(
        symbol: str, direction: str, entry_model: str | None
    ) -> str | None:
        """Convert DB fields to a consistent signal key.

        DB entry_model format: "Hybrid_V4 + MACD_Stoch_long"
        Output key format:     "EURUSD:LONG:MACD_Stoch_long"

        Returns None if entry_model is missing or unparseable.
        """
        if not entry_model:
            return None

        # Strip "Hybrid_V4 + " prefix
        if " + " in entry_model:
            signal_name = entry_model.split(" + ", 1)[1].strip()
        else:
            signal_name = entry_model.strip()

        if not signal_name:
            return None

        return f"{symbol.upper()}:{direction.upper()}:{signal_name}"
