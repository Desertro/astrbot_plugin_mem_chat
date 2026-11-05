"""Measure human conversation traffic without counting assistant output."""

import math
import time
from collections import deque


class GroupActivity:
    """Keep bounded arrival windows and smoothed rates for each conversation."""

    def __init__(self):
        self.groups = {}

    def record(self, umo: str, sender_id: str, *, now: float | None = None):
        """Record one deduplicated human message, including stickers.

        Args:
            umo: Group conversation scope.
            sender_id: Stable identifier used to merge adjacent sentence fragments.
            now: Monotonic time, injectable for deterministic verification.
        """
        now = time.monotonic() if now is None else now
        group = self.groups.setdefault(
            umo,
            {
                "rows": deque(maxlen=4096),
                "rate": 0.0,
                "sampled_at": now,
                "last_at": now,
                "unit_at": -math.inf,
                "sender": "",
                "revision": 0,
                "silent_turns": 0,
            },
        )
        merged = sender_id == group["sender"] and now - group["unit_at"] < 8
        if not merged:
            group["unit_at"] = now
        group["rows"].append((now, sender_id, 0 if merged else 1))
        group["sender"] = sender_id
        group["last_at"] = now
        group["revision"] += 1
        self.snapshot(umo, now=now)

    def snapshot(self, umo: str, *, now: float | None = None) -> dict:
        """Return traffic metrics with fast rises and gradual cooling.

        Args:
            umo: Group conversation scope.
            now: Monotonic observation time.

        Returns:
            Two-minute and ten-minute rates, recent speakers, and arrival age.
        """
        now = time.monotonic() if now is None else now
        group = self.groups.get(umo)
        if group is None:
            return {
                "rate": 0.0,
                "short_rate": 0.0,
                "long_rate": 0.0,
                "speakers": 0,
                "age": math.inf,
                "revision": 0,
                "silent_turns": 0,
            }
        rows = group["rows"]
        while rows and now - rows[0][0] > 600:
            rows.popleft()
        recent = [row for row in rows if now - row[0] <= 120]
        short = sum(row[2] for row in recent) / 2
        long = sum(row[2] for row in rows) / 10
        target = 0.7 * short + 0.3 * long
        elapsed = max(0.0, now - group["sampled_at"])
        if target >= group["rate"]:
            rate = target
        else:
            rate = target + (group["rate"] - target) * math.exp(-elapsed / 180)
        group["rate"] = rate
        group["sampled_at"] = now
        return {
            "rate": rate,
            "short_rate": short,
            "long_rate": long,
            "speakers": len({row[1] for row in recent}),
            "age": max(0.0, now - group["last_at"]),
            "revision": group["revision"],
            "silent_turns": group["silent_turns"],
        }

    def finish(self, umo: str, *, sent: bool, directed: bool = False):
        """Track intentional silence without treating endpoint failures as decisions.

        Args:
            umo: Group conversation scope.
            sent: Whether visible content was sent.
            directed: Whether this was an explicit call rather than ordinary chat.
        """
        group = self.groups.get(umo)
        if group is not None:
            if sent or directed:
                group["silent_turns"] = 0
            else:
                group["silent_turns"] = min(3, group["silent_turns"] + 1)

    def interval(self, umo: str, *, now: float | None = None) -> float:
        """Choose a bounded observation interval from recent traffic.

        Args:
            umo: Group conversation scope.
            now: Monotonic observation time.

        Returns:
            Seconds from 45 to 180, with modest backoff after intentional silence.
        """
        metrics = self.snapshot(umo, now=now)
        base = 45 + 135 * min(1.0, metrics["rate"] / 12)
        return min(180.0, base * (1 + 0.15 * metrics["silent_turns"]))
