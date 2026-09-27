"""Run-wide budgets and admission control for model and tool work."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from parallax.errors import BudgetExceededError


@dataclass(frozen=True)
class RunBudget:
    max_wall_seconds: float = 900.0
    max_tool_calls: int = 24
    max_model_turns: int = 16
    max_total_tokens: int = 200_000
    max_tool_seconds: float = 60.0
    max_output_bytes: int = 1_000_000
    max_observation_bytes: int = 32_000
    max_artifact_bytes: int = 512 * 1024 * 1024
    max_total_artifact_bytes: int = 2 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        values = {
            "max_wall_seconds": self.max_wall_seconds,
            "max_tool_calls": self.max_tool_calls,
            "max_model_turns": self.max_model_turns,
            "max_total_tokens": self.max_total_tokens,
            "max_tool_seconds": self.max_tool_seconds,
            "max_output_bytes": self.max_output_bytes,
            "max_observation_bytes": self.max_observation_bytes,
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_total_artifact_bytes": self.max_total_artifact_bytes,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid:
            raise ValueError("run budget values must be positive: " + ", ".join(invalid))


class BudgetLedger:
    def __init__(self, budget: RunBudget) -> None:
        self.budget = budget
        self.started = time.monotonic()
        self.tool_calls = 0
        self.model_turns = 0
        self.tokens = 0
        self.output_bytes = 0
        self._lock = threading.Lock()

    def remaining_wall_seconds(self) -> float:
        return max(0.0, self.budget.max_wall_seconds - (time.monotonic() - self.started))

    def reserve_tool_call(self) -> float:
        with self._lock:
            if self.tool_calls >= self.budget.max_tool_calls:
                raise BudgetExceededError("run tool-call budget exhausted")
            remaining = self.remaining_wall_seconds()
            if remaining <= 0:
                raise BudgetExceededError("run wall-clock budget exhausted")
            self.tool_calls += 1
            return min(self.budget.max_tool_seconds, remaining)

    def reserve_model_turn(self) -> None:
        with self._lock:
            if self.model_turns >= self.budget.max_model_turns:
                raise BudgetExceededError("run model-turn budget exhausted")
            if self.remaining_wall_seconds() <= 0:
                raise BudgetExceededError("run wall-clock budget exhausted")
            self.model_turns += 1

    def record_tokens(self, count: int) -> None:
        with self._lock:
            self.tokens += max(0, count)
            if self.tokens > self.budget.max_total_tokens:
                raise BudgetExceededError("run token budget exhausted")

    def record_output_bytes(self, count: int) -> None:
        with self._lock:
            self.output_bytes += max(0, count)
            if self.output_bytes > self.budget.max_output_bytes:
                raise BudgetExceededError("run output budget exhausted")

    def snapshot(self) -> dict[str, int | float]:
        return {
            "tool_calls": self.tool_calls,
            "model_turns": self.model_turns,
            "tokens": self.tokens,
            "output_bytes": self.output_bytes,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
        }

    def lease(self, budget: RunBudget) -> "BudgetLease":
        """Create a child quota that also consumes this ledger's hard limits."""

        return BudgetLease(self, budget)


class BudgetLease:
    """A child budget whose every reservation is charged to a parent ledger.

    The local lock is held while charging the parent, so concurrent calls cannot
    race past the child quota. The parent ledger remains the authoritative run-wide
    limit shared by every agent.
    """

    def __init__(self, parent: BudgetLedger | "BudgetLease", budget: RunBudget) -> None:
        self.parent = parent
        self.budget = budget
        self.started = time.monotonic()
        self.tool_calls = 0
        self.model_turns = 0
        self.tokens = 0
        self.output_bytes = 0
        self._lock = threading.Lock()

    def remaining_wall_seconds(self) -> float:
        local = max(0.0, self.budget.max_wall_seconds - (time.monotonic() - self.started))
        return min(local, self.parent.remaining_wall_seconds())

    def reserve_tool_call(self) -> float:
        with self._lock:
            if self.tool_calls >= self.budget.max_tool_calls:
                raise BudgetExceededError("agent tool-call budget exhausted")
            remaining = self.remaining_wall_seconds()
            if remaining <= 0:
                raise BudgetExceededError("agent wall-clock budget exhausted")
            parent_timeout = self.parent.reserve_tool_call()
            self.tool_calls += 1
            return min(self.budget.max_tool_seconds, remaining, parent_timeout)

    def reserve_model_turn(self) -> None:
        with self._lock:
            if self.model_turns >= self.budget.max_model_turns:
                raise BudgetExceededError("agent model-turn budget exhausted")
            if self.remaining_wall_seconds() <= 0:
                raise BudgetExceededError("agent wall-clock budget exhausted")
            self.parent.reserve_model_turn()
            self.model_turns += 1

    def record_tokens(self, count: int) -> None:
        count = max(0, count)
        with self._lock:
            if self.tokens + count > self.budget.max_total_tokens:
                raise BudgetExceededError("agent token budget exhausted")
            self.parent.record_tokens(count)
            self.tokens += count

    def record_output_bytes(self, count: int) -> None:
        count = max(0, count)
        with self._lock:
            if self.output_bytes + count > self.budget.max_output_bytes:
                raise BudgetExceededError("agent output budget exhausted")
            self.parent.record_output_bytes(count)
            self.output_bytes += count

    def snapshot(self) -> dict[str, int | float]:
        return {
            "tool_calls": self.tool_calls,
            "model_turns": self.model_turns,
            "tokens": self.tokens,
            "output_bytes": self.output_bytes,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
        }

    def lease(self, budget: RunBudget) -> "BudgetLease":
        return BudgetLease(self, budget)
