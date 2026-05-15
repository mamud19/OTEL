# Cumulative token usage tracker for GenAI tracing sessions.

import threading
import time
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TokenStats:
    # Accumulate token usage for a tracing session.
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_requests: int = 0
    model_breakdown: Dict[str, int] = field(default_factory=dict)
    session_start_time: float = field(default_factory=time.monotonic)
    average_tokens_per_request: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "model_breakdown": self.model_breakdown,
            "average_tokens_per_request": self.average_tokens_per_request,
        }

# Token usage accumulator.
class TokenTracker:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._requests: int = 0
        self._model_breakdown: Dict[str, int] = {}
        self._session_start = time.monotonic()

    # Mutation methods
    def add(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "unknown",
    ) -> None:
        # Record token usage from a single LLM call.
        with self._lock:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._requests += 1
            self._model_breakdown[model] = (
                self._model_breakdown.get(model, 0) + prompt_tokens + completion_tokens
            )

    def reset(self) -> None:
        # Reset all counters to zero and restart the session timer.
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._requests = 0
            self._model_breakdown = {}
            self._session_start = time.monotonic()

    # Query methods
    def get_stats(self) -> TokenStats:
        # Return an immutable snapshot of current token usage.
        with self._lock:
            total = self._prompt_tokens + self._completion_tokens
            avg = total / self._requests if self._requests > 0 else 0.0
            return TokenStats(
                total_prompt_tokens=self._prompt_tokens,
                total_completion_tokens=self._completion_tokens,
                total_tokens=total,
                total_requests=self._requests,
                model_breakdown=dict(self._model_breakdown),
                session_start_time=self._session_start,
                average_tokens_per_request=round(avg, 2),
            )

    def get_cost_estimate(
        self,
        cost_per_1k_input: float = 0.00015,
        cost_per_1k_output: float = 0.00060,
    ) -> float:
        # Estimate the USD cost based on token counts. Default prices reflect GPT-4o-mini pricing (as of May 2026).
        # Args: cost_per_1k_input: Cost in USD per 1,000 prompt tokens. cost_per_1k_output: Cost in USD per 1,000 completion tokens.
        # Returns: Estimated total cost in USD.
        with self._lock:
            input_cost = (self._prompt_tokens / 1000) * cost_per_1k_input
            output_cost = (self._completion_tokens / 1000) * cost_per_1k_output
            return round(input_cost + output_cost, 6)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"TokenTracker(total={stats.total_tokens}, "
            f"prompt={stats.total_prompt_tokens}, "
            f"completion={stats.total_completion_tokens}, "
            f"requests={stats.total_requests})"
        )
