# GenAI OTEL Collector — primary public interface.

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from .confidence_scorer import ConfidenceScorer, ScoringResult
from .exporters.azure_exporter import create_tracer_provider, shutdown_tracer_provider
from .pii_scrubber import PIIScrubber, PIIResult
from .spans import GenAISpanAttributes
from .token_tracker import TokenStats, TokenTracker

logger = logging.getLogger(__name__)


# Configuration 
@dataclass
class CollectorConfig:
    # Configuration for GenAICollector
    app_insights_connection_string: str
    default_model: str = "unknown"
    service_name: str = "genai-otel-collector"
    service_version: str = "1.0.0"
    capture_raw_prompts: bool = False
    enable_console_export: bool = False
    max_content_length: int = 4000

# Result

@dataclass
class CompletionResult:
    # Structured result returned by GenAICollector.record_completion.
    content: str
    scrubbed_content: str
    usage_prompt_tokens: int
    usage_completion_tokens: int
    usage_total_tokens: int
    estimated_cost: float
    confidence_score: float
    UNCERTAINTY_RISK: float
    UNCERTAINTY_RISK_level: str
    uncertainty_markers: List[str]
    finish_reason: str
    response_id: str
    response_time_ms: float
    pii_detected: bool
    pii_scrubbed_count: int
    pii_types: List[str] = field(default_factory=list)


# Collector
class GenAICollector:
    # Collects logs, traces, and metrics for Azure Application Insights
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self._pii = PIIScrubber()
        self._scorer = ConfidenceScorer()
        self._tokens = TokenTracker()

        self._provider = create_tracer_provider(
            service_name=config.service_name,
            service_version=config.service_version,
            connection_string=config.app_insights_connection_string,
            enable_console_export=config.enable_console_export,
        )
        self._tracer = trace.get_tracer(
            config.service_name, config.service_version
        )

    # Private helpers
    def _truncate(self, text: str) -> str:
        limit = self.config.max_content_length
        if len(text) > limit:
            return text[:limit] + "…[TRUNCATED]"
        return text

    @staticmethod
    def _extract_user_message(messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return messages[-1].get("content", "") if messages else ""

    def _set_optional(self, span, attr: str, value) -> None:
        if value is not None:
            span.set_attribute(attr, value)

    # Public API 
    def record_completion(
        self,
        messages: List[Dict[str, str]],
        completion_text: str,
        model: Optional[str] = None,
        provider_name: str = "unknown",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        response_id: Optional[str] = None,
        finish_reason: Optional[str] = None,
        response_time_ms: float = 0.0,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> CompletionResult:
       # Record a chat-completion execution and emit a fully-attributed OTEL span.
        effective_model = model or self.config.default_model

        with self._tracer.start_as_current_span(
            name="gen_ai.completion", kind=SpanKind.CLIENT
        ) as span:
            try:
                # Request attributes 
                span.set_attribute(GenAISpanAttributes.GEN_AI_SYSTEM, provider_name)
                span.set_attribute(GenAISpanAttributes.GEN_AI_REQUEST_MODEL, effective_model)
                span.set_attribute(GenAISpanAttributes.GEN_AI_OPERATION_NAME, "chat.completions")
                self._set_optional(span, GenAISpanAttributes.GEN_AI_SESSION_ID, session_id)
                self._set_optional(span, GenAISpanAttributes.GEN_AI_USER_ID, user_id)

                # PII-scrub the input
                user_text = self._extract_user_message(messages)
                scrubbed_input, input_pii = self._pii.scrub(user_text)

                prompt_for_span = (
                    user_text if self.config.capture_raw_prompts else scrubbed_input
                )
                span.set_attribute(
                    GenAISpanAttributes.GEN_AI_PROMPT,
                    self._truncate(prompt_for_span),
                )
                span.set_attribute(GenAISpanAttributes.AI_PII_DETECTED, input_pii.detected)
                span.set_attribute(GenAISpanAttributes.AI_PII_SCRUBBED_COUNT, input_pii.count)
                if input_pii.types_detected:
                    span.set_attribute(
                        GenAISpanAttributes.AI_PII_TYPES_DETECTED,
                        ",".join(input_pii.types_detected),
                    )

                # Scrub output PII 
                scrubbed_output, output_pii = self._pii.scrub(completion_text)

                # Score confidence / uncertainty risk 
                scoring: ScoringResult = self._scorer.score(completion_text)

                # Calculate Estimated Cost
                # Default rates based on GPT-4o-mini pricing
                cost_per_1k_input = 0.00015
                cost_per_1k_output = 0.00060
                call_cost = round(((prompt_tokens / 1000.0) * cost_per_1k_input) + ((completion_tokens / 1000.0) * cost_per_1k_output), 6)

                # Accumulate token usage
                total_tokens = prompt_tokens + completion_tokens
                self._tokens.add(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=effective_model,
                )

                # Response span attributes 
                span.set_attribute(
                    GenAISpanAttributes.GEN_AI_COMPLETION,
                    self._truncate(scrubbed_output),
                )
                span.set_attribute(GenAISpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS, prompt_tokens)
                span.set_attribute(GenAISpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS, completion_tokens)
                span.set_attribute(GenAISpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, total_tokens)
                span.set_attribute(GenAISpanAttributes.GEN_AI_USAGE_COST, call_cost)
                self._set_optional(span, GenAISpanAttributes.GEN_AI_RESPONSE_ID, response_id)
                span.set_attribute(GenAISpanAttributes.GEN_AI_RESPONSE_MODEL, effective_model)
                self._set_optional(span, GenAISpanAttributes.GEN_AI_RESPONSE_FINISH_REASON, finish_reason)
                span.set_attribute(GenAISpanAttributes.GEN_AI_RESPONSE_TIME_MS, response_time_ms)
                
                span.set_attribute(GenAISpanAttributes.AI_CONFIDENCE_SCORE, scoring.confidence_score)
                span.set_attribute(GenAISpanAttributes.AI_UNCERTAINTY_RISK, scoring.UNCERTAINTY_RISK)
                span.set_attribute(GenAISpanAttributes.AI_UNCERTAINTY_RISK_LEVEL, scoring.risk_level)
                span.set_attribute(GenAISpanAttributes.AI_UNCERTAINTY_MARKER_COUNT, scoring.uncertainty_count)
                if scoring.uncertainty_markers:
                    span.set_attribute(
                        GenAISpanAttributes.AI_UNCERTAINTY_MARKERS,
                        ",".join(scoring.uncertainty_markers),
                    )

                span.set_status(StatusCode.OK)

                combined_pii_types = list(
                    dict.fromkeys(input_pii.types_detected + output_pii.types_detected)
                )
                return CompletionResult(
                    content=completion_text,
                    scrubbed_content=scrubbed_output,
                    usage_prompt_tokens=prompt_tokens,
                    usage_completion_tokens=completion_tokens,
                    usage_total_tokens=total_tokens,
                    estimated_cost=call_cost,
                    confidence_score=scoring.confidence_score,
                    UNCERTAINTY_RISK=scoring.UNCERTAINTY_RISK,
                    UNCERTAINTY_RISK_level=scoring.risk_level,
                    uncertainty_markers=scoring.uncertainty_markers,
                    finish_reason=finish_reason or "unknown",
                    response_id=response_id or "unknown",
                    response_time_ms=response_time_ms,
                    pii_detected=input_pii.detected or output_pii.detected,
                    pii_scrubbed_count=input_pii.count + output_pii.count,
                    pii_types=combined_pii_types,
                )

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise

    def get_token_stats(self) -> TokenStats:
        # Return cumulative token-usage statistics for this collector instance.
        return self._tokens.get_stats()

    def flush(self) -> None:
        # Force-flush all buffered spans to Azure Application Insights.
        self._provider.force_flush(timeout_millis=10_000)

    def shutdown(self) -> None:
        # Flush pending spans and release all resources.
        shutdown_tracer_provider(self._provider)

    # Context-manager support
    def __enter__(self) -> "GenAICollector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
