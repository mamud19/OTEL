# genai-otel-collector - OpenTelemetry instrumentation for GenAI / LLM applications.

from .collector import CollectorConfig, CompletionResult, GenAICollector
from .confidence_scorer import ConfidenceScorer, ScoringResult
from .pii_scrubber import PIIResult, PIIScrubber
from .spans import GenAISpanAttributes
from .token_tracker import TokenStats, TokenTracker

__version__ = "1.0.0"
__author__ = "Mamud Alkali"

__all__ = [
    # Main interface
    "GenAICollector",
    "CollectorConfig",
    "CompletionResult",
    # Sub-components (also usable standalone)
    "PIIScrubber",
    "PIIResult",
    "ConfidenceScorer",
    "ScoringResult",
    "TokenTracker",
    "TokenStats",
    # Span attribute constants
    "GenAISpanAttributes",
    # Metadata
    "__version__",
]
