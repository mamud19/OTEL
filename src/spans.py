# OpenTelemetry span attribute constants for GenAI semantic conventions.

class GenAISpanAttributes:
    # System 
    GEN_AI_SYSTEM = "gen_ai.system"

    # Request attributes 
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"

    # Input / Output content
    GEN_AI_PROMPT = "gen_ai.prompt"
    GEN_AI_COMPLETION = "gen_ai.completion"

    # Token usage 
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    GEN_AI_USAGE_COST = "gen_ai.usage.cost"

    # Response metadata 
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
    GEN_AI_RESPONSE_TIME_MS = "gen_ai.response.time_ms"

    # AI quality / confidence metrics
    AI_CONFIDENCE_SCORE = "ai.confidence.score"
    AI_UNCERTAINTY_RISK = "ai.uncertainty.risk"
    AI_UNCERTAINTY_RISK_LEVEL = "ai.uncertainty.risk_level"
    AI_UNCERTAINTY_MARKER_COUNT = "ai.uncertainty.marker_count"
    AI_UNCERTAINTY_MARKERS = "ai.uncertainty.markers"

    # PII metadata
    AI_PII_DETECTED = "ai.pii.detected"
    AI_PII_SCRUBBED_COUNT = "ai.pii.scrubbed_count"
    AI_PII_TYPES_DETECTED = "ai.pii.types_detected"

    # Session / user tracking
    GEN_AI_SESSION_ID = "gen_ai.session.id"
    GEN_AI_USER_ID = "gen_ai.user.id"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
