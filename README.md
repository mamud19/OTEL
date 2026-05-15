# GenAI OpenTelemetry Collector

An LLM-agnostic, passive-observer OpenTelemetry collector for GenAI applications.

This package intercepts, analyzes, and exports telemetry data from large language model interactions. It records completions, tracks token usage, calculates cost estimates, scrubs personally identifiable information (PII), and evaluates confidence scores, seamlessly forwarding all telemetry to Azure Application Insights via OpenTelemetry.

## Features

- **LLM-Agnostic Design**: Works with any LLM provider (Azure OpenAI, standard OpenAI, Anthropic, Gemini, etc.) as a passive observer.
- **PII Scrubbing**: Automatically detects and redacts Personally Identifiable Information from prompts and completions before exporting telemetry.
- **Confidence Scoring**: Evaluates the model's response and assigns a confidence and risk level score.
- **Cost & Token Tracking**: Tracks both prompt and completion tokens, computing estimated costs dynamically.
- **Azure Application Insights Integration**: Natively exports distributed traces and spans directly to Azure Application Insights.

## Installation

Ensure you have Python 3.9+ installed. 

You can install the package using the pre-built wheel file located in the `package` directory:

```bash
pip install ./package/genai_otel_collector-1.0.0-py3-none-any.whl
```

Alternatively, you can install the package locally in editable mode for development by navigating to the project directory and running:

```bash
pip install -e ./src
```

Dependencies include `opentelemetry-sdk`, `opentelemetry-api`, `azure-monitor-opentelemetry-exporter`, and `openai`.

## Quick Start

Here is a simple example demonstrating how to use the `GenAICollector` alongside the Azure OpenAI client:

```python
import os
import uuid
import time
from opentelemetry import trace
from openai import AzureOpenAI
from src.collector import CollectorConfig, GenAICollector

# Configure Azure Application Insights and Collector
APP_INSIGHTS_KEY = os.environ.get("APP_INSIGHTS_KEY", "InstrumentationKey=your-instrumentation-key;...")
config = CollectorConfig(
    app_insights_connection_string=APP_INSIGHTS_KEY,
    enable_console_export=True,
    capture_raw_prompts=True
)

client = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_ENDPOINT", "https://your-endpoint.openai.azure.com/"),
    api_key=os.environ.get("AZURE_API_KEY", "your-api-key"),
    api_version="2023-05-15"
)

tracer = trace.get_tracer("chatbot_app")

with GenAICollector(config) as collector:
    session_id = str(uuid.uuid4())
    messages = [{"role": "user", "content": "Hello! How can you help?"}]

    # Start a tracing span for the conversation
    with tracer.start_as_current_span("conversation_group", attributes={"gen_ai.session.id": session_id}):
        start_time = time.monotonic()
        response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)
        
        # Record the results via the collector
        result = collector.record_completion(
            messages=messages,
            completion_text=response.choices[0].message.content,
            model="gpt-4.1-mini",
            provider_name="azure_openai",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            response_id=response.id,
            finish_reason=response.choices[0].finish_reason,
            response_time_ms=elapsed_ms,
            session_id=session_id
        )

        print(f"Bot Response: {result.content}")
        print(f"Tokens Used: {result.usage_total_tokens}")
        print(f"Estimated Cost: ${result.estimated_cost:.6f}")
        print(f"Confidence: {result.confidence_score:.2f}")
        if result.pii_detected:
            print(f"PII Scrubbed count: {result.pii_scrubbed_count}")
```

## Architecture

The system uses a component-based architecture for processing LLM completions:
- **Collector (`src.collector.GenAICollector`)**: The main orchestrator that coordinates analysis and creates OpenTelemetry spans.
- **PII Scrubber (`src.pii_scrubber`)**: Identifies and redacts sensitive information to keep traces compliant.
- **Confidence Scorer (`src.confidence_scorer`)**: Generates confidence scores for responses to flag potentially hallucinated or low-quality content.
- **Token Tracker (`src.token_tracker`)**: Computes cost estimates based on tokens used for different models.
- **Exporters (`src.exporters`)**: Handles pushing OTEL spans to sinks like Azure Application Insights and the console.

## Usage Example

For a complete interactive chatbot implementation using the collector, check out `sample_chatbot.py` in the root directory.
