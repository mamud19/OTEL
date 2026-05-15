# Azure Application Insights OTEL exporter setup to Azure 

import logging
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

# create_tracer_provider to setup tracer provider and export spans to Azure Monitor
def create_tracer_provider(
    service_name: str,
    service_version: str,
    connection_string: str,
    enable_console_export: bool = False,
) -> TracerProvider:
    # This function sets up the OpenTelemetry TracerProvider and exports spans to Azure Monitor.
    if not connection_string:
        raise ValueError("connection_string must not be empty")

    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "service.namespace": "genai",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
        }
    )

    provider = TracerProvider(resource=resource)

    # Register Azure Monitor exporter
    azure_exporter = AzureMonitorTraceExporter(connection_string=connection_string)
    provider.add_span_processor(BatchSpanProcessor(azure_exporter))
    logger.info("Azure Monitor trace exporter registered for service '%s'", service_name)

    if enable_console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.debug("Console span exporter enabled")

    trace.set_tracer_provider(provider)
    return provider

# shutdown_tracer_provider to flush all pending spans and shut the provider down gracefully
def shutdown_tracer_provider(provider: TracerProvider) -> None:
    # This function flushes all pending spans and shuts the provider down gracefully.
    try:
        provider.force_flush(timeout_millis=10_000)
    except Exception as exc:
        logger.warning("force_flush raised: %s", exc)
    finally:
        provider.shutdown()
