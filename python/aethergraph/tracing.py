"""OpenTelemetry tracing configuration for AetherGraph.

This module provides optional distributed tracing integration using OpenTelemetry.
When configured, it captures spans for sampling operations, enabling visibility
into the data loading pipeline in production.

Example:
    >>> from aethergraph.tracing import configure_tracing, get_tracer
    >>>
    >>> # Configure OTEL exporter (e.g., to Jaeger)
    >>> configure_tracing(endpoint="localhost:4317")
    >>>
    >>> # Tracing is now active for NeighborLoader and Ray datasource

Requires:
    pip install aethergraph[otel]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

__all__ = ["TracingConfig", "configure_tracing", "get_tracer"]

_TRACER_NAME = "aethergraph"
_tracer: Tracer | None = None
_configured: bool = False


class TracingConfig:
    """Configuration for OpenTelemetry tracing.

    Attributes:
        endpoint: OTLP exporter endpoint (e.g., "localhost:4317").
        service_name: Service name for traces. Defaults to "aethergraph".
        batch_delay_ms: Maximum delay before exporting spans in milliseconds.
        max_queue_size: Maximum number of spans to queue before dropping.
    """

    def __init__(
        self,
        endpoint: str = "localhost:4317",
        service_name: str = "aethergraph",
        batch_delay_ms: int = 5000,
        max_queue_size: int = 2048,
    ) -> None:
        """Initialize tracing configuration.

        Args:
            endpoint: OTLP exporter endpoint (e.g., "localhost:4317").
            service_name: Service name for traces. Defaults to "aethergraph".
            batch_delay_ms: Maximum delay before exporting spans in milliseconds.
            max_queue_size: Maximum number of spans to queue before dropping.
        """
        self.endpoint = endpoint
        self.service_name = service_name
        self.batch_delay_ms = batch_delay_ms
        self.max_queue_size = max_queue_size


def configure_tracing(
    endpoint: str = "localhost:4317",
    service_name: str = "aethergraph",
    config: TracingConfig | None = None,
) -> None:
    """Configure OpenTelemetry tracing with OTLP exporter.

    Sets up the global tracer provider with a BatchSpanProcessor that exports
    to the specified OTLP endpoint. Call this once at application startup
    before creating any NeighborLoader instances.

    Args:
        endpoint: OTLP gRPC endpoint (e.g., "localhost:4317" for Jaeger).
            Ignored if config is provided.
        service_name: Service name to identify traces. Defaults to "aethergraph".
            Ignored if config is provided.
        config: Optional TracingConfig for advanced configuration.

    Raises:
        ImportError: If opentelemetry packages are not installed.

    Example:
        >>> # Basic usage with Jaeger
        >>> configure_tracing(endpoint="localhost:4317")
        >>>
        >>> # With custom service name
        >>> configure_tracing(endpoint="otel-collector:4317", service_name="my-gnn-app")
        >>>
        >>> # Using config object
        >>> config = TracingConfig(
        ...     endpoint="localhost:4317",
        ...     batch_delay_ms=1000,
        ...     max_queue_size=4096,
        ... )
        >>> configure_tracing(config=config)
    """
    global _tracer, _configured

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        raise ImportError(
            "OpenTelemetry integration requires otel extras. "
            "Install with: pip install aethergraph[otel]"
        ) from e

    if config is None:
        config = TracingConfig(endpoint=endpoint, service_name=service_name)

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=config.endpoint, insecure=True)
    processor = BatchSpanProcessor(
        exporter,
        max_queue_size=config.max_queue_size,
        schedule_delay_millis=config.batch_delay_ms,
    )
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(_TRACER_NAME)
    _configured = True


def get_tracer() -> Tracer | None:
    """Get the configured tracer instance.

    Returns the tracer only when :func:`configure_tracing` has been called;
    otherwise returns ``None``. Merely having ``opentelemetry-api`` installed
    is not enough — without an explicit configure call there is no exporter,
    so spans would be created and silently dropped. Modules check for ``None``
    to skip span creation entirely when tracing is not configured.

    Returns:
        The OpenTelemetry Tracer instance, or None if not configured.

    Example:
        >>> tracer = get_tracer()
        >>> if tracer:
        ...     with tracer.start_as_current_span("my_operation"):
        ...         do_work()
    """
    if not _configured:
        return None
    return _tracer
