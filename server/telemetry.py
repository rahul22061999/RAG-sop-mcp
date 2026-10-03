import asyncio
import os
import time

from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import Histogram, MeterProvider
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

LATENCY_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60, 120]

_configured = False


def setup_telemetry(service_name: str) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    resource = Resource.create({"service.name": service_name})

    tracer_provider = TracerProvider(resource=resource)
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    views = [
        View(
            instrument_name="rag.*",
            instrument_type=Histogram,
            aggregation=ExplicitBucketHistogramAggregation(LATENCY_BUCKETS),
        )
    ]
    metrics.set_meter_provider(
        MeterProvider(
            resource=resource,
            metric_readers=[PrometheusMetricReader()],
            views=views,
        )
    )


_tracer = trace.get_tracer("wms_sop")
_meter = metrics.get_meter("wms_sop")

_request_hist = _meter.create_histogram(
    "rag.request.duration", unit="s", description="End to end RAG request time"
)
_retrieve_hist = _meter.create_histogram(
    "rag.retrieve.duration", unit="s", description="Hybrid retrieval time"
)
_generate_hist = _meter.create_histogram(
    "rag.generate.duration", unit="s", description="LLM generation time"
)
_ttft_hist = _meter.create_histogram(
    "rag.time_to_first_token", unit="s", description="Request start to first token"
)
_retries = _meter.create_counter(
    "rag.retries", description="Retried transient failures (backoff with jitter)"
)
_STAGE_HISTS = {"retrieve": _retrieve_hist, "generate": _generate_hist}


class _Stage:

    def __init__(self, rt: "RagTrace", name: str) -> None:
        self._rt, self._name = rt, name

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.span = _tracer.start_span(
            f"rag.{self._name}", context=self._rt.context
        )
        return self.span

    def __exit__(self, exc_type, exc, tb):
        if exc is not None and not isinstance(exc, (GeneratorExit, asyncio.CancelledError)):
            self.span.record_exception(exc)
            self.span.set_status(trace.StatusCode.ERROR)
        self.span.end()
        _STAGE_HISTS[self._name].record(
            time.perf_counter() - self._t0, {"rag.mode": self._rt.mode}
        )
        return False


class RagTrace:

    def __init__(self, mode: str, query: str, top_k: int) -> None:
        self.mode = mode
        self.outcome = "ok"
        self.cache = "miss"
        self.tokens = 0
        self._t0 = time.perf_counter()
        self._first_token_seen = False
        self.root = _tracer.start_span(
            "rag.request",
            attributes={
                "rag.mode": mode,
                "rag.query_length": len(query),
                "rag.top_k": top_k,
            },
        )
        self.context = trace.set_span_in_context(self.root)

    def stage(self, name: str) -> _Stage:
        return _Stage(self, name)

    def token(self) -> None:
        self.tokens += 1
        if not self._first_token_seen:
            self._first_token_seen = True
            _ttft_hist.record(
                time.perf_counter() - self._t0,
                {"rag.mode": self.mode, "rag.cache": self.cache},
            )

    def retried(self, stage: str, attempt: int, delay: float, exc: BaseException) -> None:
        _retries.add(1, {"rag.mode": self.mode, "rag.stage": stage})
        self.root.add_event(
            "retry",
            {
                "rag.stage": stage,
                "retry.attempt": attempt,
                "retry.delay_s": round(delay, 3),
                "exception.type": type(exc).__name__,
            },
        )

    def fail(self, exc: BaseException) -> None:
        self.outcome = "error"
        self.root.record_exception(exc)
        self.root.set_status(trace.StatusCode.ERROR)

    def end(self) -> None:
        self.root.set_attribute("rag.outcome", self.outcome)
        self.root.set_attribute("rag.cache", self.cache)
        self.root.set_attribute("rag.tokens", self.tokens)
        self.root.end()
        _request_hist.record(
            time.perf_counter() - self._t0,
            {"rag.mode": self.mode, "rag.outcome": self.outcome, "rag.cache": self.cache},
        )
