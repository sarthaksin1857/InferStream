import os
import time
import logging
from typing import Dict, Iterable

from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    InMemoryMetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from prometheus_client import start_http_server
except ImportError:
    PrometheusMetricReader = None

logger = logging.getLogger(__name__)

class MetricRegistry:
    """
    A backward-compatible metric registry wrapping the OpenTelemetry SDK.
    """

    def __init__(self) -> None:
        self._init_provider()

    def _init_provider(self) -> None:
        """Initializes the OTEL MeterProvider and internal state."""
        # State for our wrappers
        self.timers: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}

        self._counters: Dict[str, otel_metrics.Counter] = {}
        self._histograms: Dict[str, otel_metrics.Histogram] = {}
        self._observable_gauges: Dict[str, otel_metrics.ObservableGauge] = {}

        # Always add an InMemoryMetricReader so print_summary() and tests work
        self.reader = InMemoryMetricReader()
        readers = [self.reader]

        # Read exporter from env (default to inmemory for tests/local dev)
        exporter_type = os.environ.get("INFERSTREAM_METRICS_EXPORTER", "inmemory").lower()

        if exporter_type == "console":
            readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
        elif exporter_type == "otlp":
            readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
        elif exporter_type == "prometheus":
            if PrometheusMetricReader is None:
                logger.warning("Prometheus exporter requested but packages are missing. Install opentelemetry-exporter-prometheus")
            else:
                port = int(os.environ.get("PROMETHEUS_PORT", 9090))
                try:
                    start_http_server(port=port)
                    logger.info(f"Started Prometheus metrics server on port {port}")
                    readers.append(PrometheusMetricReader())
                except Exception as e:
                    logger.error(f"Failed to start Prometheus server on port {port}: {e}")

        self.provider = MeterProvider(metric_readers=readers)
        self.meter = self.provider.get_meter("inferstream")

    def _get_counter(self, name: str) -> otel_metrics.Counter:
        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(name=name)
        return self._counters[name]

    def _get_histogram(self, name: str) -> otel_metrics.Histogram:
        if name not in self._histograms:
            self._histograms[name] = self.meter.create_histogram(name=name)
        return self._histograms[name]

    def inc(self, name: str, value: float = 1.0) -> None:
        """Increment a counter."""
        self._get_counter(name).add(value)

    def observe(self, name: str, value: float) -> None:
        """Observe a value for a histogram."""
        self._get_histogram(name).record(value)

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to a specific value."""
        self.gauges[name] = value

        if name not in self._observable_gauges:
            def callback(options, metric_name=name) -> Iterable[otel_metrics.Observation]:
                # In OTEL, observable gauges call a callback to retrieve the value
                yield otel_metrics.Observation(self.gauges[metric_name])

            self._observable_gauges[name] = self.meter.create_observable_gauge(
                name=name, callbacks=[callback]
            )

    def start_timer(self, name: str) -> None:
        """Start a timer."""
        self.timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Stop a timer and record its duration in a histogram."""
        if name in self.timers:
            duration = time.perf_counter() - self.timers.pop(name)
            self.observe(name, duration)
            return duration
        return 0.0

    def print_summary(self) -> None:
        """Print a summary of all metrics, parsed from the InMemoryMetricReader."""
        # Force a collection to ensure gauge callbacks are fired
        metrics_data = self.reader.get_metrics_data()

        print("\n" + "=" * 40)
        print(f"{'Metrics Summary (OpenTelemetry)':^40}")
        print("=" * 40)

        if not metrics_data or not metrics_data.resource_metrics:
            print("  No metrics recorded.")
            print("=" * 40 + "\n")
            return

        counters = {}
        histograms = {}
        gauges = {}

        # Parse the nested OTEL metrics structure
        for rm in metrics_data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    name = metric.name
                    data = metric.data

                    # Pluck values out of the OTEL DataPoints
                    if hasattr(data, "data_points") and data.data_points:
                        dp = data.data_points[0]
                        # Check the actual OTEL class type to route
                        dt_name = type(data).__name__
                        if dt_name == "Sum":
                            counters[name] = dp.value
                        elif dt_name == "Gauge":
                            gauges[name] = dp.value
                        elif dt_name == "Histogram":
                            histograms[name] = {
                                "count": dp.count,
                                "sum": dp.sum,
                                "min": dp.min if hasattr(dp, "min") else 0,
                                "max": dp.max if hasattr(dp, "max") else 0,
                            }

        if counters:
            print("\nCounters:")
            print("-" * 40)
            for name, value in sorted(counters.items()):
                print(f"  {name:<25} : {value:.2f}")

        if histograms:
            print("\nHistograms / Observations:")
            print("-" * 40)
            for name, stats in sorted(histograms.items()):
                count = stats["count"]
                if count > 0:
                    avg = stats["sum"] / count
                    print(f"  {name}:")
                    print(f"    count = {count}")
                    print(f"    avg   = {avg:.4f}")
                    print(f"    min   = {stats.get('min', 0):.4f}")
                    print(f"    max   = {stats.get('max', 0):.4f}")

        if gauges:
            print("\nGauges (Current Values):")
            print("-" * 40)
            for name, value in sorted(gauges.items()):
                print(f"  {name:<25} : {value:.2f}")

        print("=" * 40 + "\n")

    def reset(self) -> None:
        """Clear all metrics by recreating the OTEL provider."""
        if hasattr(self, 'provider'):
            self.provider.shutdown()
        self._init_provider()
