import time
from collections import defaultdict
from typing import Dict, List


class MetricRegistry:
    """
    A generic metric registry to track counters, histograms, and timers.
    """

    def __init__(self) -> None:
        self.counters: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        """Increment a counter."""
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        """Observe a value for a histogram."""
        self.histograms[name].append(value)

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to a specific value."""
        self.gauges[name] = value

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
        """Print a summary of all metrics."""
        print("\n" + "=" * 40)
        print(f"{'Metrics Summary':^40}")
        print("=" * 40)

        if self.counters:
            print("\nCounters:")
            print("-" * 40)
            for name, value in sorted(self.counters.items()):
                print(f"  {name:<25} : {value:.2f}")

        if self.histograms:
            print("\nHistograms / Observations:")
            print("-" * 40)
            for name, values in sorted(self.histograms.items()):
                if not values:
                    continue
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                print(f"  {name}:")
                print(f"    count = {len(values)}")
                print(f"    avg   = {avg:.4f}")
                print(f"    min   = {min_val:.4f}")
                print(f"    max   = {max_val:.4f}")

        if self.gauges:
            print("\nGauges (Current Values):")
            print("-" * 40)
            for name, value in sorted(self.gauges.items()):
                print(f"  {name:<25} : {value:.2f}")

        print("=" * 40 + "\n")

    def reset(self) -> None:
        """Clear all metrics."""
        self.counters.clear()
        self.histograms.clear()
        self.timers.clear()
        self.gauges.clear()
