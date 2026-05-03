from .registry import MetricRegistry

# Provide a default global registry for easy access
metrics = MetricRegistry()

__all__ = ["metrics", "MetricRegistry"]
