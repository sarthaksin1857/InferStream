"""Coordinator module for InferStream.

This module acts as the brain of the distributed system, handling the in-memory queue,
assigning request IDs, and routing requests and responses to and from workers via gRPC.
"""

from inferstream.coordinator.server import CoordinatorServiceServicer, serve

__all__ = ["CoordinatorServiceServicer", "serve"]
