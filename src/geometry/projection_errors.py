"""Shared failure type for projection construction and selection."""


class ProjectionFeasibilityError(RuntimeError):
    """The current solve or finite candidate pool requires feasibility review."""
