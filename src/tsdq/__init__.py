"""Configuration-driven time-series data quality experiments."""
from .config import ExperimentConfig, map_severity
from .core import InjectionNotFeasible
from .engine import DefectInjector
from .result import InjectionResult

__all__ = ["DefectInjector", "ExperimentConfig", "InjectionNotFeasible", "InjectionResult", "map_severity"]
