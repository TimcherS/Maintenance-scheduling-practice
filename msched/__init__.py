"""Планирование остановов станков на обслуживание.

Методы: MILP (HiGHS), CP-SAT (OR-Tools), жадная эвристика H1,
локальный поиск H2, генетический алгоритм.
"""
from .evaluate import Metrics, evaluate, objective
from .instance import SIZES, Instance, generate
from .result import SolveResult
from .solvers import METHODS, solve

__all__ = ["Instance", "generate", "SIZES", "evaluate", "objective", "Metrics",
           "solve", "METHODS", "SolveResult"]
