"""Результат работы метода и замер времени."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


class Tracker:
    """Засекает время с начала работы метода и запоминает улучшения ЦФ."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.history: list[tuple[float, float]] = []

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def record(self, value: float) -> None:
        if not self.history or value < self.history[-1][1] - 1e-9:
            self.history.append((self.elapsed(), float(value)))


@dataclass
class SolveResult:
    method: str
    starts: np.ndarray | None
    status: str  # optimal | feasible | heuristic | infeasible | unknown
    time_total: float
    history: list  # (время, значение ЦФ) при каждом улучшении
    bound: float | None = None  # нижняя оценка (для точных методов)
    info: dict = field(default_factory=dict)

    @property
    def t_first(self) -> float | None:
        """Время до первого допустимого решения."""
        return self.history[0][0] if self.history else None

    @property
    def t_best(self) -> float | None:
        """Время до лучшего найденного решения."""
        return self.history[-1][0] if self.history else None

    @property
    def t_opt(self) -> float | None:
        """Время до доказательства оптимальности."""
        return self.time_total if self.status == "optimal" else None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "status": self.status,
            "starts": None if self.starts is None else [int(x) for x in self.starts],
            "time_total": self.time_total,
            "t_first": self.t_first,
            "t_best": self.t_best,
            "t_opt": self.t_opt,
            "bound": self.bound,
            "history": self.history,
            "info": self.info,
        }
