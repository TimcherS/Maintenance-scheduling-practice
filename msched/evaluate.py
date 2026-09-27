"""Оценка расписания: единые критерии для всех методов.

Расписание задаётся вектором s: s[a] — момент начала останова a.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .instance import Instance


@dataclass
class Metrics:
    objective: float  # значение целевой функции
    sum_dev: int  # Σ отклонений от плановой наработки, сут
    max_dev: int  # максимальное отклонение, сут
    conflicts: int  # число пар пересекающихся остановов соседей
    peak: int  # пиковое число одновременных остановов
    load_std: float  # СКО загрузки бригад по суткам (равномерность)
    overload: int  # Σ превышений числа бригад R, бригадо-сутки
    feasible: bool  # все ограничения выполнены

    def as_dict(self) -> dict:
        return asdict(self)


def runs_from_starts(inst: Instance, s) -> np.ndarray:
    """Фактическая наработка перед каждым остановом."""
    s = np.asarray(s, dtype=np.int64)
    L = s.copy()
    m = inst.prev >= 0
    p = inst.prev[m]
    L[m] = s[m] - s[p] - inst.dur[p]
    return L


def starts_from_runs(inst: Instance, L) -> np.ndarray:
    """Обратное преобразование: наработки -> моменты начала остановов."""
    inc = np.asarray(L, dtype=np.int64).copy()
    m = inst.prev >= 0
    inc[m] += inst.dur[inst.prev[m]]
    cs = np.cumsum(inc)
    # Индекс первого останова своего станка (остановы станка идут подряд).
    first = np.maximum.accumulate(np.where(inst.prev < 0, np.arange(inst.n), 0))
    return cs - (cs[first] - inc[first])


def load_profile(inst: Instance, s) -> np.ndarray:
    """Число одновременно идущих остановов в каждые сутки."""
    s = np.asarray(s, dtype=np.int64)
    diff = np.zeros(inst.T + 1, dtype=np.int64)
    np.add.at(diff, s, 1)
    np.add.at(diff, s + inst.dur, -1)
    return np.cumsum(diff)[: inst.T]


def conflict_mask(inst: Instance, s) -> np.ndarray:
    """Для каждой пары соседних остановов: пересекаются ли они."""
    s = np.asarray(s, dtype=np.int64)
    sa, sb = s[inst.pa], s[inst.pb]
    return (sa < sb + inst.span[inst.pb]) & (sb < sa + inst.span[inst.pa])


def objective(inst: Instance, s) -> float:
    """Быстрый расчёт только целевой функции (для эвристик и ГА)."""
    s = np.asarray(s, dtype=np.int64)
    w = inst.weights
    dev = np.abs(runs_from_starts(inst, s) - inst.run).sum()
    conf = conflict_mask(inst, s).sum()
    load = load_profile(inst, s)
    over = np.maximum(load - inst.capacity, 0).sum()
    return float(w["dev"] * dev + w["conf"] * conf + w["peak"] * load.max() + w["over"] * over)


def evaluate(inst: Instance, s) -> Metrics:
    s = np.asarray(s, dtype=np.int64)
    L = runs_from_starts(inst, s)
    dev = np.abs(L - inst.run)
    load = load_profile(inst, s)
    over = int(np.maximum(load - inst.capacity, 0).sum())
    in_window = bool(((L >= inst.lo) & (L <= inst.hi)).all())
    return Metrics(
        objective=objective(inst, s),
        sum_dev=int(dev.sum()),
        max_dev=int(dev.max()) if inst.n else 0,
        conflicts=int(conflict_mask(inst, s).sum()),
        peak=int(load.max()),
        load_std=float(load.std()),
        overload=over,
        feasible=in_window and over == 0,
    )
