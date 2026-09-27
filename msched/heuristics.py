"""Эвристики.

H1 — жадный алгоритм: остановы рассматриваются в порядке плановых
моментов, каждый ставится в лучший момент с учётом уже поставленных.
H2 — итерированный локальный поиск, который улучшает готовое расписание
(например, от H1).
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from .evaluate import conflict_mask, objective, runs_from_starts
from .instance import Instance
from .result import Tracker


def greedy(inst: Instance) -> np.ndarray:
    """H1: жадная расстановка остановов в хронологическом порядке."""
    n = inst.n
    w = inst.weights
    R = int(inst.capacity)
    s = np.full(n, -1, dtype=np.int64)
    load = np.zeros(inst.T, dtype=np.int64)
    peak = 0

    # Очередь: (плановый момент начала, останов).
    heap = [(int(inst.run[ms[0]]), ms[0]) for ms in inst.by_machine if ms]
    heapq.heapify(heap)
    while heap:
        _, a = heapq.heappop(heap)
        p = int(inst.prev[a])
        base = 0 if p < 0 else int(s[p] + inst.dur[p])  # конец предыдущего останова
        d, span_a, run = int(inst.dur[a]), int(inst.span[a]), int(inst.run[a])

        best_key = best_st = None
        for L in range(int(inst.lo[a]), int(inst.hi[a]) + 1):
            st = base + L
            dev = abs(L - run)
            conf = sum(1 for b in inst.partners[a]
                       if s[b] >= 0 and st < s[b] + inst.span[b] and s[b] < st + span_a)
            seg = load[st:st + d]
            top = int(seg.max()) + 1
            over = int((seg >= R).sum())
            cost = (w["dev"] * dev + w["conf"] * conf
                    + w["peak"] * max(0, top - peak) + w["over"] * over)
            key = (cost, dev, st)
            if best_key is None or key < best_key:
                best_key, best_st = key, st

        s[a] = best_st
        load[best_st:best_st + d] += 1
        peak = max(peak, int(load[best_st:best_st + d].max()))
        nx = int(inst.next[a])
        if nx >= 0:
            heapq.heappush(heap, (best_st + d + int(inst.run[nx]), nx))
    return s


def _move_range(inst: Instance, L, a: int, move: str):
    """Допустимые сдвиги Δ и изменяемая часть расписания для хода.

    push  — сдвиг a и всех следующих остановов станка на Δ;
    shift — сдвиг только a на Δ (наработка следующего меняется на −Δ).
    """
    nx = int(inst.next[a])
    d_lo, d_hi = int(inst.lo[a] - L[a]), int(inst.hi[a] - L[a])
    if move == "shift":
        if nx < 0:
            return None  # у последнего останова shift совпадает с push
        d_lo = max(d_lo, int(L[nx] - inst.hi[nx]))
        d_hi = min(d_hi, int(L[nx] - inst.lo[nx]))
        return d_lo, d_hi, slice(a, a + 1)
    return d_lo, d_hi, slice(a, int(inst.last[a]) + 1)


def _apply(inst: Instance, s, L, a: int, move: str, part: slice, delta: int) -> None:
    s[part] += delta
    L[a] += delta
    if move == "shift":
        L[int(inst.next[a])] -= delta


def _descend(inst: Instance, s, t_end: float, rng, tracker: Tracker | None):
    """Спуск с первым улучшением, пока проход по остановам даёт улучшение."""
    L = runs_from_starts(inst, s)
    best = objective(inst, s)
    improved = True
    while improved and time.perf_counter() < t_end:
        improved = False
        for a in rng.permutation(inst.n).tolist():
            if time.perf_counter() >= t_end:
                break
            for move in ("push", "shift"):
                rng_part = _move_range(inst, L, a, move)
                if rng_part is None:
                    continue
                d_lo, d_hi, part = rng_part
                for delta in range(d_lo, d_hi + 1):
                    if delta == 0:
                        continue
                    s[part] += delta
                    val = objective(inst, s)
                    s[part] -= delta
                    if val < best - 1e-9:
                        _apply(inst, s, L, a, move, part, delta)
                        best = val
                        improved = True
                        if tracker:
                            tracker.record(best)
                        break
    return s, best


def _perturb(inst: Instance, s, rng, k_max: int = 3):
    """Возмущение: случайные ходы для 1..k_max остановов.

    Остановы из пар с конфликтом выбираются чаще.
    """
    s = s.copy()
    L = runs_from_starts(inst, s)
    mask = conflict_mask(inst, s)
    hot = np.unique(np.concatenate([inst.pa[mask], inst.pb[mask]]))
    for _ in range(int(rng.integers(1, k_max + 1))):
        if len(hot) and rng.random() < 0.7:
            a = int(rng.choice(hot))
        else:
            a = int(rng.integers(inst.n))
        move = "push" if rng.random() < 0.5 else "shift"
        rng_part = _move_range(inst, L, a, move)
        if rng_part is None or rng_part[0] > rng_part[1]:
            continue
        d_lo, d_hi, part = rng_part
        _apply(inst, s, L, a, move, part, int(rng.integers(d_lo, d_hi + 1)))
    return s


def local_search(inst: Instance, s0, time_limit: float = 10.0, seed: int = 0,
                 tracker: Tracker | None = None, max_stall: int = 300) -> np.ndarray:
    """H2: итерированный локальный поиск (ILS).

    1. Спуск из начального расписания до локального минимума.
    2. Возмущение лучшего расписания и новый спуск.
    3. Новое расписание принимается, если оно не хуже лучшего.
    Остановка: лимит времени или max_stall возмущений без улучшения.
    """
    rng = np.random.default_rng(seed)
    t_end = time.perf_counter() + time_limit
    best_s, best = _descend(inst, np.array(s0, dtype=np.int64), t_end, rng, tracker)
    stall = 0
    while stall < max_stall and time.perf_counter() < t_end:
        s, val = _descend(inst, _perturb(inst, best_s, rng), t_end, rng, tracker)
        stall = 0 if val < best - 1e-9 else stall + 1
        if val <= best:
            best_s, best = s, val
    return best_s
