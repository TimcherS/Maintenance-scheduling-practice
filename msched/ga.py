"""Генетический алгоритм.

Хромосома — вектор наработок L[a] ∈ [lo[a], hi[a]] перед каждым остановом.
Такое кодирование всегда даёт расписание, где наработки в допустимых окнах.
Пересечения соседей и загрузка бригад учитываются через целевую функцию.
"""
from __future__ import annotations

import time

import numpy as np

from .evaluate import objective, runs_from_starts, starts_from_runs
from .instance import Instance
from .result import Tracker


def genetic(inst: Instance, time_limit: float = 30.0, *, pop_size: int = 60,
            max_gens: int = 100_000, stall_gens: int = 300, p_cx: float = 0.9,
            p_mut: float | None = None, elite: int = 2, tour: int = 3, seed: int = 0,
            init=None, tracker: Tracker | None = None):
    """Возвращает (расписание, статистика).

    Операторы:
      селекция  — турнир размера tour;
      кроссовер — поблочный равномерный: гены станка берутся целиком
                  от одного из родителей;
      мутация   — сдвиг наработки на ±1..3 сут; с вероятностью 1/2
                  наработка следующего останова меняется на обратную
                  величину (сдвиг одного останова);
      элитизм   — elite лучших особей переходят без изменений.
    Остановка: лимит времени, max_gens поколений или stall_gens поколений
    без улучшения.
    """
    rng = np.random.default_rng(seed)
    n = inst.n
    lo, hi, nxt = inst.lo, inst.hi, inst.next
    p_mut = p_mut if p_mut is not None else min(1.0, 1.5 / max(n, 1))
    blocks = [np.array(ms) for ms in inst.by_machine if ms]
    t_end = time.perf_counter() + time_limit

    def fitness(L):
        return objective(inst, starts_from_runs(inst, L))

    def tournament(fits):
        idx = rng.integers(pop_size, size=tour)
        return idx[np.argmin(fits[idx])]

    def mutate(child):
        for g in np.nonzero(rng.random(n) < p_mut)[0]:
            step = int(rng.integers(1, 4)) * (1 if rng.random() < 0.5 else -1)
            child[g] += step
            if nxt[g] >= 0 and rng.random() < 0.5:
                child[nxt[g]] -= step
        np.clip(child, lo, hi, out=child)

    pop = rng.integers(lo, hi + 1, size=(pop_size, n))
    pop[0] = np.clip(inst.run, lo, hi)  # плановое расписание без сдвигов
    if init is not None:
        pop[1] = runs_from_starts(inst, init)
    fits = np.array([fitness(ind) for ind in pop])
    i = int(np.argmin(fits))
    best, best_L = fits[i], pop[i].copy()
    if tracker:
        tracker.record(best)

    gen = stall = 0
    while gen < max_gens and stall < stall_gens and time.perf_counter() < t_end:
        order = np.argsort(fits)
        new_pop = [pop[j].copy() for j in order[:elite]]
        new_fits = [fits[j] for j in order[:elite]]
        while len(new_pop) < pop_size:
            child = pop[tournament(fits)].copy()
            if rng.random() < p_cx:
                other = pop[tournament(fits)]
                for blk in blocks:
                    if rng.random() < 0.5:
                        child[blk] = other[blk]
            mutate(child)
            new_pop.append(child)
            new_fits.append(fitness(child))
        pop, fits = np.array(new_pop), np.array(new_fits)
        gen += 1

        i = int(np.argmin(fits))
        if fits[i] < best - 1e-9:
            best, best_L = fits[i], pop[i].copy()
            stall = 0
            if tracker:
                tracker.record(best)
        else:
            stall += 1

    return starts_from_runs(inst, best_L), {"generations": gen}
