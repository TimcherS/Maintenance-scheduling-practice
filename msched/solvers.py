"""Единая точка вызова всех методов."""
from __future__ import annotations

from .cpsat import solve_cpsat
from .evaluate import objective
from .ga import genetic
from .heuristics import greedy, local_search
from .instance import Instance
from .milp import solve_milp
from .result import SolveResult, Tracker

METHODS = ("milp", "milp_cbc", "cpsat", "h1", "h1h2", "ga")


def solve(inst: Instance, method: str, time_limit: float = 60.0, seed: int = 0,
          **kwargs) -> SolveResult:
    tr = Tracker()
    bound, info = None, {}
    if method == "milp":
        starts, status, bound, info = solve_milp(inst, time_limit, tr, **kwargs)
    elif method == "milp_cbc":
        starts, status, bound, info = solve_milp(inst, time_limit, tr, solver="cbc", **kwargs)
    elif method == "cpsat":
        starts, status, bound, info = solve_cpsat(inst, time_limit, tr, seed=seed, **kwargs)
    elif method == "h1":
        starts, status = greedy(inst), "heuristic"
        tr.record(objective(inst, starts))
    elif method == "h1h2":
        starts = greedy(inst)
        tr.record(objective(inst, starts))
        starts = local_search(inst, starts, max(0.0, time_limit - tr.elapsed()), seed, tr)
        status = "heuristic"
    elif method == "ga":
        starts, info = genetic(inst, time_limit, seed=seed, tracker=tr, **kwargs)
        status = "heuristic"
    else:
        raise ValueError(f"неизвестный метод: {method}; есть {METHODS}")
    return SolveResult(method, starts, status, tr.elapsed(), tr.history, bound, info)
