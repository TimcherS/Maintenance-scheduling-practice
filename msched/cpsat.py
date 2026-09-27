"""Модель CP-SAT (Google OR-Tools).

Переменные s[a] — моменты начала остановов. Пересечения соседей
моделируются дизъюнкцией с флагом конфликта, загрузка бригад —
глобальным ограничением cumulative.
"""
from __future__ import annotations

import numpy as np
from ortools.sat.python import cp_model

from .instance import Instance
from .result import Tracker


class _Recorder(cp_model.CpSolverSolutionCallback):
    """Запоминает момент каждого нового решения."""

    def __init__(self, tracker: Tracker):
        super().__init__()
        self.tracker = tracker

    def on_solution_callback(self):
        self.tracker.record(self.objective_value)


def solve_cpsat(inst: Instance, time_limit: float = 60.0, tracker: Tracker | None = None,
                workers: int = 8, seed: int = 0, hint=None, verbose: bool = False):
    tracker = tracker or Tracker()
    m = cp_model.CpModel()
    n = inst.n
    w = inst.weights
    dur = [int(v) for v in inst.dur]
    span = [int(v) for v in inst.span]

    S = [m.new_int_var(int(inst.est[a]), int(inst.lst[a]), f"s{a}") for a in range(n)]

    # Наработка в допустимом окне и её отклонение от плана.
    dev = []
    for a in range(n):
        p = int(inst.prev[a])
        run_expr = S[a] - S[p] - dur[p] if p >= 0 else S[a]
        lo, hi, run = int(inst.lo[a]), int(inst.hi[a]), int(inst.run[a])
        if p >= 0:
            m.add(run_expr >= lo)
            m.add(run_expr <= hi)
        d = m.new_int_var(0, max(run - lo, hi - run), f"dev{a}")
        m.add(d >= run_expr - run)
        m.add(d >= run - run_expr)
        dev.append(d)

    # Соседи: либо a раньше b, либо b раньше a, либо пара считается конфликтом.
    z = []
    for a, b in zip(inst.pa.tolist(), inst.pb.tolist()):
        conf = m.new_bool_var(f"z{a}_{b}")
        a_first = m.new_bool_var(f"o{a}_{b}")
        m.add(S[a] + span[a] <= S[b]).only_enforce_if([a_first, conf.Not()])
        m.add(S[b] + span[b] <= S[a]).only_enforce_if([a_first.Not(), conf.Not()])
        z.append(conf)

    # Загрузка бригад.
    intervals = [m.new_fixed_size_interval_var(S[a], dur[a], f"i{a}") for a in range(n)]
    P = m.new_int_var(0, int(inst.capacity), "peak")
    m.add_cumulative(intervals, [1] * n, P)

    m.minimize(w["dev"] * sum(dev) + w["conf"] * sum(z) + w["peak"] * P)

    if hint is not None:
        for a in range(n):
            m.add_hint(S[a], int(hint[a]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = verbose
    code = solver.solve(m, _Recorder(tracker))

    status = {
        cp_model.OPTIMAL: "optimal",
        cp_model.FEASIBLE: "feasible",
        cp_model.INFEASIBLE: "infeasible",
    }.get(code, "unknown")
    starts = None
    if status in ("optimal", "feasible"):
        starts = np.array([solver.value(v) for v in S], dtype=np.int64)
        tracker.record(solver.objective_value)
    stats = {"branches": solver.num_branches, "conflicts_cp": solver.num_conflicts,
             "workers": workers}
    return starts, status, float(solver.best_objective_bound), stats
