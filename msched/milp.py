"""Модель MILP с индексацией по времени (PuLP, решатели HiGHS и CBC).

Переменные:
  x[a][k] ∈ {0,1} — останов a начинается в момент est[a] + k;
  dev[a] ≥ 0      — отклонение наработки от плановой;
  z[p] ∈ {0,1}    — пара соседних остановов p пересекается;
  P ∈ Z           — пиковое число одновременных остановов.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pulp

from .instance import Instance
from .result import Tracker


def build_milp(inst: Instance):
    """Строит модель PuLP. Возвращает (задача, переменные x)."""
    n = inst.n
    est = [int(v) for v in inst.est]
    lst = [int(v) for v in inst.lst]
    dur = [int(v) for v in inst.dur]
    span = [int(v) for v in inst.span]
    w = inst.weights

    prob = pulp.LpProblem("maintenance", pulp.LpMinimize)
    x = [[pulp.LpVariable(f"x_{a}_{k}", cat="Binary") for k in range(lst[a] - est[a] + 1)]
         for a in range(n)]

    # Каждый останов начинается ровно один раз.
    for a in range(n):
        prob += pulp.lpSum(x[a]) == 1, f"once_{a}"

    # Момент начала: s[a] = est[a] + Σ k·x[a][k].
    S = [est[a] + pulp.lpSum(k * v for k, v in enumerate(x[a])) for a in range(n)]

    # Наработка в допустимом окне и её отклонение от плана.
    dev = []
    for a in range(n):
        p = int(inst.prev[a])
        run_expr = S[a] - S[p] - dur[p] if p >= 0 else S[a]
        lo, hi, run = int(inst.lo[a]), int(inst.hi[a]), int(inst.run[a])
        if p >= 0:
            prob += run_expr >= lo, f"run_lo_{a}"
            prob += run_expr <= hi, f"run_hi_{a}"
        d = pulp.LpVariable(f"dev_{a}", lowBound=0, upBound=max(run - lo, hi - run))
        prob += d >= run_expr - run, f"dev_pos_{a}"
        prob += d >= run - run_expr, f"dev_neg_{a}"
        dev.append(d)

    def occupying(a: int, t: int, length: int) -> list:
        """Переменные x[a][k], при которых останов a занимает момент t."""
        k_lo = max(0, t - length + 1 - est[a])
        k_hi = min(len(x[a]) - 1, t - est[a])
        return x[a][k_lo:k_hi + 1]

    # Пересечения остановов соседних станков: z = 1, если пара пересекается.
    z = []
    for a, b in zip(inst.pa.tolist(), inst.pb.tolist()):
        zp = pulp.LpVariable(f"z_{a}_{b}", cat="Binary")
        t_lo = max(est[a], est[b])
        t_hi = min(lst[a] + span[a], lst[b] + span[b]) - 1
        for t in range(t_lo, t_hi + 1):
            va, vb = occupying(a, t, span[a]), occupying(b, t, span[b])
            if va and vb:
                prob += pulp.lpSum(va + vb) - zp <= 1, f"conf_{a}_{b}_{t}"
        z.append(zp)

    # Загрузка бригад: в каждые сутки не больше P ≤ R остановов.
    P = pulp.LpVariable("peak", lowBound=0, upBound=int(inst.capacity), cat="Integer")
    buckets = [[] for _ in range(inst.T)]
    for a in range(n):
        for k, v in enumerate(x[a]):
            for t in range(est[a] + k, est[a] + k + dur[a]):
                buckets[t].append(v)
    for t, bucket in enumerate(buckets):
        if bucket:
            prob += pulp.lpSum(bucket) <= P, f"load_{t}"

    prob += w["dev"] * pulp.lpSum(dev) + w["conf"] * pulp.lpSum(z) + w["peak"] * P
    return prob, x


class _HiGHSWithTracker(pulp.HiGHS):
    """Решатель HiGHS для PuLP, который запоминает момент каждого нового решения."""

    def __init__(self, tracker: Tracker, **kwargs):
        super().__init__(**kwargs)
        self.tracker = tracker

    def callSolver(self, lp):
        lp.solverModel.cbMipImprovingSolution.subscribe(
            lambda e: self.tracker.record(e.data_out.objective_function_value))
        lp.solverModel.run()


# Строка журнала CBC о новом целочисленном решении, например:
# "Cbc0012I Integer solution of 65 found by DiveCoefficient after 0 iterations
#  and 0 nodes (0.52 seconds)"
_CBC_SOLUTION = re.compile(r"Integer solution of (\S+) found.*\(([\d.]+) seconds\)")


def _solve_cbc(prob: pulp.LpProblem, time_limit: float, tracker: Tracker, verbose: bool):
    """CBC не даёт обратных вызовов, поэтому моменты решений берутся из его журнала."""
    t_build = tracker.elapsed()
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cbc.log"
        prob.solve(pulp.PULP_CBC_CMD(msg=verbose, timeLimit=float(time_limit), gapRel=0.0,
                                     logPath=str(log_path)))
        log = log_path.read_text(errors="ignore") if log_path.exists() else ""
    for value, sec in _CBC_SOLUTION.findall(log):
        tracker.history.append((t_build + float(sec), float(value)))
    tracker.history.sort()
    # Оставляем только улучшения.
    best = []
    for t, v in tracker.history:
        if not best or v < best[-1][1] - 1e-9:
            best.append((t, v))
    tracker.history[:] = best
    return {}


def _solve_highs(prob: pulp.LpProblem, time_limit: float, tracker: Tracker, verbose: bool):
    prob.solve(_HiGHSWithTracker(tracker, msg=verbose, timeLimit=float(time_limit), gapRel=0.0))
    info = prob.solverModel.getInfo()  # объект highspy.Highs после решения
    return {"bound": float(info.mip_dual_bound), "nodes": int(info.mip_node_count)}


def solve_milp(inst: Instance, time_limit: float = 60.0, tracker: Tracker | None = None,
               verbose: bool = False, solver: str = "highs"):
    """solver — "highs" (HiGHS через highspy) или "cbc" (CBC, встроен в PuLP)."""
    tracker = tracker or Tracker()
    prob, x = build_milp(inst)
    if solver == "highs":
        stats = _solve_highs(prob, time_limit, tracker, verbose)
    elif solver == "cbc":
        stats = _solve_cbc(prob, time_limit, tracker, verbose)
    else:
        raise ValueError(f"неизвестный решатель: {solver}")

    # sol_status отличает оптимум от решения, найденного к концу лимита времени.
    if prob.sol_status == pulp.LpSolutionOptimal:
        status = "optimal"
    elif prob.sol_status == pulp.LpSolutionIntegerFeasible:
        status = "feasible"
    elif prob.status == pulp.LpStatusInfeasible:
        status = "infeasible"
    else:
        status = "unknown"

    starts = None
    if status in ("optimal", "feasible"):
        est = inst.est
        starts = np.array([int(est[a]) + int(np.argmax([v.varValue or 0 for v in x[a]]))
                           for a in range(inst.n)], dtype=np.int64)
        tracker.record(pulp.value(prob.objective))

    bound = stats.pop("bound", None)
    stats.update({"solver": solver, "vars": len(prob.variables()),
                  "constraints": len(prob.constraints)})
    return starts, status, bound, stats
