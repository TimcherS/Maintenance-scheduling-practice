"""Вычислительные эксперименты: сравнение методов и анализ чувствительности."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluate import evaluate
from .instance import DEFAULT_WEIGHTS, SIZES, Instance, generate
from .result import SolveResult
from .solvers import solve


def result_row(inst: Instance, res: SolveResult) -> dict:
    row = {
        "instance": inst.name,
        "n_machines": inst.n_machines,
        "n_stops": inst.n,
        "n_pairs": len(inst.pa),
        "method": res.method,
        "status": res.status,
        "bound": res.bound,
        "t_first": res.t_first,
        "t_best": res.t_best,
        "t_opt": res.t_opt,
        "t_total": res.time_total,
    }
    if res.starts is not None:
        row.update(evaluate(inst, res.starts).as_dict())
    return row


def add_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Отклонение от лучшего известного решения на том же примере, %."""
    best = df.groupby("instance")["objective"].transform("min")
    df["gap_best_pct"] = 100 * (df["objective"] - best) / best.where(best > 0)
    return df


def _save_solution(out_dir: Path, inst: Instance, res: SolveResult) -> None:
    path = out_dir / "solutions" / f"{inst.name}__{res.method}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"instance": inst.name, **res.to_dict()}, ensure_ascii=False),
                    encoding="utf-8")


def _run(inst: Instance, methods, time_limit: float, seed: int, out_dir: Path, extra: dict):
    rows = []
    for method in methods:
        res = solve(inst, method, time_limit=time_limit, seed=seed)
        _save_solution(out_dir, inst, res)
        row = {**extra, **result_row(inst, res)}
        rows.append(row)
        print(f"  {inst.name:28s} {method:6s} {res.status:9s} "
              f"ЦФ={row.get('objective', float('nan')):9.1f} "
              f"t_best={res.t_best or 0:7.2f} с  всего={res.time_total:7.2f} с", flush=True)
    return rows


def run_benchmark(sizes, seeds, methods, time_limit: float, out_dir) -> pd.DataFrame:
    """Сравнение методов на примерах разного размера."""
    out_dir = Path(out_dir)
    rows = []
    for size in sizes:
        for seed in seeds:
            inst = generate(**SIZES[size], seed=seed, name=f"{size}_s{seed}")
            inst.save(out_dir / "instances" / f"{inst.name}.json")
            rows += _run(inst, methods, time_limit, seed, out_dir, {"size": size, "seed": seed})
    df = add_gaps(pd.DataFrame(rows))
    df.to_csv(out_dir / "bench.csv", index=False, encoding="utf-8-sig")
    summarize(df, ["size", "method"]).to_csv(out_dir / "bench_summary.csv", encoding="utf-8-sig")
    return df


def run_sensitivity(param: str, values, size: str, seeds, methods, time_limit: float,
                    out_dir) -> pd.DataFrame:
    """Анализ чувствительности к одному параметру генератора.

    param — параметр generate() (tol, delta, sync, capacity, run_spread, ...)
    или вес ЦФ с префиксом w_ (w_conf, w_peak, w_dev).
    """
    out_dir = Path(out_dir)
    rows = []
    for value in values:
        for seed in seeds:
            kwargs = dict(SIZES[size])
            if param.startswith("w_"):
                kwargs["weights"] = {**DEFAULT_WEIGHTS, param[2:]: value}
            else:
                kwargs[param] = value
            inst = generate(**kwargs, seed=seed, name=f"{size}_{param}={value}_s{seed}")
            rows += _run(inst, methods, time_limit, seed, out_dir,
                         {"param": param, "value": value, "size": size, "seed": seed})
    df = add_gaps(pd.DataFrame(rows))
    df.to_csv(out_dir / f"sens_{param}.csv", index=False, encoding="utf-8-sig")
    summarize(df, ["value", "method"]).to_csv(out_dir / f"sens_{param}_summary.csv",
                                              encoding="utf-8-sig")
    return df


def summarize(df: pd.DataFrame, by) -> pd.DataFrame:
    """Средние значения критериев по группам."""
    cols = ["objective", "gap_best_pct", "sum_dev", "conflicts", "peak", "load_std",
            "t_first", "t_best", "t_total"]
    agg = df.groupby(by)[[c for c in cols if c in df]].mean().round(3)
    agg["n_optimal"] = df.assign(opt=df["status"].eq("optimal")).groupby(by)["opt"].sum()
    return agg
