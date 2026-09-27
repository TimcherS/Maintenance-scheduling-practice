"""Командная строка: python -m msched <команда> ...

Команды:
  generate — создать пример задачи (JSON);
  solve    — решить пример одним методом;
  bench    — сравнить методы на примерах разного размера;
  sens     — анализ чувствительности к параметру;
  plots    — построить графики по результатам.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate
from .experiments import run_benchmark, run_sensitivity
from .instance import SIZES, Instance, generate
from .solvers import METHODS, solve


def _value(text: str):
    """Число из строки: сначала int, потом float, иначе строка."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _methods(items):
    return list(METHODS) if items == ["all"] else items


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="msched", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="создать пример задачи")
    g.add_argument("--size", choices=SIZES, help="типовой размер (задаёт станки и горизонт)")
    g.add_argument("--n-machines", type=int)
    g.add_argument("--horizon", type=int, help="горизонт планирования, сут")
    g.add_argument("--topology", choices=["line", "ring", "grid"], default="line")
    g.add_argument("--tol", type=float, default=0.15, help="допуск наработки, доля")
    g.add_argument("--delta", type=int, default=1, help="зазор между остановами соседей, сут")
    g.add_argument("--mode", choices=["overlap", "finish"], default="overlap")
    g.add_argument("--capacity", type=int, help="число ремонтных бригад R")
    g.add_argument("--sync", type=float, default=0.5, help="доля синхронных станков")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("-o", "--out", required=True)

    s = sub.add_parser("solve", help="решить пример")
    s.add_argument("instance")
    s.add_argument("-m", "--method", choices=METHODS, default="cpsat")
    s.add_argument("-t", "--time-limit", type=float, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-o", "--out", help="файл для решения (JSON)")

    b = sub.add_parser("bench", help="сравнение методов")
    b.add_argument("--sizes", nargs="+", default=["S", "M", "L"], choices=SIZES)
    b.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    b.add_argument("--methods", nargs="+", default=["all"])
    b.add_argument("-t", "--time-limit", type=float, default=60)
    b.add_argument("--out", default="results")

    e = sub.add_parser("sens", help="анализ чувствительности")
    e.add_argument("--param", required=True,
                   help="tol | delta | sync | capacity | run_spread | w_conf | w_peak | w_dev")
    e.add_argument("--values", nargs="+", type=_value, required=True)
    e.add_argument("--size", default="M", choices=SIZES)
    e.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    e.add_argument("--methods", nargs="+", default=["all"])
    e.add_argument("-t", "--time-limit", type=float, default=30)
    e.add_argument("--out", default="results")

    p = sub.add_parser("plots", help="построить графики по результатам")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--gantt", default="M_s1", help="пример для диаграммы Ганта")
    p.add_argument("--convergence", default="L_s1", help="пример для графика сходимости")

    args = ap.parse_args(argv)

    if args.cmd == "generate":
        params = dict(SIZES[args.size]) if args.size else {}
        if args.n_machines:
            params["n_machines"] = args.n_machines
        if args.horizon:
            params["horizon"] = args.horizon
        inst = generate(**params, topology=args.topology, tol=args.tol, delta=args.delta,
                        mode=args.mode, capacity=args.capacity, sync=args.sync,
                        seed=args.seed, name=Path(args.out).stem)
        inst.save(args.out)
        print(f"{args.out}: станков {inst.n_machines}, остановов {inst.n}, "
              f"пар соседних остановов {len(inst.pa)}, бригад {inst.capacity}")

    elif args.cmd == "solve":
        inst = Instance.load(args.instance)
        res = solve(inst, args.method, time_limit=args.time_limit, seed=args.seed)
        print(f"Метод: {res.method}, статус: {res.status}")
        if res.starts is None:
            print("Решение не найдено.")
            return
        for key, val in evaluate(inst, res.starts).as_dict().items():
            print(f"  {key:10s} {val}")
        print(f"  t_first    {res.t_first:.3f} с\n  t_best     {res.t_best:.3f} с\n"
              f"  t_total    {res.time_total:.3f} с")
        if res.bound is not None:
            print(f"  bound      {res.bound}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    elif args.cmd == "bench":
        run_benchmark(args.sizes, args.seeds, _methods(args.methods), args.time_limit, args.out)

    elif args.cmd == "sens":
        run_sensitivity(args.param, args.values, args.size, args.seeds, _methods(args.methods),
                        args.time_limit, args.out)

    elif args.cmd == "plots":
        from .plots import build_all
        for name in build_all(args.results, args.out, args.gantt, args.convergence):
            print(f"{args.out}/{name}")


if __name__ == "__main__":
    main()
