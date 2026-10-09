"""Графики для отчёта (matplotlib, PNG для печати)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import conflict_mask, load_profile, starts_from_runs  # noqa: E402
from .instance import Instance  # noqa: E402

# Цвет и маркер закреплены за методом и не зависят от набора методов на графике.
# Порядок цветов проверен на различимость при дальтонизме.
METHOD_STYLE = {
    "milp": ("MILP (HiGHS)", "#2a78d6", "o"),
    "cpsat": ("CP-SAT", "#1baf7a", "D"),
    "h1": ("H1", "#eda100", "^"),
    "h1h2": ("H1+H2", "#e87ba4", "v"),
    "ga": ("ГА", "#008300", "P"),
}
KIND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # этапы цикла: ТО-1, ТО-2, ТР
CRITICAL = "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SIZE_ORDER = ["S", "M", "L", "XL"]


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _save(fig, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _label(method):
    return METHOD_STYLE.get(method, (method, MUTED, "o"))[0]


def plan_starts(inst: Instance) -> np.ndarray:
    """Плановое расписание: каждый останов точно по плановой наработке."""
    return starts_from_runs(inst, np.clip(inst.run, inst.lo, inst.hi))


# --- расписание ---

def _gantt(ax, inst: Instance, s, title: str):
    cycle = inst.meta.get("generator", {}).get("cycle", [])
    kinds = [c["name"] for c in cycle] or list(dict.fromkeys(st.kind for st in inst.stops))
    color = {k: KIND_COLORS[i % len(KIND_COLORS)] for i, k in enumerate(kinds)}
    for a, st in enumerate(inst.stops):
        ax.barh(st.machine, st.dur, left=s[a], height=0.55, color=color[st.kind],
                edgecolor="white", linewidth=1)
    mask = conflict_mask(inst, s)
    for a, b in zip(inst.pa[mask], inst.pb[mask]):
        x = max(s[a], s[b])
        y = (inst.machine[a] + inst.machine[b]) / 2
        ax.plot(x, y, marker="X", markersize=8, color=CRITICAL,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.set_yticks(range(inst.n_machines), [f"Ст. {i + 1}" for i in range(inst.n_machines)])
    ax.set_ylim(inst.n_machines - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    n_conf = int(mask.sum())
    ax.set_title(f"{title}: конфликтов соседей — {n_conf}", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[k]) for k in kinds]
    labels = list(kinds)
    if n_conf:
        handles.append(plt.Line2D([], [], marker="X", color=CRITICAL, linestyle="",
                                  markersize=8))
        labels.append("конфликт соседей")
    return handles, labels


def schedule_figure(inst: Instance, schedules: dict, path) -> None:
    """Диаграммы Ганта для нескольких расписаний и профили загрузки бригад."""
    _style()
    k = len(schedules)
    height = 0.28 * inst.n_machines + 0.9
    fig, axes = plt.subplots(k + 1, 1, figsize=(6.5, k * height + 2.2), sharex=True,
                             layout="constrained",
                             gridspec_kw={"height_ratios": [height] * k + [1.6]})
    handles = labels = None
    for ax, (title, s) in zip(axes, schedules.items()):
        h, lab = _gantt(ax, inst, np.asarray(s), title)
        if handles is None or len(h) > len(handles):
            handles, labels = h, lab
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))

    ax = axes[-1]
    top = inst.capacity
    # Цвета не совпадают с цветами этапов на диаграммах выше.
    for (title, s), col in zip(schedules.items(), [MUTED, "#4a3aa7", "#e87ba4"]):
        load = load_profile(inst, np.asarray(s))
        top = max(top, int(load.max()))
        ax.step(np.arange(len(load)), load, where="post", color=col, linewidth=1.4,
                label=f"{title} (пик {load.max()})")
    ax.axhline(inst.capacity, color=INK, linewidth=1, label=f"R = {inst.capacity} бригады")
    ax.set_ylim(0, top + 0.5)
    ax.set_yticks(range(top + 1))
    ax.set_ylabel("Остановов")
    ax.set_xlabel("Время от начала планирования T₀, сут")
    ax.set_title("Загрузка ремонтных бригад", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.45), ncol=len(schedules) + 1)
    _save(fig, path)


# --- сравнение методов ---

def _grouped(ax, table: pd.DataFrame, solved: pd.DataFrame, runs: pd.DataFrame, methods,
             sizes, cap: float):
    """Сгруппированные столбцы: группы — размеры, столбцы — методы.

    Столбец выше cap обрезается, над ним пишется значение. Если метод решил
    не все примеры группы, над столбцом пишется «решено k из n».
    """
    n = len(methods)
    width = min(0.8 / n, 0.14)
    x = np.arange(len(sizes))

    def cell(tab, sz, m):
        return tab.loc[sz, m] if (sz in tab.index and m in tab.columns) else np.nan

    for j, m in enumerate(methods):
        label, color, _ = METHOD_STYLE[m]
        vals = np.array([cell(table, sz, m) for sz in sizes], dtype=float)
        pos = x + (j - (n - 1) / 2) * width
        ax.bar(pos, np.minimum(np.nan_to_num(vals, nan=0), cap), width * 0.92, color=color,
               label=label)
        for p, v, sz in zip(pos, vals, sizes):
            k, total = cell(solved, sz, m), cell(runs, sz, m)
            if np.isnan(v):
                ax.text(p, cap * 0.01, "нет решения", rotation=90, ha="center", va="bottom",
                        color=INK2, fontsize=7)
                continue
            notes = []
            if v == 0:
                notes.append("0 — лучшее")
            if v > cap:
                notes.append(f"{v:.0f} %")
            if total and k < total:
                notes.append(f"решено {int(k)} из {int(total)}")
            if notes:
                ax.text(p, min(v, cap) + cap * 0.01, ", ".join(notes), rotation=90,
                        ha="center", va="bottom", color=INK2, fontsize=7)
    ax.set_xticks(x, [f"Размер {s}" for s in sizes])
    ax.set_ylim(0, cap * 1.35)
    ax.grid(axis="x", visible=False)


def comparison_figure(df: pd.DataFrame, path, cap: float = 150.0) -> None:
    """Отклонение от лучшего известного решения по размерам задач."""
    _style()
    methods = [m for m in METHOD_STYLE if m in set(df["method"])]
    sizes = [s for s in SIZE_ORDER if s in set(df["size"])]
    d = df.assign(solved=df["objective"].notna())
    gap = d.pivot_table(index="size", columns="method", values="gap_best_pct", aggfunc="mean")
    solved = d.pivot_table(index="size", columns="method", values="solved", aggfunc="sum")
    runs = d.pivot_table(index="size", columns="method", values="solved", aggfunc="size")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    _grouped(ax, gap, solved, runs, methods, sizes, cap)
    ax.set_ylabel("Отклонение от лучшего, %")
    ax.set_title("Качество решений: среднее по решённым примерам", loc="left")
    ax.legend(ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def time_figure(df: pd.DataFrame, path) -> None:
    """Время до первого и до лучшего решения (логарифмическая шкала)."""
    _style()
    methods = [m for m in METHOD_STYLE if m in set(df["method"])]
    sizes = [s for s in SIZE_ORDER if s in set(df["size"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True, layout="constrained")
    for ax, col, title in zip(axes, ["t_first", "t_best"],
                              ["До первого решения", "До лучшего решения"]):
        tab = df.pivot_table(index="size", columns="method", values=col, aggfunc="mean")
        x = np.arange(len(sizes))
        for j, m in enumerate(methods):
            label, color, marker = METHOD_STYLE[m]
            vals = [tab.loc[sz, m] if (sz in tab.index and m in tab.columns) else np.nan
                    for sz in sizes]
            ax.plot(x + (j - (len(methods) - 1) / 2) * 0.1, vals, marker=marker,
                    linestyle="", markersize=7, color=color, markeredgecolor="white",
                    markeredgewidth=1, label=label)
        ax.set_yscale("log")
        ax.set_xticks(x, sizes)
        ax.set_xlabel("Размер задачи")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Время, с")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(methods),
               loc="outside lower center")
    _save(fig, path)


def convergence_figure(histories: dict, path, title: str) -> None:
    """Значение ЦФ лучшего найденного решения во времени (обе оси логарифмические)."""
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2), layout="constrained")
    t_max = max((h[-1][0] for h in histories.values() if h), default=1)
    for m, hist in histories.items():
        if not hist:
            continue
        label, color, marker = METHOD_STYLE[m]
        t = [max(p[0], 1e-3) for p in hist] + [t_max]
        v = [p[1] for p in hist] + [hist[-1][1]]
        ax.step(t, v, where="post", color=color, linewidth=1.6)
        ax.plot(t[-2], v[-2], marker=marker, markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.plot([], [], color=color, marker=marker, markersize=7, linewidth=1.6,
                label=f"{label}: {hist[-1][1]:.0f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("ЦФ лучшего решения")
    ax.set_title(title + " (маркер — последнее улучшение)", loc="left")
    ax.legend(ncol=2, loc="upper right")
    _save(fig, path)


def sensitivity_figure(df: pd.DataFrame, param: str, metric: str, path, xlabel: str,
                       ylabel: str) -> None:
    """Зависимость критерия от параметра задачи (по одному ряду на метод)."""
    _style()
    methods = [m for m in METHOD_STYLE if m in set(df["method"])]
    # Недопустимые решения (перегрузка бригад) сюда не входят: их ЦФ содержит штраф.
    ok = df[df["feasible"].fillna(False).astype(bool)]
    tab = ok.pivot_table(index="value", columns="method", values=metric, aggfunc="mean")
    tab = tab.reindex(columns=methods)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for m in methods:
        label, color, marker = METHOD_STYLE[m]
        ax.plot(tab.index, tab[m], color=color, linewidth=1.6, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=1, label=label)
    ax.set_xticks(tab.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Чувствительность к параметру «{param}» (только допустимые решения)",
                 loc="left")
    ax.legend(ncol=len(methods), loc="best")
    _save(fig, path)


# --- все графики по результатам экспериментов ---

SENS_LABELS = {
    "tol": "Допуск наработки, доля от плановой",
    "sync": "Доля станков с одинаковым начальным состоянием",
    "capacity": "Число ремонтных бригад R",
    "delta": "Зазор между остановами соседей, сут",
    "run_spread": "Разброс темпа износа",
}


def _load_solution(res_dir: Path, instance: str, method: str):
    path = res_dir / "solutions" / f"{instance}__{method}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_all(res_dir="results", out_dir="figures", gantt_instance="M_s1",
              convergence_instance="L_s1") -> list:
    res_dir, out_dir = Path(res_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    bench = res_dir / "bench.csv"
    if bench.exists():
        df = pd.read_csv(bench)
        comparison_figure(df, out_dir / "comparison_gap.png")
        time_figure(df, out_dir / "comparison_time.png")
        made += ["comparison_gap.png", "comparison_time.png"]

    inst_path = res_dir / "instances" / f"{gantt_instance}.json"
    if inst_path.exists():
        inst = Instance.load(inst_path)
        schedules = {"План без сдвигов": plan_starts(inst)}
        for m in ("cpsat", "h1h2"):
            sol = _load_solution(res_dir, gantt_instance, m)
            if sol and sol["starts"]:
                schedules[_label(m)] = sol["starts"]
                break
        schedule_figure(inst, schedules, out_dir / f"gantt_{gantt_instance}.png")
        made.append(f"gantt_{gantt_instance}.png")

    histories = {}
    for m in METHOD_STYLE:
        sol = _load_solution(res_dir, convergence_instance, m)
        if sol:
            histories[m] = sol["history"]
    if histories:
        convergence_figure(histories, out_dir / f"convergence_{convergence_instance}.png",
                           f"Сходимость методов на примере {convergence_instance}")
        made.append(f"convergence_{convergence_instance}.png")

    for path in sorted(res_dir.glob("sens_*.csv")):
        if path.stem.endswith("_summary"):
            continue
        param = path.stem[len("sens_"):]
        df = pd.read_csv(path)
        sensitivity_figure(df, param, "objective", out_dir / f"sens_{param}.png",
                           SENS_LABELS.get(param, param), "Значение ЦФ (среднее)")
        made.append(f"sens_{param}.png")
    return made
