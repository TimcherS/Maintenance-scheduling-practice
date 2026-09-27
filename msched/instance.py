"""Данные задачи планирования остановов и генератор тестовых примеров.

Время дискретное (сутки). Станок работает циклами из K этапов
(по умолчанию 3: ТО-1, ТО-2, ТР). Каждый этап заканчивается остановом
на обслуживание длительностью dur. Перед остановом станок должен
отработать плановую наработку run. Допустимая наработка лежит в окне
[lo, hi]. Соседние станки не должны стоять на обслуживании одновременно.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_CYCLE = (
    {"name": "ТО-1", "run": 30, "dur": 1},
    {"name": "ТО-2", "run": 30, "dur": 2},
    {"name": "ТР", "run": 30, "dur": 4},
)

# Веса критериев целевой функции:
# dev  — сутки отклонения от плановой наработки,
# conf — пара пересекающихся остановов соседних станков,
# peak — пиковое число одновременных остановов (загрузка бригад),
# over — штраф за превышение числа бригад (только для эвристик).
DEFAULT_WEIGHTS = {"dev": 1, "conf": 20, "peak": 10, "over": 1000}

# Типовые размеры примеров.
SIZES = {
    "S": {"n_machines": 5, "horizon": 120},
    "M": {"n_machines": 10, "horizon": 180},
    "L": {"n_machines": 20, "horizon": 365},
    "XL": {"n_machines": 40, "horizon": 365},
}


@dataclass
class Stop:
    """Плановый останов станка на обслуживание."""

    machine: int
    pos: int  # номер останова у станка: 0, 1, ...
    kind: str  # вид обслуживания (этап цикла)
    run: int  # плановая наработка перед остановом, сут
    dur: int  # длительность останова, сут
    lo: int  # минимально допустимая наработка
    hi: int  # максимально допустимая наработка


@dataclass
class Instance:
    """Пример задачи. После создания строит массивы numpy для решателей."""

    name: str
    n_machines: int
    neighbors: list
    stops: list
    capacity: int  # R — число ремонтных бригад
    delta: int = 1  # минимальный зазор между остановами соседей, сут
    mode: str = "overlap"  # "overlap" — остановы не пересекаются, "finish" — не совпадают окончания этапов
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.neighbors = [tuple(sorted((int(i), int(k)))) for i, k in self.neighbors]
        self.stops = [s if isinstance(s, Stop) else Stop(**s) for s in self.stops]
        self.stops.sort(key=lambda s: (s.machine, s.pos))
        self.weights = {**DEFAULT_WEIGHTS, **self.weights}
        self._build_arrays()

    def _build_arrays(self):
        st = self.stops
        n = self.n = len(st)
        i64 = np.int64
        self.machine = np.array([s.machine for s in st], dtype=i64)
        self.run = np.array([s.run for s in st], dtype=i64)
        self.dur = np.array([s.dur for s in st], dtype=i64)
        self.lo = np.array([s.lo for s in st], dtype=i64)
        self.hi = np.array([s.hi for s in st], dtype=i64)

        # Порядок остановов у каждого станка.
        self.by_machine = [[] for _ in range(self.n_machines)]
        self.prev = np.full(n, -1, dtype=i64)
        self.next = np.full(n, -1, dtype=i64)
        for a, s in enumerate(st):
            ms = self.by_machine[s.machine]
            if ms:
                self.prev[a] = ms[-1]
                self.next[ms[-1]] = a
            ms.append(a)
        self.last = np.zeros(n, dtype=i64)
        for ms in self.by_machine:
            for a in ms:
                self.last[a] = ms[-1]

        # Окна [est, lst] для момента начала каждого останова.
        self.est = np.zeros(n, dtype=i64)
        self.lst = np.zeros(n, dtype=i64)
        for ms in self.by_machine:
            e = l = 0
            for a in ms:
                self.est[a] = e + self.lo[a]
                self.lst[a] = l + self.hi[a]
                e = self.est[a] + self.dur[a]
                l = self.lst[a] + self.dur[a]

        # Интервал, который не должен пересекаться с интервалом соседа.
        if self.mode == "overlap":
            self.span = self.dur + self.delta
        elif self.mode == "finish":
            self.span = np.full(n, max(self.delta, 1), dtype=i64)
        else:
            raise ValueError(f"неизвестный режим: {self.mode}")

        self.T = int((self.lst + np.maximum(self.dur, self.span)).max()) + 1 if n else 1

        # Пары остановов соседних станков, которые могут пересечься.
        pa, pb = [], []
        for i, k in self.neighbors:
            for a in self.by_machine[i]:
                for b in self.by_machine[k]:
                    if (self.est[a] < self.lst[b] + self.span[b]
                            and self.est[b] < self.lst[a] + self.span[a]):
                        pa.append(a)
                        pb.append(b)
        self.pa = np.array(pa, dtype=i64)
        self.pb = np.array(pb, dtype=i64)
        self.partners = [[] for _ in range(n)]
        for a, b in zip(pa, pb):
            self.partners[a].append(b)
            self.partners[b].append(a)

    # --- ввод-вывод ---

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_machines": self.n_machines,
            "neighbors": [list(e) for e in self.neighbors],
            "capacity": self.capacity,
            "delta": self.delta,
            "mode": self.mode,
            "weights": self.weights,
            "meta": self.meta,
            "stops": [asdict(s) for s in self.stops],
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Instance":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def make_topology(n: int, topology: str) -> list:
    """Граф соседства станков."""
    if topology == "line":
        return [(i, i + 1) for i in range(n - 1)]
    if topology == "ring":
        return [(i, (i + 1) % n) for i in range(n)] if n > 2 else make_topology(n, "line")
    if topology == "grid":
        cols = math.ceil(math.sqrt(n))
        edges = []
        for i in range(n):
            if (i + 1) % cols and i + 1 < n:
                edges.append((i, i + 1))
            if i + cols < n:
                edges.append((i, i + cols))
        return edges
    raise ValueError(f"неизвестная топология: {topology}")


def generate(
    n_machines: int = 10,
    horizon: int = 180,
    *,
    topology: str = "line",
    cycle=DEFAULT_CYCLE,
    run_spread: float = 0.2,
    tol: float = 0.15,
    delta: int = 1,
    mode: str = "overlap",
    capacity: int | None = None,
    sync: float = 0.5,
    weights: dict | None = None,
    seed: int = 0,
    name: str | None = None,
) -> Instance:
    """Случайный пример задачи.

    run_spread — разброс темпа износа станков (±доля от плановой наработки);
    tol        — допустимое отклонение наработки (доля от плановой);
    sync       — доля станков с одинаковым начальным состоянием. Чем она
                 больше, тем чаще плановые остановы соседей совпадают.
    """
    rng = np.random.default_rng(seed)
    K = len(cycle)
    common_phase = int(rng.integers(K))
    common_frac = float(rng.random())
    stops = []
    for i in range(n_machines):
        factor = rng.uniform(1 - run_spread, 1 + run_spread)
        runs = [max(2, round(c["run"] * factor)) for c in cycle]
        if rng.random() < sync:
            k, frac = common_phase, common_frac
        else:
            k, frac = int(rng.integers(K)), float(rng.random())
        # Первый останов: остаток наработки на момент начала планирования T0.
        r = max(1, round(runs[k] * (1 - frac)))
        t = pos = 0
        while t + r < horizon:
            slack = round(tol * runs[k])
            dur = int(cycle[k]["dur"])
            stops.append(Stop(i, pos, cycle[k]["name"], r, dur, max(0, r - slack), r + slack))
            t += r + dur
            pos += 1
            k = (k + 1) % K
            r = runs[k]
    if capacity is None:
        capacity = max(1, math.ceil(n_machines / 4))
    meta = {
        "generator": {
            "n_machines": n_machines, "horizon": horizon, "topology": topology,
            "cycle": list(cycle), "run_spread": run_spread, "tol": tol,
            "sync": sync, "seed": seed,
        }
    }
    return Instance(
        name=name or f"N{n_machines}_H{horizon}_s{seed}",
        n_machines=n_machines,
        neighbors=make_topology(n_machines, topology),
        stops=stops,
        capacity=capacity,
        delta=delta,
        mode=mode,
        weights={**DEFAULT_WEIGHTS, **(weights or {})},
        meta=meta,
    )
