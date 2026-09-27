"""Таблица расписания и выгрузка в Excel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import conflict_mask, evaluate, runs_from_starts
from .instance import Instance


def schedule_table(inst: Instance, starts) -> pd.DataFrame:
    """Расписание остановов в виде таблицы (одна строка — один останов)."""
    s = np.asarray(starts, dtype=np.int64)
    L = runs_from_starts(inst, s)
    in_conflict = np.zeros(inst.n, dtype=bool)
    mask = conflict_mask(inst, s)
    in_conflict[inst.pa[mask]] = True
    in_conflict[inst.pb[mask]] = True
    return pd.DataFrame({
        "Станок": inst.machine + 1,
        "№ останова": [st.pos + 1 for st in inst.stops],
        "Вид": [st.kind for st in inst.stops],
        "Плановая наработка": inst.run,
        "Допустимо от": inst.lo,
        "Допустимо до": inst.hi,
        "Фактическая наработка": L,
        "Отклонение": L - inst.run,
        "Начало, сут": s,
        "Конец, сут": s + inst.dur,
        "Конфликт с соседом": np.where(in_conflict, "да", ""),
    })


def export_excel(inst: Instance, starts, path, method: str = "") -> None:
    """Лист «Расписание» и лист «Критерии»."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(inst, starts).as_dict()
    names = {
        "objective": "Значение ЦФ",
        "sum_dev": "Σ отклонений от плановой наработки, сут",
        "max_dev": "Максимальное отклонение, сут",
        "conflicts": "Конфликтов соседних станков",
        "peak": "Пиковое число одновременных остановов",
        "load_std": "СКО загрузки бригад",
        "overload": "Превышение числа бригад, бригадо-сут",
        "feasible": "Все ограничения выполнены",
    }
    crit = pd.DataFrame({"Критерий": ["Пример", "Метод"] + [names[k] for k in metrics],
                         "Значение": [inst.name, method] + list(metrics.values())})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        schedule_table(inst, starts).to_excel(xw, sheet_name="Расписание", index=False)
        crit.to_excel(xw, sheet_name="Критерии", index=False)
