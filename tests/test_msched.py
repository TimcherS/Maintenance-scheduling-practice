import numpy as np
import pytest

from msched import METHODS, Instance, evaluate, generate, solve
from msched.evaluate import runs_from_starts, starts_from_runs


@pytest.fixture(scope="module")
def tiny():
    return generate(n_machines=4, horizon=80, sync=1.0, seed=3)


def test_runs_starts_roundtrip(tiny):
    rng = np.random.default_rng(0)
    L = rng.integers(tiny.lo, tiny.hi + 1)
    assert (runs_from_starts(tiny, starts_from_runs(tiny, L)) == L).all()


def test_json_roundtrip(tiny, tmp_path):
    path = tmp_path / "inst.json"
    tiny.save(path)
    other = Instance.load(path)
    assert other.n == tiny.n
    assert (other.est == tiny.est).all()
    assert (other.pa == tiny.pa).all()


@pytest.mark.parametrize("method", METHODS)
def test_method_gives_feasible_schedule(tiny, method):
    res = solve(tiny, method, time_limit=10)
    assert res.starts is not None
    L = runs_from_starts(tiny, res.starts)
    assert ((L >= tiny.lo) & (L <= tiny.hi)).all()
    # Жадная H1 может превысить число бригад: это её известный недостаток.
    if method != "h1":
        assert evaluate(tiny, res.starts).feasible


def test_exact_methods_agree(tiny):
    """MILP и CP-SAT доказывают один и тот же оптимум."""
    milp = solve(tiny, "milp", time_limit=60)
    cpsat = solve(tiny, "cpsat", time_limit=60)
    assert milp.status == cpsat.status == "optimal"
    opt = evaluate(tiny, cpsat.starts).objective
    assert evaluate(tiny, milp.starts).objective == pytest.approx(opt)
    # Значение ЦФ решателя совпадает с независимой оценкой.
    assert cpsat.history[-1][1] == pytest.approx(opt)
    # Эвристики не лучше оптимума.
    for method in ("h1", "h1h2", "ga"):
        assert evaluate(tiny, solve(tiny, method, time_limit=5).starts).objective >= opt - 1e-9


def test_finish_mode():
    inst = generate(n_machines=4, horizon=80, mode="finish", delta=2, seed=1)
    assert (inst.span == 2).all()
    res = solve(inst, "cpsat", time_limit=10)
    assert res.status == "optimal"


def test_excel_export(tiny, tmp_path):
    import pandas as pd

    from msched.plots import plan_starts
    from msched.report import export_excel

    path = tmp_path / "schedule.xlsx"
    export_excel(tiny, plan_starts(tiny), path, method="план")
    table = pd.read_excel(path, sheet_name="Расписание")
    assert len(table) == tiny.n
