#!/usr/bin/env bash
# Полный набор экспериментов для отчёта (~70 мин на 8 ядрах).
# Запуск из корня проекта: bash scripts/run_experiments.sh
set -e
PY=${PY:-python}

# 1. Сравнение всех методов: S, M, L по 3 примера, лимит 60 с.
$PY -m msched bench --sizes S M L --seeds 1 2 3 -t 60 --out results

# 2. Анализ чувствительности на примерах M (без MILP, лимит 20 с).
SENS="--size M --seeds 1 2 --methods cpsat h1 h1h2 ga -t 20 --out results"
$PY -m msched sens --param tol      --values 0.05 0.1 0.15 0.2 0.3 $SENS
$PY -m msched sens --param sync     --values 0 0.25 0.5 0.75 1     $SENS
$PY -m msched sens --param capacity --values 1 2 3 4              $SENS
