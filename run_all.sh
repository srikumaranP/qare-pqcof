#!/bin/sh
# Reproduces every table and figure. Run from the code/ directory. Total time: roughly 10-20 minutes on one vCPU.
set -e
cd "$(dirname "$0")/code"
python3 exp1.py          # Table 3: ML-KEM / ML-DSA reference latency and sizes  -> exp1.json   (timings vary by machine)
python3 exp2.py          # Table 4 / Fig. 2: QAOA vs classical solvers          -> exp2.json
python3 exp3.py          # Tables 5-7 / Figs. 3-4: circuits, pruning, mitigation -> exp3.json
python3 agg.py           # statistics, tables, Figs. 2-4 -> agg.json, fig*.png
python3 make_fig1_fig2.py  # final Fig. 1 and Fig. 2 (reads agg.json from the current folder)
