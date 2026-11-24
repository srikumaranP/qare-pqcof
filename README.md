# qare-pqcof
# QARE-PQCOF: reproducibility package

Code, seeds and raw results for the paper
"Optimization Techniques for Enhancing the Efficiency of Quantum-Safe Cryptographic Systems: A Performance-Driven Analysis"
(P. Srikumaran, M. Ilayaraja).

## Contents
- `code/exp1.py`   Baseline latency and sizes of reference ML-KEM and ML-DSA (kyber-py, dilithium-py). 40 repetitions per KEM operation, 20 per signature operation.
- `code/qlib.py`   Instance generator, QUBO and Ising mapping, exact QAOA simulator, optimiser (COBYLA, 5 restarts, interpolation warm start).
- `code/exp2.py`   QAOA (p = 1..3) against random sampling, generic simulated annealing and structure-aware annealing. 15 instances each for n = 6, 9, 12.
- `code/exp3.py`   Circuit reduction (transpiler levels, edge-colour ordering), pruning ablation, noisy simulation with readout mitigation and ZNE.
- `code/agg.py`, `code/make_fig1_fig2.py`   Statistics, tables and figures.
- `results/`       Raw outputs used in the paper (`exp1.json`, `exp2.json`, `exp3.json`, `agg.json`).
- `figures/`       Figures 1 to 4.

## Run
    pip install -r requirements.txt
    ./run_all.sh

## Seeds
- Instance seed: `1000*n + s` for n qubits and s = 0..14 (s = 0..7 for the circuit table, 0..9 for pruning, 0..11 for the noisy runs).
- Classical baselines: `s`, `100+s`, `200+s`. QAOA restarts: `s`. Shot sampling: `500+s`. Noisy simulator: `100+s+lambda`. Readout calibration: 11. Transpiler: 7 (routing study), 3 (noisy runs).

## Important limits (also stated in the paper)
- Configuration instances are synthetic. They test the optimiser, not any PQC speed-up.
- ML-KEM / ML-DSA timings come from pure-Python reference code on one vCPU and are not representative of optimised C, assembly or hardware. Re-running exp1.py on other machines will give different absolute times; sizes are fixed by FIPS 203/204.
- QAOA ran on an exact simulator; noise parameters are assumed, not measured. No quantum hardware was used.
- Energy, CPU utilisation, side channels and NIST ACVP known-answer vectors were not evaluated.
- `exp2.py` and `exp3.py` will give the same numbers as `results/` for the same library versions; small differences are possible across versions or platforms.

## Software used
Python 3.12.3, Qiskit 2.6.2, Qiskit Aer 0.18.4, NumPy 2.5.4, SciPy 1.18.1, kyber-py 1.2.0, dilithium-py 1.5.1.

## License and citation
Add your chosen licence (for example MIT) and a `CITATION.cff` once the repository is created. Cite the Zenodo DOI after archiving a release.
