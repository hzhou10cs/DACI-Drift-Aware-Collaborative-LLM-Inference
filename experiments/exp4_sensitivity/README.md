# Exp4: Sensitivity + Predictor Accuracy (§5.5)

## Fig A: Sensitivity — 4 line plots (TTLT vs axis)

### 4-A: Window length W
```bash
bash W_sweep.sh   # W ∈ {5, 10, 20, 50, 100}
```
Expected: flat-bottomed U-curve minimum near W=20.

### 4-B: Horizon ceiling H_max
```bash
bash H_sweep.sh   # H_max ∈ {1, 3, 5, 8, 16}
```
Expected: saturation around H_max=8.

### 4-C: Robust slack lambda
```bash
bash lambda_sweep.sh   # λ ∈ {0, 0.5, 1.0, 1.5}
```
Expected: moderate λ reduces #Reconf with marginal TTLT change; large λ → DACI ≈ SDA.

### 4-D: Delta_max (new axis, Eq. 27)
```bash
bash delta_sweep.sh   # Δ_max ∈ {1, 3, 5, L=28}
```
Expected: Δ_max=1 too restrictive, Δ_max=L recovers full DACI; sweet spot 3–5.

## Fig B: Predictor Accuracy — 3 line plots (RMSE vs lead k)

```bash
bash predictor_accuracy.sh   # 20 DACI traces, full logs
# Then: python experiments/aggregate.py predictor outputs/exp4_sensitivity/predictor_accuracy
```

The aggregate script will:
1. For each window r in each trace, read `phi_hat_curr` (k=0) and `phi_true`.
2. Re-run predictor forward rollout k=1..H_max steps from window r's state, compare to actual phi_true at window r+k.
3. Similarly for q_cmp and q_mem ground truth vs AR(1) forecast.
4. Plot RMSE(k) curves: (a) Thermal Kalman vs persistence vs EWMA, (b) AR(1) cmp vs persistence vs EWMA, (c) AR(1) mem vs persistence vs EWMA.

## Run all Fig A in parallel
```bash
# Run each sweep in a separate shell so parallel jobs don't share a seed
bash W_sweep.sh & bash H_sweep.sh & bash lambda_sweep.sh & bash delta_sweep.sh & wait
```

## Output
```
outputs/exp4_sensitivity/
  W_sweep/{W_5, W_10, ...}/summary.csv
  H_sweep/{H_1, H_3, ...}/summary.csv
  lambda_sweep/{lambda_0, lambda_0_5, ...}/summary.csv
  delta_sweep/{delta_1, delta_3, ...}/summary.csv
  predictor_accuracy/{seed_42, seed_43, ...}/traces/
```
