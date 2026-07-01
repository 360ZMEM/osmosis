# STDW Focused Safety Constraints Report — 2026-06-28

## Run

Focused matrix:

```bash
bash custom_workflows/run_with_isaac_env.sh workflows/sweep_stdw_safety_pressure.py \
  --profile focused --run \
  --results_root .results/stdw_safety_focused_20260628_224901
```

Diagnostics:

```bash
# positive-x drift, no slow-loop update
bash custom_workflows/run_with_isaac_env.sh workflows/play_stdw_adapt.py \
  --workflow_config workflows/configs/matrix_wave_medium_full.yaml \
  --total_steps 1500 --seed 0 --embodiment asymmetric \
  --auto_drift_router False --drift_router_mode off --enable_micro_probe False \
  --use_stdw True --target_drift 0.05 --slow_loop_interval 99999 \
  --results_root .results/stdw_focused_diagnostics/asym_medium_no_slow_pos/results \
  --artifacts_root .results/stdw_focused_diagnostics/asym_medium_no_slow_pos/artifacts

# manual corrective drift toward smaller xy COM-COB offset
bash custom_workflows/run_with_isaac_env.sh workflows/play_stdw_adapt.py \
  --workflow_config workflows/configs/matrix_wave_medium_full.yaml \
  --total_steps 1500 --seed 0 --embodiment asymmetric \
  --auto_drift_router False --drift_router_mode off --enable_micro_probe False \
  --use_stdw True --target_drift -0.05 --drift_axes 0,1 \
  --results_root .results/stdw_focused_diagnostics/asym_medium_corrective_xy/results \
  --artifacts_root .results/stdw_focused_diagnostics/asym_medium_corrective_xy/artifacts
```

All focused cells returned `0`.

## Focused Matrix Shape

`--profile focused` intentionally replaces the broader multiplicative matrix with 20 cells:

- 8 Lyapunov/asymmetric cells:
  - wave: medium, storm
  - variant: off, default STDW, batch-trust STDW, guarded zero-drift
- 8 controller mismatch cells:
  - embodiment: base, asymmetric
  - mismatch: all PID gains at `0.01`
  - variant: off, default STDW, batch-trust STDW, guarded zero-drift
- 4 flip360 sentinel cells:
  - embodiment: base, asymmetric
  - variant: off, default STDW

This keeps the matrix aligned with the current research questions and avoids multiplying less-informative mismatch types.

## Results

### 1. Batch-trust does not protect asymmetric OPR-off

| case | off | default STDW | batch-trust STDW | guarded zero-drift |
|---|---:|---:|---:|---:|
| asymmetric medium | 0.2263047 | 0.5456321 (+141.1%) | 0.5455823 (+141.1%) | 0.2263049 |
| asymmetric storm | 0.2258533 | 0.5392160 (+138.7%) | 0.5392135 (+138.7%) | 0.2258534 |

`batch_trust` rejected only a few updates:

- medium: accepted `17`, rejected `3`
- storm: accepted `18`, rejected `2`

The batch proxy metrics improved locally while the closed-loop trajectory still degraded. Therefore the first batch-level
acceptance test is useful as a behavior trust region, but it is not a safety proof for this failure mode.

### 2. Guarded zero-drift is the only current OPR-off safety fallback

`lyap_guard_zero` entered `FALLBACK` at step `229` in all normal asymmetric/base mismatch guard cells and matched clean-off
within tolerance. It prevents harm but also removes STDW benefit.

This is acceptable as a conservative safety fallback, not as the final adaptive-control mechanism.

### 3. Severe 1% all-PD mismatch is not the main asymmetric failure source

| embodiment | off | default STDW | batch-trust STDW | guarded zero-drift |
|---|---:|---:|---:|---:|
| base | 0.2257045 | 0.0725009 (-67.9%) | 0.0725009 (-67.9%) | 0.2257045 |
| asymmetric | 0.2263047 | 0.5456321 (+141.1%) | 0.5455823 (+141.1%) | 0.2263049 |

The same STDW update is beneficial on `base` but harmful on `asymmetric`, which points back to embodiment-dependent drift
direction rather than controller-gain mismatch.

### 4. Drift-only and corrective-drift diagnostics decide the next direction

Medium asymmetric diagnostics:

| variant | final_mse | final_mse_after_drift | slow-loop triggers |
|---|---:|---:|---:|
| clean off | 0.2263047 | 0.2704616 | 0 |
| default STDW | 0.5456321 | 0.5176601 | 20 |
| batch-trust STDW | 0.5455823 | 0.5176267 | 20 |
| guarded zero-drift | 0.2263049 | 0.2704618 | 0 |
| `+x` drift, no slow-loop | 0.3147295 | 0.3369155 | 0 |
| corrective `-xy` drift | 0.0723394 | 0.0734644 | 19 |

Interpretation:

- Positive `+x` drift is harmful even without slow-loop update.
- Slow-loop update amplifies the harm from `0.3147` to `0.5456`.
- Corrective `-xy` drift is strongly beneficial and beats off by about `-68%`.

Therefore the next repair cannot be only a Lyapunov sample/update gate. The safety constraint must move to the drift layer.

## Decision Point

The current evidence supports one of these decisions:

1. **Conservative safety claim**:
   Keep `guarded_drift + zero_drift` for OPR-off safety. Claim: STDW will not harm, but may fall back to clean-off behavior.

2. **Adaptive-performance claim**:
   Add a drift-level hard projection/router. The projected drift must only allow COM-COB offset norm reduction, or must choose
   the sign/axes through OPR/router/micro-probe. Claim: STDW can improve asymmetric control when drift direction is physically
   valid.

3. **Training-side claim for flip360**:
   Keep flip360 outside this STDW safety matrix. Current clean-off flip360 is already poor for base, so 360 needs training
   curriculum before STDW safety claims are meaningful.

Recommended next implementation: implement drift-level projection as a hard safety mode, then rerun the 20-cell focused
matrix with `stdw_projected_drift` replacing or supplementing `stdw_batch_trust`.
