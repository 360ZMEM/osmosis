# STDW Safety Pressure Matrix Report — 2026-06-28

## Run

```bash
bash custom_workflows/run_with_isaac_env.sh workflows/sweep_stdw_safety_pressure.py \
  --profile small_hard --run \
  --results_root .results/stdw_safety_pressure_20260628_221237
```

- policy: `logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt`
- cells: 32
- returncode failures: 0
- result files:
  - `.results/stdw_safety_pressure_20260628_221237/pressure_runs.csv`
  - `.results/stdw_safety_pressure_20260628_221237/pressure_summary.json`
  - `.results/stdw_safety_pressure_20260628_221237/pressure_report.md`

## Key Findings

### 1. Asymmetric + OPR off: default STDW still learns the wrong direction

Matched clean off baseline:

| wave | off final_mse | default STDW final_mse | delta |
|---|---:|---:|---:|
| medium | 0.2263047287 | 0.5456321194 | +141.1% |
| storm | 0.2258532758 | 0.5392160070 | +138.7% |

`strict_sample_mask` does not fix this:

| wave | strict final_mse | delta |
|---|---:|---:|
| medium | 0.5470516142 | +141.7% |
| storm | 0.5403822214 | +139.3% |

`guarded_drift + zero_drift` recovers the clean-off behavior:

| wave | guard final_mse | delta |
|---|---:|---:|
| medium | 0.2263047287 | +1.97e-09% |
| storm | 0.2258532737 | -9.19e-07% |

The medium cell is marked `pass_vs_off=False` by the strict `<=` comparator because it is larger than off by about
`4.45e-12` absolute MSE. This is numerical equality for engineering purposes, but the raw artifact intentionally
preserves the strict flag.

### 2. 1% controller mismatch: base benefits from STDW; asymmetric repeats the same failure

For `base`, all three mismatch types behave identically in this run:

| mismatch | off final_mse | default STDW final_mse | delta | guard final_mse |
|---|---:|---:|---:|---:|
| depth `zeta1=0.01` | 0.2257044662 | 0.0725009027 | -67.9% | 0.2257044331 |
| attitude `zeta1=0.01` | 0.2257044662 | 0.0725009027 | -67.9% | 0.2257044331 |
| all `zeta1/zeta2=0.01` | 0.2257044662 | 0.0725009027 | -67.9% | 0.2257044331 |

For `asymmetric`, all three mismatch types also behave identically:

| mismatch | off final_mse | default STDW final_mse | delta | guard final_mse |
|---|---:|---:|---:|---:|
| depth `zeta1=0.01` | 0.2263047287 | 0.5456321194 | +141.1% | 0.2263047287 |
| attitude `zeta1=0.01` | 0.2263047287 | 0.5456321194 | +141.1% | 0.2263047287 |
| all `zeta1/zeta2=0.01` | 0.2263047287 | 0.5456321194 | +141.1% | 0.2263047287 |

There were no nonfinite failures. Reset counts matched clean off (`4`) in all mismatch cells.

Interpretation: the severe controller mismatch itself is not the dominant failure mode in this matrix. The dominant
failure is still the asymmetric hull combined with the fixed positive-x STDW drift direction when OPR/router/probe are off.

### 3. 360-degree flip eval fails the current policy family

| embodiment | off final_mse | default STDW final_mse | delta | guard final_mse | guard delta |
|---|---:|---:|---:|---:|---:|
| base | 3.3666830162 | 5.5661954830 | +65.3% | 16.2594970763 | +383.0% |
| asymmetric | 5.8955368285 | 5.3791569348 | -8.8% | 11.7252511304 | +98.9% |

Even clean-off 360 tracking is very poor compared with normal wave eval. This supports adding `flip360_sine` or a
curriculum variant to training before treating 360 as a deployment capability.

The current `guarded_drift + zero_drift` parameters are not valid for 360-degree reference tracking. They trigger many
blocks (`1027` for base, `1201` for asymmetric) and make final MSE worse. The Lyapunov baseline/window logic is tuned for
small-angle tracking and should not be shared with large-angle flip maneuvers without a separate reference-aware design.

## Conclusions

1. The original concern is confirmed: with OPR/router/probe disabled, default STDW still makes `asymmetric` worse by
   about `+139%` to `+141%`.
2. The existing strict Lyapunov sample mask is insufficient; it behaves like default STDW in the asymmetric failure case.
3. A conservative safety fallback (`guarded_drift + zero_drift`) prevents asymmetric degradation in normal medium/storm
   eval by reverting to matched clean-off behavior.
4. Severe `0.01x` controller mismatch does not create a new failure mode in this matrix; base benefits from STDW, while
   asymmetric repeats the same drift-direction failure.
5. 360-degree flip eval is not ready. It should be treated as a training curriculum requirement, not as an eval-only
   extension of the current A3 stage2 policy.

## Next Decisions

- For paper ablations, keep:
  - clean off
  - default STDW
  - strict Lyapunov sample mask
  - guarded zero-drift fallback
- Do not claim Lyapunov sample masking alone guarantees safety.
- For asymmetric safety claims without OPR, use guarded fallback and report that it is conservative: it prevents harm but
  does not recover the STDW benefit.
- Add a training-stage experiment with `reference_mode=flip360_sine`, preferably as a curriculum from smaller roll/pitch
  amplitudes to `π`.
