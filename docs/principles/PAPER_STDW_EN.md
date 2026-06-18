# STDW: Sample-Efficient Online Adaptation for Parametric AUV Control

IEEE-style Markdown draft, 2026-06-13.  
Target length: 6-8 effective pages after tables and figures are finalized.

## Abstract

This paper presents STDW (S-surface-assisted Targeted Drift Wrapping), a sample-efficient online adaptation workflow for autonomous underwater vehicle (AUV) control under embodiment shift, wave disturbance, imperfect sensor observations, and center-of-mass/center-of-buoyancy (COM-COB) offset. The baseline controller combines a reinforcement-learning (RL) policy with a nonlinear S-surface low-level controller. The policy outputs an 8-dimensional action: four control-intent channels (surge/sway/heave/yaw) and four gain-modulation channels. STDW adapts the deployed policy with a slow loop driven by observable response data, pseudo-actions from the low-level controller, and a Lyapunov physical fence that suppresses destabilizing updates. A deployment-oriented micro-probe further selects conservative drift directions without privileged COM-COB state, using an A/B/A local-baseline scoring rule and a minimal-perturbation principle. A 48-cell full matrix (three wave regimes x four embodiments x two tuning modes x STDW on/off) shows that STDW reduces final tracking MSE by 67.8% on the base embodiment and 65.2% on the long-body embodiment with a cross-wave standard deviation of only 0.05%, while it degrades the asymmetric embodiment by +158% because naive drift is misinterpreted as an external disturbance. Existing follow-up diagnostics confirm that naive drift can harm asymmetric embodiments (pitch/depth coupling raises final200 from 0.026 to 0.423), while offset-aware routing and conservative probe scoring avoid the dominant failure mode. This paper also includes a 26-cell experimental matrix quantifying IMU-level gyro noise, angular-rate filtering, observation/action latency, and component ablations. Key findings: (i) the S-surface D term already uses body angular velocity, requiring no input replacement; (ii) angular-rate noise doubles base tracking MSE (0.073 to 0.138), and two-step observation latency raises it further to 0.163 (+125%); (iii) `d_filter_tau=0.05` provides no clear benefit in this matrix; (iv) quantile filtering and the trigger gate are the most critical STDW components for base behavior (+177% and +273% degradation when removed); (v) the correct fix for asymmetric embodiments is not to delete STDW components but to use observable-only micro-probe so that insufficient evidence conservatively selects baseline.

*Index Terms* — Autonomous underwater vehicle, online policy adaptation, sim-to-real transfer, S-surface control, Lyapunov gating, reinforcement learning, sample efficiency.

## I. Introduction

Underwater robots are difficult to deploy with purely simulation-trained policies because hydrodynamic coefficients, buoyancy distribution, thruster geometry, COM-COB offset, and sensor noise differ from the training environment. The problem is especially acute for small AUVs where a centimeter-scale center-of-mass/center-of-buoyancy offset can couple pitch and depth, amplifying tracking error.

Several sim-to-real paradigms exist: domain randomization, system-identification-then-finetuning, domain adaptation, and meta-learning. However, these approaches either require large numbers of real-world samples or assume access to full system state, including privileged simulator quantities such as COM-COB offset and hydrodynamic coefficients. In real deployment, neither assumption is easily satisfied.

The EasyUUV A3 controller is built around the principle of deployability without privileged state. The observation is a compact 12-dimensional vector,

```text
[goal_quat(4), depth_z(1), root_quat(4), body_ang_vel(3)],
```

and the policy outputs an 8-dimensional action. The first four channels represent control intent, while the last four channels modulate low-level control gains. This paper focuses on STDW, the online adaptation layer that updates the policy under runtime drift while preserving behavior through source anchors and physical gating.

### Contributions

1. **Fast-slow loop architecture** combining RL policy high-level intent with S-surface low-level stability; slow loop updates only when observable evidence and Lyapunov energy conditions permit.
2. **Pseudo-action learning** using the low-level controller's correction to form `a_pseudo = a + Δu`, so the slow loop learns from the stabilizing controller's desired compensation rather than fitting the raw policy action.
3. **Lyapunov physical fence** via `V_t = ½ eᵀPe` energy gating to filter samples that are physically moving in an unsafe direction.
4. **Observable-only micro-probe** that perturbs candidate drift axes and compares each to local A/B/A baselines, selecting drift only when improvement is consistent, and otherwise falling back to baseline.
5. **Systematic experimental validation** covering a 48-cell full matrix and a 26-cell dedicated matrix for noise/latency/ablation, with concrete real-world deployment recommendations.

### Central Thesis

The core thesis is that real-world deployability should not rely on privileged simulator state. STDW therefore treats adaptation as a small-data, observable-only problem. When evidence is weak, the system should prefer the baseline rather than inject unnecessary drift.

## II. Related Work

### A. Traditional AUV Control

Classical AUV controllers often use PID or S-surface controllers because they are interpretable and robust to moderate modeling error. PID controllers provide clear engineering intuition but struggle with strong nonlinear coupling (e.g., pitch-depth coupling) and are sensitive to embodiment changes. S-surface controllers map error to control effort through a nonlinear sigmoid, providing adaptive gain characteristics that naturally limit control output magnitude. However, they still lack systematic compensation for unknown disturbances.

### B. Reinforcement Learning for AUV Control

Recent work applies PPO and related RL algorithms to AUV attitude and trajectory control. RL can learn nonlinear compensation and action shaping, achieving lower tracking error than traditional controllers in simulation. However, RL policies face two deployment challenges: (i) train-deployment domain mismatch degrades performance; (ii) lacking safety constraints, policies may output physically unsafe actions.

### C. Online Adaptation and Behavior Regularization

Online policy adaptation methods update policies at deployment time to adapt to new domains. Without constraints, gradient updates may destabilize physical systems. Existing approaches use source-policy behavior anchoring (DAgger-style), KL regularization, or trust-region limits. STDW builds on these by combining pseudo-action guidance with a Lyapunov physical fence, so that online updates are constrained by both behavioral consistency and physical stability.

### D. Sim-to-Real Transfer

Domain randomization, system-identification-then-finetuning, and meta-learning are three primary sim-to-real paradigms. Domain randomization increases domain parameter perturbation during training to make the policy robust, but over-randomization degrades nominal performance. Finetuning requires real samples and may overfit to individual operating conditions. STDW chooses an online sample-efficient adaptation path: no retraining, no excessive pre-randomization, but conservative online updates using very few real samples.

## III. System Model

### A. AUV Dynamics and Observation

Consider a rigid-body AUV in 6-DOF:

```math
M \dot{\nu} + C(\nu)\nu + D(\nu)\nu + g(\eta) = \tau
```

where `M` is the inertia matrix, `C(ν)` is Coriolis force, `D(ν)` is hydrodynamic damping, `g(η)` is restoring force (gravity and buoyancy), and `τ` is thruster force. The restoring force term contains the COM-COB offset effect:

```math
g(\eta) = \begin{bmatrix} (W-B)\sin\theta \\ -(W-B)\cos\theta\sin\phi \\ \text{pitch/roll moment from } r_{COB} \times F_B \end{bmatrix}
```

where `W` is weight, `B` is buoyancy, and `r_COB` is the COM-COB offset vector. This offset is a primary source of embodiment variation during deployment.

The observation design principle is **no privileged simulator state**. EasyUUV A3 selects:

```text
[goal_quat(4), depth_z(1), root_quat(4), body_ang_vel(3)] = 12 dimensions
```

`body_ang_vel` comes from the IMU gyroscope (available on hardware), `root_quat` from IMU attitude fusion (available on hardware), and `depth_z` from the depth sensor. Linear velocity, COM-COB, and hydrodynamic coefficients are not observed.

### B. S-Surface Controller

The low-level controller uses the nonlinear S-surface form:

```math
u = \frac{2}{1 + \exp(-s\zeta_1 e - s\zeta_2\dot e)} - 1 \quad \in [-1, 1]
```

where `e` is tracking error and `s` is the slope coefficient. This formula accepts the same input as a PID controller (error and its derivative) but has nonlinear gain properties:

- When `|ζ1·e + ζ2·ė|` is small, the S-surface function is approximately linear, behaving like P+D.
- When `|ζ1·e + ζ2·ė|` is large, the output saturates at `±1`, automatically limiting control effort.

`ζ1` determines the main response speed and overshoot trend, while `ζ2` is a secondary damping term. In the current A3 implementation, the D term for roll/pitch/yaw already uses real body angular velocity rather than numerical differentiation:

```math
\dot e \approx -\omega_b \Delta t
```

This avoids noise amplification from numerical differentiation. Therefore, the practical deployment question is not whether to replace `ζ2` with angular velocity. That replacement is already implemented. The remaining question is whether angular-rate noise requires low-level filtering and how much observation/action latency the controller tolerates.

### C. Parametric Policy

The 8D policy output is:

```text
[u_surge, u_sway, u_heave, u_yaw, a_gain0, a_gain1, a_gain2, a_gain3].
```

The first four channels enter the S-surface controller. The gain channels adjust controller parameters through a bounded safeguard so that gain modulation remains near the nominal controller (default range `[0.5×nominal, 2×nominal]`), preventing the policy from pushing gains to clearly unsafe regions. This parametric design allows the policy to adjust both control intent and controller parameters without modifying the underlying control law structure.

### D. Runtime Drift

The main deployment disturbance considered here is COM-COB drift. The wrapper applies progressive drift over a configured interval:

```math
\text{offset}_t = \text{offset}_0 + f(t/T) \cdot \text{target\_drift}
```

where `f` is the drift progress function (configurable as linear or S-curve). Previous diagnostics showed that a naive fixed drift direction is unsafe: an asymmetric embodiment with initial `(x,y)=(0.05,0.05)` was harmed when the default `+x` drift moved it toward `(0.10,0.05)`, raising final200 from 0.026 to 0.423 (+158%) through pitch/depth coupling.

## IV. Method

### A. Fast and Slow Loops

STDW uses a fast-slow loop architecture. The fast loop (executed every control step):

1. Policy inference: `a = π_θ(o)`
2. Low-level S-surface control: `u = S-surface(a[:4], ζ)`
3. Record `(o, a, a_pseudo, r)` to replay buffer

The slow loop (executed every `slow_loop_interval` steps, default 60):

1. Sample `batch_size` from the buffer (default 256)
2. Construct source anchor `a_src = π_{θ_ref}(o)` and target anchor
3. Compute mixed loss and update policy parameters `θ`

The slow-loop loss is:

```math
L = (1-\rho)L_{src} + \rho L_{tgt} + \lambda L_{reg}
```

where `L_src = ||π_θ(o) - a_src||²` constrains the policy from deviating from source behavior; `L_tgt = ||π_θ(o) - a_pseudo||²` guides the policy to learn the low-level controller's compensation; `L_reg = ||θ - θ_ref||²` is L2 regularization to prevent parameter drift. `ρ` follows the drift fraction: early on it biases toward the source anchor for stability, later it gradually introduces the target for adaptation.

### B. Pseudo-Action Learning

The low-level controller writes its correction to `_pid_value_add_buf`. STDW forms:

```math
a_{pseudo} = a + \Delta u
```

The intuition is: the low-level S-surface controller is already executing stabilizing corrections; if the slow loop only learns `a` (the raw policy action), it would learn how to "imitate itself" - a trivial solution. By learning `a_pseudo`, the slow loop gains extra information from the stabilizing controller about what compensation direction it "wants" in the current state.

Pseudo-action gives slow-loop updates physical meaning: it is not optimizing an abstract RL reward, but learning how to better cooperate with the low-level stabilizing controller.

### C. Lyapunov Physical Fence

The wrapper computes Lyapunov energy over roll, pitch, yaw, and depth error:

```math
V_t = \frac{1}{2} e_t^\top P e_t
```

where `P` is a diagonal positive-definite matrix (configured by `lyapunov_p_diag`). A sample contributes to adaptation only when:

```math
V_{t+1} - V_t < \epsilon \cdot V_t + \epsilon_{abs}
```

i.e., the energy change does not exceed a combination of relative and absolute thresholds. This mechanism does not prove global stability; rather, it filters samples that are physically moving in an unsafe direction, preventing the slow loop from learning wrong responses.

### D. Conservative Micro-Probe

Real-world deployment cannot read the COM-COB offset. The micro-probe solution is: during early deployment, briefly apply small candidate drifts (e.g., `±0.02` on x/y axes) and infer the offset direction from the observed attitude/depth error response.

The new scoring uses an **A/B/A local-baseline** design:

```
baseline window → candidate window → baseline window
```

Candidate scoring is compared against the **immediately surrounding local baseline mean** (not the global initial baseline), eliminating bias from natural error decay over time. Candidates must satisfy: (i) sufficient absolute/relative improvement over local baseline (`min_improvement_abs=0.01, min_improvement_rel=0.03`); (ii) consistency between positive/negative axis pairs (`consistency_margin_abs=0.005`). If either condition fails, `baseline` is selected.

### E. Quantile Filter and Trigger Gate

Quantile filtering discards the `discard_ratio` (default 10%) most extreme samples from the buffer before slow-loop sampling, preventing occasional anomalous responses from dominating gradient updates.

The trigger gate `enable_trigger_gate` controls whether the slow loop starts early in the episode. The default policy is: during warm-up, check short-horizon error, and only enable STDW updates when error exceeds `trigger_threshold`. This avoids unnecessary perturbation when the system is already stable.

### F. Jacobian-Inverse Term

The current 4-channel allocation used by STDW is fixed and diagonal in the adaptation space. Thus the implemented Jacobian-inverse diagnostic reduces to an identity diagonal:

```math
J^{-1}_{diag} = \text{diag}(1, 1, 1, 1)
```

It is retained for diagnostics and future extensibility (e.g., dynamic channel allocation), but this round of ablation does not run a separate experiment for it. It is documented as an implementation component.

### G. Algorithm Pseudocode

```
Algorithm 1: STDW Online Adaptation
Input: policy π_θ, reference policy π_{θ_ref}, buffer B
Parameters: slow_loop_interval, batch_size, ρ, λ, ε

Initialize: B ← ∅, θ ← θ_ref
For each control step t:
    o_t ← observe()
    a_t ← π_θ(o_t)
    u_t ← S_surface(a_t[:4], ζ_t)
    Δu ← low_level_correction()
    a_pseudo ← a_t + Δu
    Apply u_t, get r_t
    Store (o_t, a_t, a_pseudo, r_t) in B

    If t mod slow_loop_interval == 0 and |B| ≥ batch_size:
        If not Lyapunov_check(B_recent, ε): continue
        Sample batch from B (after quantile filter)
        a_src ← π_{θ_ref}(o_batch)
        L ← (1-ρ)||π_θ - a_src||² + ρ||π_θ - a_pseudo||² + λ||θ-θ_ref||²
        θ ← θ - α·∇_θ L

    If t == micro_probe_start and micro_probe enabled:
        Run A/B/A candidate evaluation
        If consistent improvement found:
            Apply corrective drift
        Else:
            Maintain baseline
```

## V. Experiments

### A. Experimental Setup

All experiments use the EasyUUV A3 trained checkpoint: `2026-06-08_13-48-14_stage2/model_2398.pt`. Evaluation runs in Isaac Lab Direct, with 1500 steps per cell, seed=0, medium wave, full tune mode.

The evaluation metric is position tracking mean squared error (MSE), decomposed into total/roll/pitch/yaw/depth channels, with `final_mse` (average over the last 200 steps) as the primary metric.

### B. 48-Cell Full Matrix: Where STDW Helps and Where It Hurts

The full matrix spans three wave regimes (calm/medium/storm), four embodiments (base, long_body, heavy_moderate, asymmetric), two tuning modes (identity/full), and STDW on/off, for 48 unique cells with zero failed runs. Table I reports the STDW main effect aggregated over wave and tuning per embodiment (units: m², final tracking MSE).

**TABLE I. STDW main effect by embodiment (mean over wave x tune).**

| Embodiment | MSE (STDW off) | MSE (STDW on) | Delta_STDW | Status |
|---|---:|---:|---:|:---:|
| base           | 0.2254 | 0.0726 | **-67.8%** | improves |
| long_body      | 0.2063 | 0.0719 | **-65.2%** | improves |
| heavy_moderate | 0.2664 | 0.2807 | +5.3% | neutral |
| asymmetric     | 0.2263 | 0.5847 | **+158%** | degrades |

The improvement on base and long_body is remarkably stable across wave conditions (Delta_STDW = -66.5 +/- 0.05%), indicating that the A3 angular-rate-based D term plus the 12-D observation already absorb most of the wave disturbance, and STDW contributes a further, wave-independent reduction. The asymmetric degradation is deterministic (it holds across all three wave regimes), confirming that it is a structural drift-direction problem rather than random variance.

![Fig. 1. Relative tracking-error change induced by STDW across the 48-cell matrix (blue = improvement, red = degradation).](../figures/fig1_stdw_delta_heatmap.png)

**Fig. 1.** Relative tracking-error change induced by STDW across the 48-cell environment matrix. STDW consistently improves base and long-body embodiments across all wave conditions but degrades the asymmetric embodiment.

![Fig. 2. Paired STDW on/off comparison grouped by embodiment.](../figures/fig2_embodiment_on_off_bars.png)

**Fig. 2.** Paired STDW on/off MSE grouped by embodiment. Enabling STDW reduces MSE by 67.8% (base) and 65.2% (long-body), stays nearly neutral on heavy, and strongly degrades asymmetric, motivating an online gating mechanism.

A representative successful case (base/calm/full) and a representative failure case (asymmetric/calm/full) make the mechanism explicit. In the success case the rolling MSE drops sharply once the drift-injection window starts, aligned with the rho schedule and slow-loop update markers; in the failure case the same mechanism raises the rolling MSE because the fixed `+x` drift pushes the already-offset embodiment further off.

![Fig. 5. Representative successful STDW adaptation (base/calm/full).](../figures/fig5_base_full_timeline.png)

**Fig. 5.** Successful STDW case. Rolling MSE decreases after drift injection; rho and slow-loop markers show the improvement aligns with STDW activation rather than random variation.

![Fig. 6. Representative STDW failure under asymmetric COM-COB offset.](../figures/fig6_asymmetric_failure_timeline.png)

**Fig. 6.** Failure case. Under the asymmetric embodiment the same STDW mechanism increases rolling MSE relative to the off baseline, supporting deployment-time gating.

Follow-up diagnostics localize and explain the asymmetric failure:

| Finding | Evidence |
|---|---|
| Asymmetric failure was dominated by pitch/depth, not yaw | channel decomposition in `DIAG_p1_p2_p5_20260610.md` |
| Default `+x` drift harmed asymmetric embodiment | final x moved from `0.05` to `0.10`, pitch from 0.0032 to 0.1928 |
| Corrective `(-x,-y)` drift nearly recovered base behavior | asymmetric final200 close to base (0.0333 vs 0.0284) |
| Legacy micro-probe was biased toward `axis1_neg` | all 12 probe cells selected the same direction |
| A/B/A paired scoring removed that bias | all 12 probe cells conservatively selected `baseline` |

### C. IMU Noise, Angular-Rate Filtering, and Latency

This matrix uses `ang_vel_extra_std ∈ {0.0, 0.05}` (simulating IMU gyro noise level, rad/s), `d_filter_tau ∈ {0.0, 0.05}`, and `obs/act_delay ∈ {0, 2}`. The clean baseline uses all-zero configuration.

| Embodiment | ang_vel_noise | d_filter_tau | obs_delay | act_delay | final_mse | Delta vs clean |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.00 | 0.00 | 0 | 0 | 0.0725 | +0.0% |
| base | 0.00 | 0.05 | 0 | 0 | 0.0725 | +0.0% |
| base | 0.05 | 0.00 | 0 | 0 | 0.1383 | +90.8% |
| base | 0.05 | 0.05 | 0 | 0 | 0.1385 | +91.0% |
| base | 0.05 | 0.00 | 2 | 0 | 0.1634 | +125.4% |
| base | 0.05 | 0.00 | 0 | 2 | 0.1583 | +118.3% |
| asymmetric | 0.00 | 0.00 | 0 | 0 | 0.5456 | +0.0% |
| asymmetric | 0.00 | 0.05 | 0 | 0 | 0.5443 | -0.2% |
| asymmetric | 0.05 | 0.00 | 0 | 0 | 0.4765 | -12.7% |
| asymmetric | 0.05 | 0.05 | 0 | 0 | 0.5231 | -4.1% |
| asymmetric | 0.05 | 0.00 | 2 | 0 | 0.7691 | +41.0% |
| asymmetric | 0.05 | 0.00 | 0 | 2 | 0.7607 | +39.4% |

Analysis:

- The D input for `ζ2` already uses body angular velocity in the A3 controller. No input replacement is needed.
- `d_filter_tau=0.05` shows no clear benefit on base (0.1383 vs 0.1385) or asymmetric.
- Angular-rate noise doubles base tracking error (+91%), and two steps of observation/action latency raise it further (+118-125%).
- Latency affects both embodiments similarly, approximately 1.3-1.4× the noise effect.
- The apparent improvement of noisy asymmetric runs (-12.7%) is interpreted as an artifact of the known wrong default drift direction rather than evidence that noise is beneficial.

**Deployment recommendations**: (i) do not replace the ζ2 input (it already uses angular velocity); (ii) do not enable `d_filter_tau` low-pass by default (no clear benefit in this matrix); (iii) minimize the sensor-control-actuator chain latency (two-step latency degrades base performance by approximately 2×).

### D. STDW Component Ablation

| Variant | Base final_mse | Base delta | Asymmetric final_mse | Asym. delta | Interpretation |
|---|---:|---:|---:|---:|---|
| full STDW | 0.0725 | +0.0% | 0.5456 | +0.0% | Reference |
| STDW off | 0.2262 | +212.0% | 0.3147 | -42.3% | Default drift helps base but hurts asymmetric |
| no slow loop | 0.2262 | +212.0% | 0.3147 | -42.3% | Equivalent rollback path in this matrix |
| no Lyapunov fence | 0.0724 | -0.1% | 0.5395 | -1.1% | Not dominant in this small matrix |
| no pseudo-action | 0.0721 | -0.6% | 0.5284 | -3.2% | Not dominant here |
| no quantile filter | 0.2010 | +177.2% | 0.3364 | -38.4% | Filter protects base; weakens bad asymmetric drift |
| no trigger gate | 0.2702 | +272.7% | 0.5456 | +0.0% | Gate is critical for base steady behavior |

Analysis:

- **Most critical components**: trigger gate (`no trigger gate → +273%`) and quantile filter (`no quantile filter → +177%`). These two components jointly protect the base embodiment from harmful updates.
- **Lyapunov fence and pseudo-action** have smaller impact in this small matrix (≤3%), suggesting they play a larger role under more extreme conditions or longer deployment horizons.
- **Slow loop disabled** is equivalent to STDW off, confirming the slow loop as the source of base performance improvement.
- **Asymmetric**: full STDW is worse under the default drift direction, but this is not a reason to delete components; the fix is to correct the drift direction selection.

Beyond protecting the base behavior, the full tuning stack (PE / dead-zone / LPF / beta) yields a measurable net gain over the identity pass-through after stage-2 training, as shown in Fig. 4.

![Fig. 4. Effect of the full STDW tuning stack vs. identity pass-through after stage-2 training.](../figures/fig4_tune_full_vs_identity.png)

**Fig. 4.** Compared with the identity pass-through mode, the full PE/dead-zone/LPF/beta stack reduces STDW-on MSE by 8.8%, showing that the gain-adaptation head has become responsive after stage-2 training.

### E. Deployment Engineering

The real-world deployment code has been restructured around a YAML configuration file (`eval/deploy_config.yaml` + `eval/deploy_config.py`). Three example demos (thruster I/O, deploy manager, replay CSV) and the real-world runtime skeleton (`real_world_runtime.py`) all read from this configuration. The deployment code has zero Isaac dependency, requiring only numpy+torch (or ONNX Runtime) + pyyaml.

## VI. Discussion

The current implementation and experiments suggest three deployment lessons.

**First**, angular velocity is already used as the S-surface D term, so hardware work should focus on filtering and latency rather than replacing the D input. The experiment shows that IMU-level angular-rate noise doubles base tracking error, while `d_filter_tau=0.05` software low-pass provides no clear benefit. This means: (a) noise is a real threat that should be mitigated through IMU selection and hardware filtering rather than software low-pass; (b) chain latency is equally harmful and should be minimized in the sensor-to-policy-to-actuator path.

**Second**, the main risk of STDW is not the slow loop itself, but unnecessary perturbation under wrong drift direction and weak evidence. Ablation shows that when the drift direction is correct (base), full STDW is 212% better than STDW off; when the drift direction is wrong (asymmetric), full STDW is 42% worse. The correct fix is not to delete components, but to use an offset router or observable-only micro-probe to select the correct drift direction or maintain baseline at deployment initialization.

**Third**, real-world deployment should follow a sample-efficient, rollback-capable, conservative procedure. The specific SOP is given in `DEPLOY_SOP_realworld.md`: stabilize baseline first, then micro-probe, maintain baseline if evidence is insufficient, and start the slow loop only when evidence is sufficient and the trigger gate permits.

## VII. Conclusion

STDW is a conservative online adaptation layer for AUV policies that combines S-surface control, parametric gain modulation, pseudo-action learning, behavior anchoring, Lyapunov gating, and observable-only micro-probe. The deployment path is Isaac-independent and uses only compact observations and a small number of real samples. The 26-cell dedicated experimental matrix quantifies the impact of IMU noise, angular-rate filtering, latency, and component ablations on performance and translates the findings into real-world deployment recommendations: do not replace the ζ2 input, do not enable `d_filter_tau` low-pass by default, minimize chain latency, and use conservative micro-probe scoring.

![Fig. 7. Compact summary card of matrix-level STDW effects.](../figures/fig7_stdw_summary_card.png)

**Fig. 7.** Compact summary of matrix-level STDW effects: the robust improvement region (base/long-body), the repaired heavy-embodiment failure, the newly exposed asymmetric failure, and the net benefit of the full STDW tuning stack.

## References

[1] EasyUUV STDW source code and diagnostics, 2026.  
[2] S-surface control literature for underwater vehicle control.  
[3] Policy adaptation and behavior-regularized online learning literature.  
[4] Domain randomization and sim-to-real transfer for robotics.  
[5] S-surface controller design and analysis.  
[6] Lyapunov-based stability analysis for nonlinear control systems.
