"""Focused safety-pressure matrix for STDW Lyapunov gates.

This driver intentionally avoids the old multiplicative full matrix.  It runs
matched off/on/guarded comparisons for three questions:

* asymmetric hull with OPR/router/probe disabled,
* 1% initial-controller mismatch,
* roll/pitch 360-degree flip reference tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "custom_workflows" / "run_with_isaac_env.sh"
PLAY = REPO_ROOT / "workflows" / "play_stdw_adapt.py"
DEFAULT_POLICY = REPO_ROOT / "logs/rsl_rl/easyuuv_parametric/2026-06-08_13-48-14_stage2/model_2398.pt"

POLICY_PARTS = ("logs", "rsl_rl")
MEDIUM_CFG = "workflows/configs/matrix_wave_medium_full.yaml"
STORM_CFG = "workflows/configs/matrix_wave_storm_full.yaml"
FLIP360_CFG = "workflows/configs/pressure_flip360_medium_full.yaml"
MISMATCH_AMP_CFG = "workflows/configs/pressure_mismatch_amplified.yaml"
BOUNDARY_RESIDUAL_CFG = "workflows/configs/pressure_boundary_residual.yaml"
BOUNDARY_FULL_CFG = "workflows/configs/pressure_boundary_full.yaml"
ESUOT_CFG = "workflows/configs/pressure_esuot.yaml"
EMBODIMENT_UUV6_CFG = "workflows/configs/embodiment_uuv6.yaml"
EMBODIMENT_UUV4_CFG = "workflows/configs/embodiment_uuv4.yaml"
EMBODIMENT_UUV6_ANGLED_CFG = "workflows/configs/embodiment_uuv6_angled.yaml"
EMBODIMENT_UUV4_ANGLED_CFG = "workflows/configs/embodiment_uuv4_angled.yaml"
EMBODIMENT_SUBMERGE_FLIP360_CFG = "workflows/configs/embodiment_submerge_flip360.yaml"


def _bool(value: bool) -> str:
    return "True" if value else "False"


def _policy_to_load_args(policy_path: Path) -> tuple[str, str, str]:
    policy_path = policy_path.resolve()
    parts = policy_path.parts
    for idx in range(0, len(parts) - 4):
        if parts[idx : idx + 2] == POLICY_PARTS:
            logs_root = Path(*parts[: idx + 2])
            experiment_name = parts[idx + 2]
            load_run = parts[idx + 3]
            checkpoint = parts[idx + 4]
            if experiment_name != "easyuuv_parametric":
                raise ValueError(f"expected easyuuv_parametric policy, got {experiment_name!r}")
            return str(logs_root), load_run, checkpoint
    raise ValueError(f"policy_path must contain logs/rsl_rl/<experiment>/<run>/<ckpt>: {policy_path}")


def _variant_extra(variant: str) -> list[str]:
    if variant == "off_clean":
        return ["--use_stdw", "False", "--target_drift", "0.0"]
    if variant == "stdw_default":
        return ["--use_stdw", "True", "--target_drift", "0.05"]
    if variant == "stdw_batch_trust":
        return [
            "--use_stdw", "True",
            "--target_drift", "0.05",
            "--stdw_update_acceptance", "batch_trust",
            "--stdw_max_behavior_mse", "0.0001",
            "--stdw_max_action_delta_mse", "0.001",
            "--stdw_max_target_mse_increase", "0.0001",
            "--stdw_min_effective_batch_frac", "0.2",
        ]
    if variant == "lyap_strict":
        return [
            "--use_stdw", "True",
            "--target_drift", "0.05",
            "--lyapunov_gate_mode", "strict_sample_mask",
            "--lyapunov_abs_margin", "0.005",
            "--lyapunov_rel_margin", "0.05",
        ]
    if variant == "lyap_guard_zero":
        return [
            "--use_stdw", "True",
            "--target_drift", "0.05",
            "--lyapunov_gate_mode", "guarded_drift",
            "--lyapunov_guard_action", "zero_drift",
            "--lyapunov_abs_margin", "0.01",
            "--lyapunov_rel_margin", "0.25",
            "--lyapunov_window_steps", "60",
            "--lyapunov_min_pass_rate", "0.25",
            "--lyapunov_guard_confirm_steps", "30",
            "--lyapunov_guard_recover_steps", "30",
        ]
    raise ValueError(f"unknown variant: {variant}")


def _case(
    *,
    group: str,
    name: str,
    wave: str,
    config: str,
    embodiment: str,
    variant: str,
    reference_mode: str = "sine_sweep",
    pid_multipliers: dict[str, float] | None = None,
    ctrl_mismatch: dict[str, Any] | None = None,
    lyapunov_v_mode: str = "",
    lyapunov_q_diag: str = "",
    lyapunov_decay_alpha: str = "",
    boundary_effect: str = "",
    domain_adapt_backend: str = "",
) -> dict[str, Any]:
    return {
        "group": group,
        "name": name,
        "wave": wave,
        "config": config,
        "embodiment": embodiment,
        "variant": variant,
        "reference_mode": reference_mode,
        "pid_multipliers": pid_multipliers or {},
        "ctrl_mismatch": ctrl_mismatch or {},
        "lyapunov_v_mode": lyapunov_v_mode,
        "lyapunov_q_diag": lyapunov_q_diag,
        "lyapunov_decay_alpha": lyapunov_decay_alpha,
        "boundary_effect": boundary_effect,
        "domain_adapt_backend": domain_adapt_backend,
    }


def build_cases(profile: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if profile == "ordinary_only":
        return [
            _case(
                group="ordinary_eval",
                name=f"ordinary_{embodiment}_off_clean",
                wave="medium",
                config=MEDIUM_CFG,
                embodiment=embodiment,
                variant="off_clean",
            )
            for embodiment in ("base", "asymmetric")
        ]
    if profile == "flip360_only":
        return [
            _case(
                group="flip360_eval",
                name=f"flip360_{embodiment}_off_clean",
                wave="medium",
                config=FLIP360_CFG,
                embodiment=embodiment,
                variant="off_clean",
                reference_mode="flip360_sine",
            )
            for embodiment in ("base", "asymmetric")
        ]
    if profile == "embodiment_zoo":
        zoo_specs: list[tuple[str, str, str]] = [
            ("uuv6", EMBODIMENT_UUV6_CFG, "full"),
            ("uuv4", EMBODIMENT_UUV4_CFG, "underactuated"),
            ("uuv6_angled", EMBODIMENT_UUV6_ANGLED_CFG, "angled_full"),
            ("uuv4_angled", EMBODIMENT_UUV4_ANGLED_CFG, "angled_underactuated"),
        ]
        cases.append(_case(
            group="embodiment_zoo",
            name="embodiment_base_submerge_flip360_off_clean",
            wave="medium",
            config=EMBODIMENT_SUBMERGE_FLIP360_CFG,
            embodiment="base",
            variant="off_clean",
            reference_mode="flip360_sine",
        ))
        for embodiment, config, tag in zoo_specs:
            cases.append(_case(
                group="embodiment_zoo",
                name=f"embodiment_{embodiment}_{tag}_off_clean",
                wave="medium",
                config=config,
                embodiment=embodiment,
                variant="off_clean",
                reference_mode="flip360_sine",
            ))
            cases.append(_case(
                group="embodiment_zoo",
                name=f"embodiment_{embodiment}_{tag}_stdw_default",
                wave="medium",
                config=config,
                embodiment=embodiment,
                variant="stdw_default",
                reference_mode="flip360_sine",
            ))
        return cases
    if profile == "mismatch_amplified":
        # M3: 验证突破 67% 天花板。CM-A pid_gain（饱和基线）对照 CM-B actuator_scale /
        # CM-C s_surface_struct（作用于非饱和段/控制律结构）。asym + OPR off。
        # off_clean 与 stdw_default 共享同一失配 spec，故 _match_key 可配对出 STDW 相对增益。
        all_pd_0p01 = {
            "roll_zeta1": 0.01, "roll_zeta2": 0.01,
            "pitch_zeta1": 0.01, "pitch_zeta2": 0.01,
            "yaw_zeta1": 0.01, "yaw_zeta2": 0.01,
            "depth_zeta1": 0.01, "depth_zeta2": 0.01,
        }
        # (name, pid_multipliers, ctrl_mismatch)
        amplified_specs: list[tuple[str, dict[str, float], dict[str, Any]]] = [
            ("cmA_pid_gain_0p01", all_pd_0p01, {}),
            ("cmB_thrust_0p50", {}, {"mode": "actuator_scale", "thrust_scale": 0.5}),
            ("cmB_thrust_0p20", {}, {"mode": "actuator_scale", "thrust_scale": 0.2}),
            ("cmB_thrust_0p05", {}, {"mode": "actuator_scale", "thrust_scale": 0.05}),
            ("cmC_sratio_0p50_add0", {}, {"mode": "s_surface_struct", "s_ratio_scale": 0.5, "add_scale": 0.0}),
            ("cmC_sratio_0p20_add0", {}, {"mode": "s_surface_struct", "s_ratio_scale": 0.2, "add_scale": 0.0}),
        ]
        for spec_name, pid_mult, mismatch in amplified_specs:
            for variant in ("off_clean", "stdw_default"):
                cases.append(_case(
                    group="controller_misset_amplified",
                    name=f"misset_amp_asym_{spec_name}_{variant}",
                    wave="medium",
                    config=MISMATCH_AMP_CFG,
                    embodiment="asymmetric",
                    variant=variant,
                    pid_multipliers=pid_mult,
                    ctrl_mismatch=mismatch,
                ))
        return cases

    if profile == "hard_constraints_v2":
        # M6 §7: four focused groups wiring M1 V-ablation, M3 amplified misset,
        # M4 boundary-effect pressure, M5 E-SUOT vs OPR. Each group pairs an
        # off_clean baseline against the on-variants sharing the same _match_key
        # environment (group/wave/embodiment/reference_mode/pid/mismatch/boundary),
        # so _annotate_passes can compute delta_vs_off_pct per mechanism.

        # G1 lyapunov_v_ablation: four V definitions, asym + OPR off.
        # off_clean pairs against every stdw variant (they share the same env).
        cases.append(_case(
            group="lyapunov_v_ablation",
            name="lyap_v_asym_off_clean",
            wave="medium",
            config=MEDIUM_CFG,
            embodiment="asymmetric",
            variant="off_clean",
        ))
        v_specs: list[tuple[str, str, str]] = [
            ("pose_quadratic", "", ""),
            ("so3_consistent", "", ""),
            ("energy_with_rate", "0.1,0.1,0.1,0.1", ""),
            ("control_lyapunov", "0.1,0.1,0.1,0.1", "0.02"),
        ]
        for v_mode, q_diag, decay in v_specs:
            cases.append(_case(
                group="lyapunov_v_ablation",
                name=f"lyap_v_asym_{v_mode}_stdw",
                wave="medium",
                config=MEDIUM_CFG,
                embodiment="asymmetric",
                variant="stdw_default",
                lyapunov_v_mode=v_mode,
                lyapunov_q_diag=q_diag,
                lyapunov_decay_alpha=decay,
            ))

        # G2 controller_misset_amplified: CM-A pid_gain baseline vs CM-B/CM-C
        # (thrust/structure) that bypass sigmoid saturation. off_clean shares each
        # mismatch spec so STDW relative gain is pairable.
        amp_specs: list[tuple[str, dict[str, float], dict[str, Any]]] = [
            ("cmA_pid_gain_0p01", {
                "roll_zeta1": 0.01, "roll_zeta2": 0.01,
                "pitch_zeta1": 0.01, "pitch_zeta2": 0.01,
                "yaw_zeta1": 0.01, "yaw_zeta2": 0.01,
                "depth_zeta1": 0.01, "depth_zeta2": 0.01,
            }, {}),
            ("cmB_thrust_0p20", {}, {"mode": "actuator_scale", "thrust_scale": 0.2}),
            ("cmC_sratio_0p20", {}, {"mode": "s_surface_struct", "s_ratio_scale": 0.2, "add_scale": 0.0}),
        ]
        for spec_name, pid_mult, mismatch in amp_specs:
            for variant in ("off_clean", "stdw_default"):
                cases.append(_case(
                    group="controller_misset_amplified",
                    name=f"misset_amp_asym_{spec_name}_{variant}",
                    wave="medium",
                    config=MISMATCH_AMP_CFG,
                    embodiment="asymmetric",
                    variant=variant,
                    pid_multipliers=pid_mult,
                    ctrl_mismatch=mismatch,
                ))

        # G3 boundary_effect_pressure: off/residual/free_surface/full x off/on.
        # boundary_effect is part of _match_key so each boundary env pairs its own
        # off_clean baseline.
        boundary_specs: list[tuple[str, str]] = [
            ("off", ""),
            ("residual", "residual_buoyancy"),
            ("free_surface", "free_surface"),
            ("full", "full"),
        ]
        for tag, effect in boundary_specs:
            config = BOUNDARY_FULL_CFG if tag in ("free_surface", "full") else BOUNDARY_RESIDUAL_CFG
            if tag == "off":
                config = MEDIUM_CFG
            for variant in ("off_clean", "stdw_default"):
                cases.append(_case(
                    group="boundary_effect_pressure",
                    name=f"boundary_asym_{tag}_{variant}",
                    wave="medium",
                    config=config,
                    embodiment="asymmetric",
                    variant=variant,
                    boundary_effect=effect,
                ))

        # G4 esuot_vs_opr: OPR baseline vs esuot_full/esuot_light/none, asym +
        # boundary on (full). off_clean anchors the pairing; the on-variants carry
        # different domain_adapt_backend but share the same boundary env.
        cases.append(_case(
            group="esuot_vs_opr",
            name="esuot_asym_boundary_off_clean",
            wave="medium",
            config=BOUNDARY_FULL_CFG,
            embodiment="asymmetric",
            variant="off_clean",
            boundary_effect="full",
        ))
        for backend in ("opr", "esuot_full", "esuot_light", "none"):
            cases.append(_case(
                group="esuot_vs_opr",
                name=f"esuot_asym_boundary_{backend}",
                wave="medium",
                config=ESUOT_CFG,
                embodiment="asymmetric",
                variant="stdw_default",
                boundary_effect="full",
                domain_adapt_backend=backend,
            ))
        return cases

    if profile == "focused":
        for wave, config in (("medium", MEDIUM_CFG), ("storm", STORM_CFG)):
            for variant in ("off_clean", "stdw_default", "stdw_batch_trust", "lyap_guard_zero"):
                cases.append(_case(
                    group="lyapunov_asym_guard",
                    name=f"lyap_asym_{wave}_{variant}",
                    wave=wave,
                    config=config,
                    embodiment="asymmetric",
                    variant=variant,
                ))

        all_pd_0p01 = {
            "roll_zeta1": 0.01, "roll_zeta2": 0.01,
            "pitch_zeta1": 0.01, "pitch_zeta2": 0.01,
            "yaw_zeta1": 0.01, "yaw_zeta2": 0.01,
            "depth_zeta1": 0.01, "depth_zeta2": 0.01,
        }
        for embodiment in ("base", "asymmetric"):
            for variant in ("off_clean", "stdw_default", "stdw_batch_trust", "lyap_guard_zero"):
                cases.append(_case(
                    group="controller_misset",
                    name=f"misset_{embodiment}_all_pd_0p01_{variant}",
                    wave="medium",
                    config=MEDIUM_CFG,
                    embodiment=embodiment,
                    variant=variant,
                    pid_multipliers=all_pd_0p01,
                ))

        for embodiment in ("base", "asymmetric"):
            for variant in ("off_clean", "stdw_default"):
                cases.append(_case(
                    group="flip360_eval",
                    name=f"flip360_{embodiment}_{variant}",
                    wave="medium",
                    config=FLIP360_CFG,
                    embodiment=embodiment,
                    variant=variant,
                    reference_mode="flip360_sine",
                ))
        return cases

    variants = ["off_clean", "stdw_default", "stdw_batch_trust", "lyap_strict", "lyap_guard_zero"]
    for wave, config in (("medium", MEDIUM_CFG), ("storm", STORM_CFG)):
        for variant in variants:
            cases.append(_case(
                group="lyapunov_asym_guard",
                name=f"lyap_asym_{wave}_{variant}",
                wave=wave,
                config=config,
                embodiment="asymmetric",
                variant=variant,
            ))

    mismatch_specs: list[tuple[str, dict[str, float]]] = [
        ("depth_p_0p01", {"depth_zeta1": 0.01}),
        ("attitude_p_0p01", {"roll_zeta1": 0.01, "pitch_zeta1": 0.01, "yaw_zeta1": 0.01}),
        ("all_pd_0p01", {
            "roll_zeta1": 0.01, "roll_zeta2": 0.01,
            "pitch_zeta1": 0.01, "pitch_zeta2": 0.01,
            "yaw_zeta1": 0.01, "yaw_zeta2": 0.01,
            "depth_zeta1": 0.01, "depth_zeta2": 0.01,
        }),
    ]
    for embodiment in ("base", "asymmetric"):
        for mismatch_name, multipliers in mismatch_specs:
            for variant in ("off_clean", "stdw_default", "stdw_batch_trust", "lyap_guard_zero"):
                cases.append(_case(
                    group="controller_misset",
                    name=f"misset_{embodiment}_{mismatch_name}_{variant}",
                    wave="medium",
                    config=MEDIUM_CFG,
                    embodiment=embodiment,
                    variant=variant,
                    pid_multipliers=multipliers,
                ))

    for embodiment in ("base", "asymmetric"):
        for variant in ("off_clean", "stdw_default", "stdw_batch_trust", "lyap_guard_zero"):
            cases.append(_case(
                group="flip360_eval",
                name=f"flip360_{embodiment}_{variant}",
                wave="medium",
                config=FLIP360_CFG,
                embodiment=embodiment,
                variant=variant,
                reference_mode="flip360_sine",
            ))

    if profile == "smoke":
        names = {
            "lyap_asym_medium_off_clean",
            "lyap_asym_medium_stdw_default",
            "lyap_asym_medium_stdw_batch_trust",
            "lyap_asym_medium_lyap_guard_zero",
            "flip360_base_off_clean",
        }
        return [case for case in cases if case["name"] in names]
    if profile != "small_hard":
        raise ValueError(
            "profile must be smoke/focused/small_hard/flip360_only/ordinary_only/"
            f"mismatch_amplified/hard_constraints_v2/embodiment_zoo, got {profile!r}"
        )
    return cases


def _match_key(row: dict[str, Any]) -> tuple[Any, ...]:
    pid = row.get("pid_multipliers") or {}
    if isinstance(pid, str):
        pid_key = pid
    else:
        pid_key = json.dumps(pid, sort_keys=True)
    mismatch = row.get("ctrl_mismatch") or {}
    if isinstance(mismatch, str):
        mismatch_key = mismatch
    else:
        mismatch_key = json.dumps(mismatch, sort_keys=True)
    return (
        row.get("group"),
        row.get("wave"),
        row.get("embodiment"),
        row.get("reference_mode"),
        pid_key,
        mismatch_key,
        row.get("boundary_effect") or "",
    )


def build_cmd(case: dict[str, Any], args: argparse.Namespace, case_dir: Path) -> list[str]:
    policy_path = Path(args.policy_path)
    logs_root, load_run, checkpoint = _policy_to_load_args(policy_path)
    cmd = [
        "bash", str(RUNNER), str(PLAY),
        "--headless",
        "--task", "EasyUUV-Direct-Parametric-v1",
        "--num_envs", "1",
        "--experiment_name", "easyuuv_parametric",
        "--logs_root", logs_root,
        "--load_run", load_run,
        "--checkpoint", checkpoint,
        "--workflow_config", str(case["config"]),
        "--total_steps", str(args.total_steps),
        "--seed", str(args.seed),
        "--embodiment", str(case["embodiment"]),
        "--auto_drift_router", "False",
        "--drift_router_mode", "off",
        "--enable_micro_probe", "False",
        "--enable_trigger_gate", "True",
        "--trigger_threshold", "0.05",
        "--results_root", str(case_dir / "results"),
        "--artifacts_root", str(case_dir / "artifacts"),
    ]
    cmd.extend(_variant_extra(str(case["variant"])))
    if case.get("pid_multipliers"):
        cmd.extend(["--pid_multipliers", json.dumps(case["pid_multipliers"], sort_keys=True)])
    if case.get("ctrl_mismatch"):
        cmd.extend(["--ctrl_mismatch", json.dumps(case["ctrl_mismatch"], sort_keys=True)])
    if case.get("lyapunov_v_mode"):
        cmd.extend(["--lyapunov_v_mode", str(case["lyapunov_v_mode"])])
    if case.get("lyapunov_q_diag"):
        cmd.extend(["--lyapunov_q_diag", str(case["lyapunov_q_diag"])])
    if case.get("lyapunov_decay_alpha"):
        cmd.extend(["--lyapunov_decay_alpha", str(case["lyapunov_decay_alpha"])])
    if case.get("boundary_effect"):
        cmd.extend(["--boundary_effect", str(case["boundary_effect"])])
    if case.get("domain_adapt_backend"):
        cmd.extend(["--domain_adapt_backend", str(case["domain_adapt_backend"])])
    return cmd


def latest_summary(case_dir: Path) -> tuple[str, dict[str, Any]]:
    summaries = sorted((case_dir / "results").glob("**/summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        return "", {}
    path = summaries[-1]
    return str(path), json.loads(path.read_text(encoding="utf-8"))


def _numeric(summary: dict[str, Any], key: str) -> Any:
    value = summary.get(key)
    if value is None:
        return ""
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["case"])
        writer.writeheader()
        writer.writerows(rows)


def _annotate_passes(rows: list[dict[str, Any]], *, pass_abs_tol: float, pass_rel_tol: float) -> None:
    off_by_key: dict[tuple[Any, ...], float] = {}
    reset_by_key: dict[tuple[Any, ...], int] = {}
    nonfinite_by_key: dict[tuple[Any, ...], int] = {}
    for row in rows:
        if row.get("variant") != "off_clean":
            continue
        try:
            off_by_key[_match_key(row)] = float(row["final_mse"])
            reset_by_key[_match_key(row)] = int(row.get("reset_count") or 0)
            nonfinite_by_key[_match_key(row)] = int(row.get("nonfinite_guard_count") or 0)
        except Exception:
            continue

    for row in rows:
        key = _match_key(row)
        off = off_by_key.get(key)
        if off is None or row.get("variant") == "off_clean" or row.get("final_mse") in ("", None):
            row["matched_off_final_mse"] = off if off is not None else ""
            row["pass_vs_off"] = ""
            continue
        final_mse = float(row["final_mse"])
        reset_ok = int(row.get("reset_count") or 0) <= reset_by_key.get(key, 0)
        nonfinite_ok = int(row.get("nonfinite_guard_count") or 0) <= nonfinite_by_key.get(key, 0)
        row["matched_off_final_mse"] = off
        row["delta_vs_off_pct"] = (final_mse - off) / max(off, 1.0e-9) * 100.0
        tolerance = max(float(pass_abs_tol), float(pass_rel_tol) * abs(off))
        row["pass_tolerance"] = tolerance
        row["pass_vs_off"] = bool(final_mse <= off + tolerance and reset_ok and nonfinite_ok)


def _write_summary(root: Path, rows: list[dict[str, Any]], *, pass_abs_tol: float, pass_rel_tol: float) -> None:
    _annotate_passes(rows, pass_abs_tol=pass_abs_tol, pass_rel_tol=pass_rel_tol)
    payload = {
        "rows": rows,
        "pass_abs_tol": float(pass_abs_tol),
        "pass_rel_tol": float(pass_rel_tol),
        "counts": {
            "total": len(rows),
            "failed_returncode": sum(1 for row in rows if int(row.get("returncode") or 0) != 0),
            "failed_vs_off": sum(1 for row in rows if row.get("pass_vs_off") is False),
        },
    }
    (root / "pressure_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# STDW Safety Pressure Report",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- rows: {len(rows)}",
        "",
        "| group | case | variant | embodiment | wave | final_mse | matched_off | delta_vs_off_pct | pass_vs_off | blocks | rejected | state | nonfinite |",
        "|---|---|---|---|---|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {group} | {case} | {variant} | {embodiment} | {wave} | {final_mse} | "
            "{matched} | {delta} | {passed} | {blocks} | {rejected} | {state} | {nonfinite} |".format(
                group=row.get("group", ""),
                case=row.get("case", ""),
                variant=row.get("variant", ""),
                embodiment=row.get("embodiment", ""),
                wave=row.get("wave", ""),
                final_mse=row.get("final_mse", ""),
                matched=row.get("matched_off_final_mse", ""),
                delta=row.get("delta_vs_off_pct", ""),
                passed=row.get("pass_vs_off", ""),
                blocks=row.get("lyapunov_block_count", ""),
                rejected=row.get("stdw_update_rejected_count", ""),
                state=row.get("stdw_safety_state_final", ""),
                nonfinite=row.get("nonfinite_guard_count", ""),
            )
        )
    (root / "pressure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run focused STDW safety pressure matrix.")
    parser.add_argument("--profile", default="focused", choices=[
        "smoke", "focused", "small_hard", "flip360_only", "ordinary_only",
        "mismatch_amplified", "hard_constraints_v2", "embodiment_zoo",
    ])
    parser.add_argument("--results_root", default=None)
    parser.add_argument("--policy_path", default=str(DEFAULT_POLICY))
    parser.add_argument("--total_steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--run", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pass_abs_tol", type=float, default=1.0e-9)
    parser.add_argument("--pass_rel_tol", type=float, default=1.0e-6)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.results_root or f".results/stdw_safety_pressure_{timestamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args.profile)
    if args.limit is not None:
        cases = cases[: int(args.limit)]

    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        case_dir = root / str(case["name"])
        cmd = build_cmd(case, args, case_dir)
        print(f"[{idx:02d}/{len(cases):02d}] {case['name']}")
        print(" ".join(cmd))
        started = datetime.now().isoformat(timespec="seconds")
        rc = 0
        if args.run and not args.dry_run:
            case_dir.mkdir(parents=True, exist_ok=True)
            with (case_dir / "run.log").open("w", encoding="utf-8") as f:
                rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        ended = datetime.now().isoformat(timespec="seconds")
        summary_path, summary = latest_summary(case_dir)
        row = {
            "case": case["name"],
            "group": case["group"],
            "wave": case["wave"],
            "config": case["config"],
            "embodiment": case["embodiment"],
            "variant": case["variant"],
            "reference_mode": case["reference_mode"],
            "pid_multipliers": json.dumps(case.get("pid_multipliers") or {}, sort_keys=True),
            "ctrl_mismatch": json.dumps(case.get("ctrl_mismatch") or {}, sort_keys=True),
            "lyapunov_v_mode": case.get("lyapunov_v_mode") or "",
            "boundary_effect": case.get("boundary_effect") or "",
            "domain_adapt_backend": case.get("domain_adapt_backend") or "",
            "returncode": rc,
            "command": " ".join(cmd),
            "started_at": started,
            "ended_at": ended,
            "summary_path": summary_path,
            "final_mse": _numeric(summary, "final_mse"),
            "final_mse_after_drift": _numeric(summary, "final_mse_after_drift"),
            "mean_total_mse": _numeric(summary, "mean_total_mse"),
            "mean_roll_mse": _numeric(summary, "mean_roll_mse"),
            "mean_pitch_mse": _numeric(summary, "mean_pitch_mse"),
            "mean_yaw_mse": _numeric(summary, "mean_yaw_mse"),
            "mean_depth_mse": _numeric(summary, "mean_depth_mse"),
            "slow_loop_triggers": _numeric(summary, "slow_loop_triggers"),
            "gate_silenced_count": _numeric(summary, "gate_silenced_count"),
            "lyapunov_block_count": _numeric(summary, "lyapunov_block_count"),
            "stdw_safety_state_final": _numeric(summary, "stdw_safety_state_final"),
            "stdw_fallback_step": _numeric(summary, "stdw_fallback_step"),
            "stdw_update_acceptance": _numeric(summary, "stdw_update_acceptance"),
            "stdw_update_accepted_count": _numeric(summary, "stdw_update_accepted_count"),
            "stdw_update_rejected_count": _numeric(summary, "stdw_update_rejected_count"),
            "stdw_last_reject_reason": _numeric(summary, "stdw_last_reject_reason"),
            "effective_batch_frac_mean": _numeric(summary, "effective_batch_frac_mean"),
            "reset_count": _numeric(summary, "reset_count"),
            "nonfinite_guard_count": _numeric(summary, "nonfinite_guard_count"),
        }
        rows.append(row)
        _write_summary(root, rows, pass_abs_tol=args.pass_abs_tol, pass_rel_tol=args.pass_rel_tol)
        _write_rows(root / "pressure_runs.csv", rows)
        if rc != 0:
            print(f"[ERROR] case failed: {case['name']} rc={rc}")
            break

    print(f"[DONE] rows={len(rows)} -> {root / 'pressure_runs.csv'}")
    return 0 if all(int(row["returncode"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
