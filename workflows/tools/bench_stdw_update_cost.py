"""Benchmark the compute *cost* of one STDW effective slow-loop update.

This is NOT an experiment-result script. It quantifies the price tag of a
single STDW gradient update and contrasts it with the per-step fast-loop
inference that the edge controller must run anyway, so we can judge whether
on-device adaptation is deployment-friendly.

What "one effective update" means here is a faithful, dependency-free replica
of the slow loop in ``workflows/play_stdw_adapt.py`` (lines ~1492-1563):

    a_src_pred  = policy.actor(obs_src)          # grad forward  (B=256)
    a_tgt_pred  = policy.actor(obs_tgt)          # grad forward  (B=256)
    with no_grad:
        a_src_anchor = policy_ref.actor(obs_src) # frozen forward (B=256)
        a_tgt_anchor = policy_ref.actor(obs_tgt) # frozen forward (B=256)
    loss = (1-rho)*L_src + rho*L_tgt + lambda_reg*L_reg   # behavior_kl reg
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()                             # Adam, lr=5e-5

Architecture / hyper-params are taken verbatim from
``agents/rsl_rl_ppo_cfg.py`` (EasyUUVParametricPPORunnerCfg) and the
play_stdw_adapt CLI defaults:

    actor MLP  : 12 -> 128 -> 128 -> 8   (ELU)     [the deployed network]
    critic MLP : 12 -> 128 -> 128 -> 1   (ELU)     [updated too: Adam over
                                                    policy.parameters()]
    batch_size           = 256
    slow_loop_interval   = 60 ctrl steps
    ctrl rate            = 60 Hz (dt 1/120 * decimation 2) -> 16.67 ms budget
    optimizer            = Adam(lr=5e-5)
    empirical_normalization = False  (normalizer is identity -> no extra cost)

It times: (1) the fast-loop single-sample inference (what runs every control
tick on the edge), and (2) the full effective update, on whatever devices are
available (CUDA, CPU multi-thread, CPU single-thread). It also reports the
analytic MAC/FLOP count, parameter counts, and peak activation memory, then
writes a JSON summary consumed by ``plot_ppt_figures.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / ".results" / "stdw_update_cost" / "stdw_update_cost.json"

# --- architecture / hyper-params (verbatim from the real pipeline) ----------
OBS_DIM = 12          # num_observations (parametric, A3: 9 + ang_vel 3)
ACT_DIM = 8           # num_actions (parametric meta-control)
HIDDEN = [128, 128]   # actor/critic hidden dims
BATCH = 256           # --batch_size default
LR = 5e-5             # --g_C_lr default
LAMBDA_REG = 1e-3     # --lambda_reg default
SLOW_INTERVAL = 60    # --slow_loop_interval default (ctrl steps)
CTRL_HZ = 60.0        # dt 1/120 * decimation 2


class ActorCritic(nn.Module):
    """Minimal stand-in matching rsl_rl ActorCritic actor/critic MLPs.

    Only the .actor / .critic Sequential stacks matter for cost; the Adam in
    the real loop optimises ``policy.parameters()`` which includes both heads
    plus the action log-std vector.
    """

    def __init__(self) -> None:
        super().__init__()
        self.actor = self._mlp(OBS_DIM, HIDDEN, ACT_DIM)
        self.critic = self._mlp(OBS_DIM, HIDDEN, 1)
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM))

    @staticmethod
    def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        return nn.Sequential(*layers)


def _count_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _actor_macs() -> int:
    """Multiply-accumulate ops for one actor forward over one sample."""
    dims = [OBS_DIM] + HIDDEN + [ACT_DIM]
    return int(sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1)))


def _critic_macs() -> int:
    dims = [OBS_DIM] + HIDDEN + [1]
    return int(sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1)))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_callable(fn: Callable[[], None], device: torch.device,
                   warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    return {
        "mean_ms": mean,
        "median_ms": samples[n // 2],
        "p95_ms": samples[min(n - 1, int(0.95 * n))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "iters": n,
    }


def _build_update_step(device: torch.device):
    policy = ActorCritic().to(device)
    policy.train()
    policy_ref = copy.deepcopy(policy).to(device)
    policy_ref.eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    obs_src = torch.randn(BATCH, OBS_DIM, device=device)
    obs_tgt = torch.randn(BATCH, OBS_DIM, device=device)
    # pseudo target actions (analogous to B_tgt["pseudo_actions"])
    pseudo_tgt = torch.randn(BATCH, ACT_DIM, device=device)
    rho = 0.5

    def step() -> None:
        a_src_pred = policy.actor(obs_src)
        a_tgt_pred = policy.actor(obs_tgt)
        with torch.no_grad():
            a_src_anchor = policy_ref.actor(obs_src).detach()
            a_tgt_anchor = policy_ref.actor(obs_tgt).detach()
        mse_src = ((a_src_pred - a_src_anchor) ** 2).mean(dim=-1)
        mse_tgt = ((a_tgt_pred - pseudo_tgt) ** 2).mean(dim=-1)
        L_src = mse_src.mean()
        L_tgt = mse_tgt.mean()
        # behavior_kl regularizer (the default reg_mode)
        L_reg_src = ((a_src_pred - a_src_anchor) ** 2).mean()
        L_reg_tgt = ((a_tgt_pred - a_tgt_anchor) ** 2).mean()
        L_reg = (1.0 - rho) * L_reg_src + rho * L_reg_tgt
        loss = (1.0 - rho) * L_src + rho * L_tgt + LAMBDA_REG * L_reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    return policy, step


def _build_inference(device: torch.device, policy: ActorCritic):
    infer_net = copy.deepcopy(policy.actor).to(device)
    infer_net.eval()
    obs1 = torch.randn(1, OBS_DIM, device=device)

    def infer() -> None:
        with torch.no_grad():
            infer_net(obs1)

    return infer


def _peak_mem_update(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    _, step = _build_update_step(device)
    for _ in range(3):
        step()
    _sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "baseline_mib": base / 1024 ** 2,
        "peak_update_mib": peak / 1024 ** 2,
        "delta_update_mib": (peak - base) / 1024 ** 2,
    }


def _bench_device(device: torch.device, warmup: int, iters: int) -> Dict:
    policy, step = _build_update_step(device)
    infer = _build_inference(device, policy)
    update_t = _time_callable(step, device, warmup, iters)
    infer_t = _time_callable(infer, device, max(warmup * 5, 50), max(iters * 5, 500))
    out = {
        "device": str(device),
        "update": update_t,
        "inference": infer_t,
        "update_over_inference_x": update_t["mean_ms"] / max(infer_t["mean_ms"], 1e-9),
    }
    out.update({"memory": _peak_mem_update(device)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()

    actor_params = _count_params(ActorCritic().actor)
    critic_params = _count_params(ActorCritic().critic)
    trainable_params = _count_params(ActorCritic())  # Adam over policy.parameters()

    actor_macs = _actor_macs()
    critic_macs = _critic_macs()

    # FLOPs accounting for one effective update (per-sample MACs * batch):
    #   2 grad actor forwards + 2 frozen actor forwards = 4 actor forwards
    #   backward ~= 2x forward MACs over the actor graph (1 grad forward path)
    # We report forward-only and a backward-inclusive estimate.
    fwd_actor_macs_update = 4 * actor_macs * BATCH           # 4 forwards x B
    bwd_actor_macs_update = 2 * (2 * actor_macs) * BATCH     # 2 grad fwds -> ~2x bwd
    update_macs_total = fwd_actor_macs_update + bwd_actor_macs_update
    infer_macs = actor_macs * 1                              # batch 1

    summary: Dict = {
        "meta": {
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "config": {
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
            "hidden": HIDDEN,
            "batch_size": BATCH,
            "lr": LR,
            "lambda_reg": LAMBDA_REG,
            "slow_loop_interval_steps": SLOW_INTERVAL,
            "ctrl_hz": CTRL_HZ,
            "reg_mode": "behavior_kl",
            "optimizer": "Adam",
        },
        "model": {
            "actor_params": actor_params,
            "critic_params": critic_params,
            "trainable_params_total": trainable_params,
            "actor_macs_per_sample": actor_macs,
            "critic_macs_per_sample": critic_macs,
            "actor_kflops_per_sample": 2 * actor_macs / 1e3,  # 1 MAC = 2 FLOP
        },
        "flops": {
            "inference_macs": infer_macs,
            "inference_kflops": 2 * infer_macs / 1e3,
            "update_forward_macs": fwd_actor_macs_update,
            "update_backward_macs_est": bwd_actor_macs_update,
            "update_total_macs_est": update_macs_total,
            "update_mflops_est": 2 * update_macs_total / 1e6,
            "update_over_inference_flop_x": update_macs_total / max(infer_macs, 1),
        },
        "cadence": {
            "updates_per_second_if_every_interval": CTRL_HZ / SLOW_INTERVAL,
            "ctrl_period_ms": 1000.0 / CTRL_HZ,
            "note": (
                "Slow loop fires at most every slow_loop_interval ctrl steps "
                "(~1 s) and only when the trigger gate opens; ~20 effective "
                "updates per 1500-step episode in practice."
            ),
        },
        "benchmarks": [],
    }

    devices: List[torch.device] = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    devices.append(torch.device("cpu"))

    for device in devices:
        if device.type == "cpu":
            # multi-thread (default) then single-thread for edge-MCU proxy
            prev = torch.get_num_threads()
            res_mt = _bench_device(device, args.warmup, args.iters)
            res_mt["threads"] = prev
            res_mt["label"] = f"CPU (x{prev} threads)"
            summary["benchmarks"].append(res_mt)

            torch.set_num_threads(1)
            res_st = _bench_device(device, args.warmup, args.iters)
            res_st["threads"] = 1
            res_st["label"] = "CPU (1 thread, edge proxy)"
            summary["benchmarks"].append(res_st)
            torch.set_num_threads(prev)
        else:
            res = _bench_device(device, args.warmup, args.iters)
            res["label"] = summary["meta"]["cuda_device_name"] or "CUDA"
            summary["benchmarks"].append(res)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[OK] wrote {out_path}")

    # Console digest
    print("\n=== STDW one-effective-update cost ===")
    print(f"actor params      : {actor_params:,}")
    print(f"trainable params  : {trainable_params:,} (Adam over policy.parameters())")
    print(f"actor MACs/sample : {actor_macs:,}  ({2*actor_macs/1e3:.2f} kFLOP)")
    print(f"update MACs (est) : {update_macs_total:,}  ({2*update_macs_total/1e6:.3f} MFLOP)")
    print(f"update/infer FLOP : {update_macs_total/max(infer_macs,1):.0f}x")
    for b in summary["benchmarks"]:
        lbl = b.get("label", b["device"])
        print(
            f"[{lbl}] update {b['update']['mean_ms']:.3f} ms "
            f"(p95 {b['update']['p95_ms']:.3f}) | "
            f"infer {b['inference']['mean_ms']*1e3:.1f} us | "
            f"update/infer {b['update_over_inference_x']:.0f}x"
        )


if __name__ == "__main__":
    main()
