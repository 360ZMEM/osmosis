"""ParametricGainTuner: 8 维元控制下，把后 4 维 ``a_gain`` 映射到 S-Surface 增益 ζ_runtime。

四个控制学机制串联（每一步都经过）：

    a_gain (raw, [-1,1])
      └─[1] Singular-perturbation Low-Pass Filter
      └─[2] Dead-Zone Parameter Freezing  (基于 compound_error)
      └─[3] Bounded Safeguard:  ζ_i = ζ_nom · (1 + β · a_gain_i)
      └─[4] Persistent Excitation 注入:  ζ_i ← ζ_modulated + ζ_nom · a(t) sin(ω_d t)
                                         其中 a(t) = pe_amp / (1 + γ ‖ω_body‖²)
    （可选）identity_init: 直接旁路全部，输出 ζ_runtime ≡ ζ_nominal。

每个机制均可独立开关 / 调参；模块仅依赖 ``torch``，便于独立单测。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


class ParametricGainTuner:
    """Stateful per-env adaptive gain tuner for the 8-dim meta-control policy.

    Args:
        num_envs: 并行 env 数量；用于分配 LPF 状态。
        num_axes: ζ 控制的维度（Roll/Pitch/Yaw/Depth = 4）。
        dt: 一个决策步长（建议 ``sim.dt * decimation``），LPF/PE 都基于这个 dt。
        device: torch 设备。
        gain_beta: Bounded Safeguard 因子，限制 ζ 在 ±β 标称值波动内（默认 0.2）。
        enable_pe / pe_freq / pe_amp / pe_decay_gamma: PE 注入参数（频率 Hz / 振幅 / 状态相关衰减）。
        enable_deadzone / deadzone_threshold: 死区拦截参数（阈值与 ``compound_error`` 同量纲）。
        enable_param_lpf / param_lpf_cutoff: 一阶 LPF 截止频率（Hz）。
        identity_init: 幂等初始化；启用时整个模块退化为恒等映射。
    """

    def __init__(
        self,
        num_envs: int,
        num_axes: int,
        dt: float,
        device,
        *,
        gain_beta: float = 0.2,
        enable_pe: bool = True,
        pe_freq: float = 0.5,
        pe_amp: float = 0.05,
        pe_decay_gamma: float = 5.0,
        enable_deadzone: bool = True,
        deadzone_threshold: float = 0.02,
        enable_param_lpf: bool = True,
        param_lpf_cutoff: float = 1.0,
        identity_init: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_axes = int(num_axes)
        self.dt = float(dt)
        self.device = device

        self.gain_beta = float(gain_beta)
        self.enable_pe = bool(enable_pe)
        self.pe_freq = float(pe_freq)
        self.pe_amp = float(pe_amp)
        self.pe_decay_gamma = float(pe_decay_gamma)
        self.enable_deadzone = bool(enable_deadzone)
        self.deadzone_threshold = float(deadzone_threshold)
        self.enable_param_lpf = bool(enable_param_lpf)
        self.param_lpf_cutoff = float(param_lpf_cutoff)
        self.identity_init = bool(identity_init)

        # LPF 状态：一阶 IIR 输出缓存。reset 时置零。
        self._a_gain_lpf = torch.zeros(self.num_envs, self.num_axes, device=self.device)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """重置 LPF 状态。当 env reset / DR 重抽样时调用。"""
        if env_ids is None:
            self._a_gain_lpf.zero_()
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._a_gain_lpf[ids] = 0.0

    @property
    def a_gain_lpf(self) -> torch.Tensor:
        """暴露给 eval 脚本读取的 LPF 后 a_gain（不可写）。"""
        return self._a_gain_lpf

    # ------------------------------------------------------------------
    # Core mapping
    # ------------------------------------------------------------------
    def step(
        self,
        a_gain_raw: torch.Tensor,        # (N, num_axes), in [-1, 1]
        zeta_nominal: torch.Tensor,      # (N, num_axes)
        body_ang_vel: torch.Tensor,      # (N, 3)
        compound_error: torch.Tensor,    # (N,) scalar per-env
        sim_time_s: torch.Tensor,        # (N,) seconds
    ) -> Tuple[torch.Tensor, dict]:
        """Apply the 4 mechanisms in series. Returns ``(zeta_runtime, debug_dict)``."""

        # === Identity initialization shortcut ==============================
        # 当 identity_init=True 时，整个调制器退化为 ζ_runtime ≡ ζ_nominal，
        # 等价于 (β=0, PE 关, LPF 关, 死区无影响)。用于课程学习起步。
        if self.identity_init:
            return zeta_nominal.clone(), {
                "pe_active": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "deadzone_active": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "a_gain_eff": torch.zeros_like(a_gain_raw),
            }

        # === 1) Singular-perturbation Low-Pass Filter =====================
        # 控制学依据 (Khalil, Singular Perturbation Methods)：
        # 当快慢变量的时间尺度比 ε = τ_fast / τ_slow << 1 时，慢系统的稳态可视
        # 为快系统的"边界层"。这里用一阶 LPF 把 RL 的 a_gain 信号限制在慢时间
        # 尺度 (cutoff f_c) 上，从而保证内环 (S-Surface) 仍处于其奇异摄动收敛
        # 域内。
        # 离散形式 (forward Euler)：
        #     RC = 1 / (2π f_c)
        #     α = dt / (RC + dt)
        #     y[k] = (1-α) y[k-1] + α x[k]
        if self.enable_param_lpf:
            rc = 1.0 / (2.0 * math.pi * max(self.param_lpf_cutoff, 1e-6))
            alpha = self.dt / (rc + self.dt)
            self._a_gain_lpf = (1.0 - alpha) * self._a_gain_lpf + alpha * a_gain_raw
            a_gain = self._a_gain_lpf
        else:
            a_gain = a_gain_raw

        # === 2) Dead-Zone Parameter Freezing ==============================
        # 自适应控制理论 (Ioannou & Sun, Robust Adaptive Control, Ch.8.5)：
        # 当跟踪误差落入死区 [-ε, +ε] 时，强行关闭增益更新动态以避免：
        #   (a) 测量噪声驱动的参数随机游走 (parameter drift)；
        #   (b) 持续小信号下的"参数爆破" (parameter bursting)。
        # 死区直接作用在调用方提供的 compound_error 上（建议 ‖ω_body‖）。
        if self.enable_deadzone:
            in_deadzone = compound_error < self.deadzone_threshold     # (N,)
            a_gain = torch.where(
                in_deadzone.unsqueeze(-1),
                torch.zeros_like(a_gain),
                a_gain,
            )
        else:
            in_deadzone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # === 3) Bounded Safeguard (对角残差映射) ==========================
        # 物理安全锁：限制每一维 ζ 的偏离不超过标称值的 ±β。
        # 公式 ζ_i = ζ_nom · (1 + β·a_gain_i) 等价于一个对角线性残差映射，
        # 当 a_gain ∈ [-1, 1] 时保证 ζ_i ∈ [(1-β)·ζ_nom, (1+β)·ζ_nom]，从而
        # 阻止策略噪声把 S-Surface 推到非 Hurwitz 区域。
        zeta_modulated = zeta_nominal * (1.0 + self.gain_beta * a_gain)

        # === 4) Persistent Excitation (PE) ================================
        # 自适应控制收敛性需要 PE 条件 (Narendra & Annaswamy 1989, Th.2.7.1)：
        # regressor φ(t) 必须满足 ∫_t^{t+T} φφᵀ dτ ≥ α I。为此，向 ζ 注入正
        # 弦微振 a(t)·sin(ω_d t)，频率由 pe_freq 决定。
        # 状态相关探测振幅衰减——当系统已剧烈机动 (‖ω‖ 大) 时，自身已富激励，
        # 外加扰动只会破坏稳定性，因此：
        #     a(t) = pe_amp / (1 + γ·‖ω_body‖²)
        # 这是 σ-modification (Ioannou) 的一个反向变体。
        if self.enable_pe:
            omega_norm_sq = (body_ang_vel ** 2).sum(dim=-1)        # (N,)
            a_t = self.pe_amp / (1.0 + self.pe_decay_gamma * omega_norm_sq)
            pe_signal = a_t.unsqueeze(-1) * torch.sin(
                2.0 * math.pi * self.pe_freq * sim_time_s.unsqueeze(-1)
            )
            zeta_runtime = zeta_modulated + zeta_nominal * pe_signal
            pe_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            zeta_runtime = zeta_modulated
            pe_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        debug = {
            "pe_active": pe_active,
            "deadzone_active": in_deadzone,
            "a_gain_eff": a_gain,
        }
        return zeta_runtime, debug


# =====================================================================
# Lightweight self-test (run with ``python gain_tuner.py``)
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    N, K = 8, 4
    dt = 0.0167  # 60 Hz
    device = torch.device("cpu")

    zeta_nom = torch.tensor([[1.0, 1.0, 2.0, 0.4]], device=device).repeat(N, 1)

    # Test 1: identity_init -> exact equality with zeta_nominal
    tuner = ParametricGainTuner(N, K, dt, device, identity_init=True)
    zeta, dbg = tuner.step(
        a_gain_raw=torch.randn(N, K),
        zeta_nominal=zeta_nom,
        body_ang_vel=torch.randn(N, 3),
        compound_error=torch.rand(N),
        sim_time_s=torch.zeros(N),
    )
    assert torch.allclose(zeta, zeta_nom), "identity_init must yield zeta == zeta_nominal"
    print("[OK] identity_init bypasses all mechanisms.")

    # Test 2: a_gain=0, all mechanisms off -> zeta == zeta_nominal
    tuner = ParametricGainTuner(
        N, K, dt, device,
        enable_pe=False, enable_deadzone=False, enable_param_lpf=False,
        identity_init=False,
    )
    zeta, _ = tuner.step(
        a_gain_raw=torch.zeros(N, K),
        zeta_nominal=zeta_nom,
        body_ang_vel=torch.zeros(N, 3),
        compound_error=torch.full((N,), 1.0),  # outside deadzone irrelevant
        sim_time_s=torch.zeros(N),
    )
    assert torch.allclose(zeta, zeta_nom), "zero a_gain w/ all off must yield zeta == zeta_nominal"
    print("[OK] zero a_gain + mechanisms off -> identity.")

    # Test 3: deadzone freezes a_gain
    tuner = ParametricGainTuner(
        N, K, dt, device,
        enable_pe=False, enable_deadzone=True, deadzone_threshold=0.5,
        enable_param_lpf=False, gain_beta=0.2,
    )
    a_gain = torch.full((N, K), 1.0)
    zeta, dbg = tuner.step(
        a_gain_raw=a_gain,
        zeta_nominal=zeta_nom,
        body_ang_vel=torch.zeros(N, 3),
        compound_error=torch.full((N,), 0.1),  # below deadzone
        sim_time_s=torch.zeros(N),
    )
    assert torch.allclose(zeta, zeta_nom), "deadzone must freeze gains to nominal"
    assert dbg["deadzone_active"].all(), "deadzone_active flag must be True"
    print("[OK] deadzone parameter freezing.")

    # Test 4: bounded safeguard caps within [(1-β), (1+β)]
    tuner = ParametricGainTuner(
        N, K, dt, device,
        enable_pe=False, enable_deadzone=False, enable_param_lpf=False,
        gain_beta=0.2,
    )
    zeta, _ = tuner.step(
        a_gain_raw=torch.full((N, K), 1.0),
        zeta_nominal=zeta_nom,
        body_ang_vel=torch.zeros(N, 3),
        compound_error=torch.full((N,), 1.0),
        sim_time_s=torch.zeros(N),
    )
    assert torch.allclose(zeta, zeta_nom * 1.2), "max bound must be (1+β)·zeta_nom"
    zeta2, _ = tuner.step(
        a_gain_raw=torch.full((N, K), -1.0),
        zeta_nominal=zeta_nom,
        body_ang_vel=torch.zeros(N, 3),
        compound_error=torch.full((N,), 1.0),
        sim_time_s=torch.zeros(N),
    )
    assert torch.allclose(zeta2, zeta_nom * 0.8), "min bound must be (1-β)·zeta_nom"
    print("[OK] bounded safeguard within ±β.")

    # Test 5: PE injects sinusoid; amplitude decays with body_ang_vel
    tuner = ParametricGainTuner(
        N, K, dt, device,
        enable_pe=True, pe_freq=1.0, pe_amp=0.1, pe_decay_gamma=10.0,
        enable_deadzone=False, enable_param_lpf=False, gain_beta=0.0,
    )
    # Quiet system: PE should produce non-zero perturbation
    zeta, dbg = tuner.step(
        a_gain_raw=torch.zeros(N, K),
        zeta_nominal=zeta_nom,
        body_ang_vel=torch.zeros(N, 3),
        compound_error=torch.full((N,), 1.0),
        sim_time_s=torch.full((N,), 0.25),  # sin(2π·1·0.25)=1
    )
    delta = (zeta - zeta_nom).abs().max().item()
    assert delta > 1e-4, f"PE should perturb when system is quiet; delta={delta}"
    # Excited system: PE amplitude is heavily attenuated
    zeta2, _ = tuner.step(
        a_gain_raw=torch.zeros(N, K),
        zeta_nominal=zeta_nom,
        body_ang_vel=torch.full((N, 3), 5.0),
        compound_error=torch.full((N,), 1.0),
        sim_time_s=torch.full((N,), 0.25),
    )
    delta2 = (zeta2 - zeta_nom).abs().max().item()
    assert delta2 < delta, "PE amplitude must decay when ω_body is large"
    print(f"[OK] PE active: quiet|delta|={delta:.4f} > excited|delta|={delta2:.4f}.")

    # Test 6: LPF causes lag
    tuner = ParametricGainTuner(
        N, K, dt, device,
        enable_pe=False, enable_deadzone=False, enable_param_lpf=True,
        param_lpf_cutoff=1.0, gain_beta=1.0,
    )
    a_gain = torch.full((N, K), 1.0)
    z1, _ = tuner.step(a_gain, zeta_nom, torch.zeros(N, 3), torch.full((N,), 1.0), torch.zeros(N))
    z2, _ = tuner.step(a_gain, zeta_nom, torch.zeros(N, 3), torch.full((N,), 1.0), torch.zeros(N))
    # First-step LPF output should be < second-step output (slow rise)
    assert (z2 - zeta_nom).abs().sum() > (z1 - zeta_nom).abs().sum(), "LPF should produce slow rise"
    print("[OK] LPF first-order lag.")

    print("\nAll ParametricGainTuner self-tests PASSED.")
