"""M1 standalone synthetic test: Lyapunov V redefinition (4 v_modes).

Exercises EasyUUVStdwWrapper's V / dV / mask computation WITHOUT Isaac by
feeding scripted channel-error vectors.  Validates the §3 plan invariants:

  - V-A pose_quadratic (legacy): V = 0.5 * sum(P * e^2); gate dV < eps.
  - V-B so3_consistent: attitude error read as a single SO(3) geodesic
    (quat_error_magnitude), avoiding Euler-wrap jumps.  A stub math module is
    injected so the SO(3) path runs offline.
  - V-C energy_with_rate: V adds 0.5 * sum(Q * edot^2); a *changing* error
    raises V above the pose-only value, while a *static* error matches it.
  - V-D control_lyapunov: gate requires exponential decay dV <= -alpha*V_prev,
    strictly harder to pass than plain dV < 0.

Run:  python workflows/tools/test_m1_lyapunov_v.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_math_stub() -> None:
    """Inject a minimal omni.isaac.lab.utils.math with quat_error_magnitude.

    Only installed if the real module is unavailable (offline runs), so the
    so3_consistent path can be exercised without Isaac.
    """
    try:
        from omni.isaac.lab.utils.math import quat_error_magnitude  # noqa: F401
        return
    except Exception:
        pass

    def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        # geodesic angle between unit quaternions: 2*acos(|<q1,q2>|)
        q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
        return 2.0 * torch.acos(dot)

    mod_chain = [
        "omni", "omni.isaac", "omni.isaac.lab", "omni.isaac.lab.utils",
        "omni.isaac.lab.utils.math",
    ]
    for name in mod_chain:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["omni.isaac.lab.utils.math"].quat_error_magnitude = quat_error_magnitude
    # wire submodule attributes so `from ... import` resolves
    sys.modules["omni.isaac.lab.utils"].math = sys.modules["omni.isaac.lab.utils.math"]
    sys.modules["omni.isaac.lab"].utils = sys.modules["omni.isaac.lab.utils"]
    sys.modules["omni.isaac"].lab = sys.modules["omni.isaac.lab"]
    sys.modules["omni"].isaac = sys.modules["omni.isaac"]


_install_math_stub()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from easyuuv_stdw_wrapper import EasyUUVStdwWrapper  # noqa: E402


class _DummyData:
    def __init__(self):
        self.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.root_pos_w = torch.tensor([[0.0, 0.0, 1.5]])


class _DummyRobot:
    def __init__(self):
        self.data = _DummyData()


class _DummyCfg:
    starting_depth = 1.5


class _DummyUnwrapped:
    def __init__(self):
        self.com_to_cob_offsets = torch.zeros(1, 3)
        self._base_com_to_cob_offsets = torch.zeros(1, 3)
        self._pid_value_add_buf = torch.zeros(1, 4)
        self._tracking_error_rpy_depth = torch.zeros(1, 4)
        self._robot = _DummyRobot()
        self._goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.cfg = _DummyCfg()


class _DummyEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self._inner = _DummyUnwrapped()

    @property
    def unwrapped(self):
        return self._inner

    def reset(self, *args, **kwargs):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


def _make(**kw) -> EasyUUVStdwWrapper:
    env = _DummyEnv()
    return EasyUUVStdwWrapper(
        env,
        drift_start_step=0,
        drift_end_step=10,
        target_drift=0.0,
        enable_filter=False,
        sim_dt_seconds=0.01,
        **kw,
    )


def _set_err(w: EasyUUVStdwWrapper, e: list[float]) -> None:
    w.env.unwrapped._tracking_error_rpy_depth = torch.tensor([e], dtype=torch.float32)


def test_pose_quadratic_matches_legacy() -> None:
    w = _make(lyapunov_v_mode="pose_quadratic", lyapunov_P_diag=(1.0, 1.0, 1.0, 1.0))
    _set_err(w, [0.2, 0.0, 0.0, 0.0])
    V0, m0, dV0, _ = w._compute_lyapunov_mask()
    expected = 0.5 * (0.2 ** 2)
    assert abs(V0 - expected) < 1e-6, f"V should be 0.5*e^2={expected}, got {V0}"
    assert m0 == 0.0, "first step (no V_prev) must not pass"
    w._V_prev = V0
    _set_err(w, [0.1, 0.0, 0.0, 0.0])
    V1, m1, dV1, _ = w._compute_lyapunov_mask()
    assert V1 < V0 and dV1 < 0.0 and m1 == 1.0, "decreasing error must pass sample_mask"
    print(f"[V-A pose_quadratic] V0={V0:.4f} V1={V1:.4f} dV={dV1:.4f} mask={m1}")


def test_so3_consistent_uses_geodesic() -> None:
    w = _make(lyapunov_v_mode="so3_consistent", lyapunov_P_diag=(1.0, 1.0, 1.0, 1.0))
    # set a goal 180deg about x (roll=pi); root identity -> geodesic = pi
    w.env.unwrapped._goal = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    e = w._read_so3_error_vec()
    assert abs(float(e[0].item()) - np.pi) < 1e-3, f"geodesic should be ~pi, got {e[0]}"
    assert float(e[1].item()) == 0.0 and float(e[2].item()) == 0.0, "channels 1/2 zeroed"
    # depth error component
    w.env.unwrapped._robot.data.root_pos_w = torch.tensor([[0.0, 0.0, 1.0]])
    e2 = w._read_so3_error_vec()
    assert abs(float(e2[3].item()) - 0.5) < 1e-6, f"depth err should be 0.5, got {e2[3]}"
    print(f"[V-B so3_consistent] e_att={float(e[0].item()):.4f} depth_err={float(e2[3].item()):.4f}")


def test_energy_with_rate_adds_kinetic() -> None:
    P = (1.0, 1.0, 1.0, 1.0)
    Q = (1.0, 1.0, 1.0, 1.0)
    # Static error: edot=0 -> V equals pose-only.
    w_static = _make(lyapunov_v_mode="energy_with_rate", lyapunov_P_diag=P, lyapunov_Q_diag=Q)
    _set_err(w_static, [0.3, 0.0, 0.0, 0.0])
    w_static._compute_lyapunov_mask()
    _set_err(w_static, [0.3, 0.0, 0.0, 0.0])  # unchanged -> edot=0
    V_static, _, _, _ = w_static._compute_lyapunov_mask()
    pose_only = 0.5 * (0.3 ** 2)
    assert abs(V_static - pose_only) < 1e-6, f"static error V should equal pose-only, got {V_static}"

    # Changing error: edot != 0 -> V strictly greater than pose-only.
    w_dyn = _make(lyapunov_v_mode="energy_with_rate", lyapunov_P_diag=P, lyapunov_Q_diag=Q)
    _set_err(w_dyn, [0.0, 0.0, 0.0, 0.0])
    w_dyn._compute_lyapunov_mask()
    _set_err(w_dyn, [0.3, 0.0, 0.0, 0.0])  # jumped -> large edot
    V_dyn, _, _, _ = w_dyn._compute_lyapunov_mask()
    assert V_dyn > pose_only + 1e-6, f"changing error must add kinetic energy, got {V_dyn} <= {pose_only}"
    print(f"[V-C energy_with_rate] V_static={V_static:.4f} (=pose {pose_only:.4f}) V_dyn={V_dyn:.4f} (>pose)")


def test_control_lyapunov_requires_exp_decay() -> None:
    # alpha=0.5: passing requires dV <= -0.5*V_prev, harder than dV<0.
    w = _make(lyapunov_v_mode="control_lyapunov", lyapunov_P_diag=(1.0, 1.0, 1.0, 1.0),
              lyapunov_decay_alpha=0.5)
    _set_err(w, [1.0, 0.0, 0.0, 0.0])
    V0, _, _, _ = w._compute_lyapunov_mask()  # V0 = 0.5
    # Small decrease: dV = -0.01*... insufficient for exp-decay gate.
    w._V_prev = V0
    w._e_prev = torch.tensor([1.0, 0.0, 0.0, 0.0])
    _set_err(w, [0.99, 0.0, 0.0, 0.0])
    V1, m_small, dV1, _ = w._compute_lyapunov_mask()
    assert dV1 < 0.0, "error did decrease"
    assert m_small == 0.0, "tiny decrease must FAIL exponential-decay gate"
    # Large decrease: should satisfy dV <= -0.5*V_prev.
    w._V_prev = V0
    w._e_prev = torch.tensor([1.0, 0.0, 0.0, 0.0])
    _set_err(w, [0.3, 0.0, 0.0, 0.0])
    V2, m_big, dV2, _ = w._compute_lyapunov_mask()
    assert m_big == 1.0, f"large decrease must PASS exp-decay gate (dV={dV2}, thr={-0.5*V0})"
    print(f"[V-D control_lyapunov] alpha=0.5 V0={V0:.4f}: small dV={dV1:.4f} mask={m_small} | "
          f"big dV={dV2:.4f} mask={m_big}")


def main() -> None:
    torch.manual_seed(0)
    test_pose_quadratic_matches_legacy()
    test_so3_consistent_uses_geodesic()
    test_energy_with_rate_adds_kinetic()
    test_control_lyapunov_requires_exp_decay()
    print("All M1 lyapunov-V tests PASSED.")


if __name__ == "__main__":
    main()
