from __future__ import annotations

from collections.abc import Sequence

import torch

import omni.isaac.lab.utils.math as math_utils


def _sample_uniform(bounds, shape, device, dtype=torch.float32) -> torch.Tensor:
    range_tensor = torch.as_tensor(bounds, device=device, dtype=dtype).flatten()
    if range_tensor.numel() != 2:
        raise ValueError(f"Expected a [low, high] range, got: {bounds!r}")
    return math_utils.sample_uniform(range_tensor[0], range_tensor[1], shape, device).to(dtype=dtype)


def _sample_axis_ranges(bounds, num_envs: int, device, dtype=torch.float32) -> torch.Tensor:
    range_tensor = torch.as_tensor(bounds, device=device, dtype=dtype)
    if range_tensor.shape == (2,):
        return _sample_uniform(bounds, (num_envs, 3), device, dtype=dtype)
    if range_tensor.shape != (3, 2):
        raise ValueError(
            "inertia_scale_range must be either [low, high] or [[x_low, x_high], [y_low, y_high], [z_low, z_high]]."
        )
    low = range_tensor[:, 0].reshape(1, 3).repeat(num_envs, 1)
    high = range_tensor[:, 1].reshape(1, 3).repeat(num_envs, 1)
    return math_utils.sample_uniform(low, high, low.shape, device).to(dtype=dtype)


def apply_expanded_domain_randomization(env, env_ids: Sequence[int]) -> None:
    dr_cfg = env.cfg.domain_randomization
    if not getattr(dr_cfg, "use_expanded_randomization", False):
        return

    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids_t.numel() == 0:
        return

    num_envs = int(env_ids_t.numel())

    mass_scales = _sample_uniform(getattr(dr_cfg, "mass_scale_range", [1.0, 1.0]), (num_envs, 1), env.device)
    env.masses[env_ids_t] = env._base_masses[env_ids_t] * mass_scales
    env._robot.root_physx_view.set_masses(
        env.masses[env_ids_t].detach().cpu(),
        env._robot._ALL_INDICES[env_ids_t].detach().cpu(),
    )

    inertia_scales = _sample_axis_ranges(getattr(dr_cfg, "inertia_scale_range", [1.0, 1.0]), num_envs, env.device)
    env.inertia_tensors[env_ids_t] = env._base_inertia_tensors[env_ids_t] * inertia_scales
    env.inertia_tensors_mean[env_ids_t] = env.inertia_tensors[env_ids_t].mean(dim=1, keepdim=True)

    dyn_time_constants = _sample_uniform(
        getattr(dr_cfg, "dyn_time_constant_range", [float(env.cfg.dyn_time_constant), float(env.cfg.dyn_time_constant)]),
        (num_envs,),
        env.device,
    )
    env.domain_dyn_time_constants[env_ids_t] = dyn_time_constants
    env.thruster_dynamics.tau = env.domain_dyn_time_constants

    drag_multipliers = _sample_uniform(getattr(dr_cfg, "drag_multiplier_range", [1.0, 1.0]), (num_envs, 1), env.device)
    env.drag_multipliers[env_ids_t] = drag_multipliers

    thruster_scales = _sample_uniform(
        getattr(dr_cfg, "thruster_com_offset_scale_range", [1.0, 1.0]),
        (num_envs, 1, 1),
        env.device,
    )
    env.thruster_com_offsets[env_ids_t] = env._base_thruster_com_offsets[env_ids_t] * thruster_scales

    water_rho = _sample_uniform(
        getattr(dr_cfg, "water_rho_range", [float(env.cfg.water_rho), float(env.cfg.water_rho)]),
        (num_envs, 1),
        env.device,
    )
    env.domain_water_rho[env_ids_t] = water_rho

    water_beta = _sample_uniform(
        getattr(dr_cfg, "water_beta_range", [float(env.cfg.water_beta), float(env.cfg.water_beta)]),
        (num_envs, 1),
        env.device,
    )
    env.domain_water_beta[env_ids_t] = water_beta

    rotor_constants = _sample_uniform(
        getattr(dr_cfg, "rotor_constant_range", [float(env.cfg.rotor_constant), float(env.cfg.rotor_constant)]),
        (num_envs, 1),
        env.device,
    )
    env.domain_rotor_constants[env_ids_t] = rotor_constants
    env.thruster_conversion.rotorConstant = env.domain_rotor_constants
