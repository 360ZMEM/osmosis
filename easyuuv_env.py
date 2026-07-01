"""
@file easyuuv_env.py
@brief EasyUUV Direct RL environment (4D ctrl baseline + 8D meta-control parametric).
@details
  Two cfg classes register two Gym tasks via __init__.py:
    - EasyUUVEnvCfg            -> EasyUUV-Direct-v1            (num_actions=4)
    - EasyUUVParametricEnvCfg  -> EasyUUV-Direct-Parametric-v1 (num_actions=8, tune_gains=True)
  The same EasyUUVEnv class handles both; the 8D path splits action[:, :4]
  as low-level control and action[:, 4:8] as a_gain inputs to ParametricGainTuner.

  Critical functions (search for "@dox"):
    - _pre_physics_step              splits 8D action into ctrl + a_gain
    - _apply_action                  routes ParametricGainTuner.step() into PID_args
    - _compute_dynamics              4-thruster mapping
    - get_current_fluid_velocity     JONSWAP wave injection point
    - _refresh_domain_randomization_defaults  zeta_nominal snapshot timing
    - _reset_idx                     gain_tuner.reset() hook

  Authors: Kevin Chang, Levi "Veevee" Cai (cail@mit.edu); STDW/meta extensions 2026-06.
"""

from __future__ import annotations

import random
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Mapping, Tuple

from .assets.warpauv import WARPAUV_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform, normalize
from omni.isaac.lab.markers import CUBOID_MARKER_CFG, VisualizationMarkers, RED_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul
import omni.isaac.lab.utils.math as math_utils

##
# Hydrodynamic model
##
from omni.isaac.lab.utils.math import quat_apply, quat_conjugate
from .rigid_body_hydrodynamics import HydrodynamicForceModels
from .boundary_effects import BoundaryEffectModels
from .thruster_dynamics import DynamicsFirstOrder, ConversionFunctionBasic, get_thruster_com_and_orientations
from .thrust_allocation import (
    ThrusterLayout,
    build_wrench_matrix,
    dof_weight_vector,
    control_channels_to_wrench,
    allocate as allocate_thrust,
)
from .wave_disturbance_manager import JonswapWaveDisturbanceManager

class EasyUUVEnvWindow(BaseEnvWindow):
    """Window manager for the EasyUUV environment."""

    def __init__(self, env: EasyUUVEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class EasyUUVEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = EasyUUVEnvWindow

    sim: SimulationCfg = SimulationCfg(dt=1 / 120)

    # robot
    robot_cfg: RigidObjectCfg = WARPAUV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=True)
    debug_vis = True

    # env
    decimation = 2
    cap_episode_length = True
    episode_length_s = 3.0
    episode_length_before_reset = None
    num_actions = 4 # 若不考虑原位约束则dim=4(yaw,pitch,roll,depth)，引入原位之后是5(vel)
    num_observations = 9 # 后续可能要修改
    num_states = 0
    use_boundaries = True
    max_auv_x = 7
    max_auv_y = 7
    max_auv_z = 7
    starting_depth = 1.5 # 深度符号有待确定！
    min_goal_steps = 100
    goal_completion_radius = 0.01
    goal_dims = 4
    eval_mode = False

    # === 下沉-then-flip360 近边界协议（opt-in，默认关=零行为变更，见 EMBODIMENT_ZOO §4）===
    # 所有 embodiment（含 base）先下沉+保持竖直（阶段 1），再 flip360（阶段 2），深度不触水面。
    # 安全关系式：submerge_depth < z_surface_guard − vehicle_height/2 − surface_margin（默认 <2.7）。
    submerge_phase_enable = False   # 开启 episode 内两相位调度
    submerge_depth = 1.5            # 阶段 1 目标下沉深度（世界 z，越小越深；经 spawn starting_depth 实现）
    submerge_hold_steps = 0         # 阶段 1 保持竖直的步数（episode_length_buf < 此值时锁竖直）
    surface_guard_enable = False    # 开启破面守卫
    surface_margin = 0.15           # 破面判据裕度
    z_surface_guard = 3.0           # 破面守卫的世界 Z 水面（与 boundary_z_surface 对齐）

    # === 参考轨迹（reference）生成 ===
    # "step"       : 旧行为，每个 episode 重置时给一个随机姿态硬阶跃（用于复现/基线）。
    # "sine_sweep" : 每轴用低频正弦平滑生成连续 roll/pitch/yaw 目标（depth 固定），
    #                曲线平滑无硬跳变，便于评估稳态跟踪与超调抑制。
    # "flip360_sine": roll/pitch 使用 ±π 连续正弦参考，用于 360 度后空翻压测。
    # "mixed_sine_flip360": 每个 env 在 ordinary sine 与 flip360 之间抽样，用于防遗忘调优。
    reference_mode = "step"
    # sine_sweep 逐轴幅度 (rad)：[roll, pitch, yaw]；课程式从低幅起步逐步上调。
    ref_sine_amp = [0.3, 0.3, 0.3]
    # sine_sweep 逐轴频率 (Hz)：[roll, pitch, yaw]；不同频率避免三轴同相。
    ref_sine_freq = [0.10, 0.13, 0.07]
    # mixed_sine_flip360 中抽到 flip360 子任务的 env 比例；剩余 env 使用 ref_mix_sine_*。
    ref_mix_flip_prob = 0.5
    ref_mix_sine_amp = [0.35, 0.35, 0.35]
    ref_mix_sine_freq = [0.18, 0.22, 0.15]

    class disturbance_cfg:
        mode = "none"
        base_vel = [0.0, 0.0, 0.0]
        amplitude = [0.0, 0.0, 0.0]
        frequency = [0.0, 0.0, 0.0]
        jonswap_hs = 0.5
        jonswap_fp = 0.1
        jonswap_gamma = 3.3
        jonswap_depth = 30.0
        jonswap_direction = 0.0
        jonswap_seed = 7

    goal_spawn_radius = 2.0
    init_guidance_rate = 0.1
    init_vel_max = 1.0

    # rewards
    rew_scale_terminated = 0.0
    rew_scale_alive = 0.0
    rew_scale_completion = 1000

    rew_scale_pos = 0.1
    rew_scale_ang = 0.5
    rew_scale_vel = 0.0
    rew_scale_ang_vel = 0.0
    rew_scale_lin_vel = 0.0
    rew_scale_actions = 0.00
    # 阻尼项（抑制超调/振荡）：默认 0 保持旧行为，课程式从极低幅度起步逐步上调。
    rew_scale_action_rate = 0.0   # 惩罚相邻动作差 ‖a_t - a_{t-1}‖，抑制控制抖动
    rew_scale_action_jerk = 0.0   # 惩罚动作二阶差（jerk），专门压制高频震荡

    # F4：难区间（接近倒置）姿态跟踪容差整形。默认关闭（relax=0）保持旧行为。
    # 当 goal 相对竖直的倾角进入 [band_lo, band_hi] 时，按 smoothstep 放宽 rew_ang
    # 的高斯宽度（等效降低该区误差惩罚斜率），避免始终在线的姿态项在物理不可达/不
    # 稳定的倒置目标上持续扣分、污染易控区策略。band 单位 rad（goal 倾角，0=正立，
    # pi=完全倒置）。物理依据：倒置区为浮力回正力矩的不稳定平衡，asym 机型 pitch 轴
    # 控制力矩预算 < 最坏回正力矩，部分倒置姿态静态不可达。
    flip_tol_relax = 0.0
    flip_tol_band_lo = 2.094395    # 120 deg
    flip_tol_band_hi = 3.1415926   # 180 deg

    # dynamics
    com_to_cob_offset = [0.0, 0.0, 0.01] # in meters, add this (xyz) to COM to get COB location
    water_rho = 997.0 # kg/m^3
    water_beta = 0.001306 # Pa s, dynamic viscosity of water @ 50 deg F
    rotor_constant = 0.1 / 100.0 # rotor constant used in Gazebo, note /10 because 0.04 is "10x bigger than it should be"
    dyn_time_constant = 0.05 # time constant for linear dynamics for each rotor 
    volume = 0.022747843530591776 # assuming cubic meters - NEUTRALLY BOUYANT. In orignal sim file volume = 0.0223
    # volume = 0.03
    mass = 2.2701e+01 # kg
    PID_PWM_value = 0.6 # 这里的正则化相当于实际的250
    PID_init_args = [[0.6 / PID_PWM_value, 0.08/ PID_PWM_value, 0 / PID_PWM_value], # x-axis rotation, Roll
                     [0.6 / PID_PWM_value, 0.08/ PID_PWM_value, 0 / PID_PWM_value], # y-axis rotation, Pitch
                     [1.0 / PID_PWM_value, 0.13/ PID_PWM_value, 0 / PID_PWM_value], # z-axis rotation, Yaw (B1 已回退至原始增益；当前唯一改动为 D1×1/4)
                     [0.16 / PID_PWM_value, 0.07/ PID_PWM_value, 0 / PID_PWM_value]] # depth

    # A3：S 面 D 项是否改用真实 body 角速度（roll/pitch/yaw 三轴），depth 维仍用动作差分。
    # 默认 False 保持基类行为；Parametric cfg 中开启。
    d_use_ang_vel = False
    # A3：观测是否拼接 root_ang_vel_b (3 维)。开启时需同步把 num_observations 从 9 调到 12。
    obs_include_ang_vel = False
    # A1：S 面 D 项一阶 EMA 低通时间常数 (秒)。0.0 = 关闭。tau=0.05 对应截止 ~3.2Hz，
    # 刚好压在 A3 当前 ripple 主频 1.8-2.4Hz 上方一档，抑制残余高频抖动。
    d_filter_tau = 0.0
    # A5：D 项低通后的相位超前预测，补偿 EMA 滞后。默认 0.0 完全关闭，保持旧行为。
    d_filter_phase_lead_steps = 0.0
    d_filter_phase_lead_clip = 0.0

    # A5：运行期直接力矩脉冲扰动。默认关闭，仅用于 STDW 强验证。
    torque_pulse_enable = False
    torque_pulse_level = 0.0
    torque_pulse_interval_min_s = 3.0
    torque_pulse_interval_max_s = 5.0
    torque_pulse_duration_s = 0.25
    torque_pulse_axes = [1.0, 1.0, 1.0]
    torque_pulse_seed = 13
    # A6：运行期推进器角度偏移。默认关闭，仅用于动态参数辨识学术诊断。
    thruster_angle_shift_enable = False
    thruster_angle_shift_rad = 0.0
    thruster_angle_shift_thrusters = [4]
    thruster_angle_shift_axis = "yaw"
    
    # 使用控制方法
    control_method = 'Ssurface' # Ssurface & PID
    s_ratio = 4 # (Ssurface coeff / PID coeff) — B2(4→2)曾试，但削弱全局控制权威致 Depth 崩溃，已回退
    cascade_control = True # True: PID/S-surface cascade, False: RL direct PWM duty cycle
    self_adapt = False # True: enable A-S-Surface adaptation term when control_method == 'Ssurface'
    gain_update_targets = ["zeta1"] # configurable online gain targets; S-surface commonly uses zeta1/zeta2

    # === SO(3) 流形 S-Surface 升级（Phase 1，默认 euler = 旧行为，零行为变更）===
    # "euler": 旧逐轴解耦（policy 4 通道直接进 sigmoid）；"so3": 在 SO(3) 流形上算
    # e_R（四元数误差虚部）与 e_ω（body 角速度 − 参考角速度），三轴向量化 S 面。
    attitude_error_mode = "euler"
    # 取代硬编码的 action_lim（line ~369），改为可由 YAML 设置；so3 模式抬高 roll/pitch 解封力矩。
    action_lim_vec = [0.15, 0.15, 0.3, 1.0]
    geo_zeta1 = 1.0          # e_R 增益 ζ1（进 sigmoid）
    geo_zeta2 = 0.5          # e_ω 增益 ζ2（进 sigmoid）
    geo_residual_scale = 0.0 # RL 残差对 so3 S 面输入的调制（0=纯解析；>0=残差 RL，保 12 维 obs）
    # 固定手工分配矩阵 (line ~1610) 对 roll/yaw 通道的 PID_value→body torque 是反号的：
    # 数值标定（单位命令的净 body 力矩）：roll τx=-0.84、pitch τy=+0.516、yaw τz=-0.56。
    # e_R[i]>0 表示需绕 body 轴 i 施加 +τ，故须乘该符号向量把解析 S 面输出转成 restoring。
    # 若不加，纯解析基线在 roll/yaw 上是反阻尼（发散），会毁掉从 model_2846 的 fine-tune。
    geo_channel_sign = [-1.0, 1.0, -1.0]

    # === M3: controller mismatch injection (plug-and-play, default = pid_gain 现状) ===
    # 旧 --pid_multipliers 路径属于 pid_gain 模式（缩 ζ 增益），已知会被 sigmoid 饱和吞掉，
    # 故 STDW 相对改善恒定在 ~67%。新增三种作用在「对输出真正敏感」参数上的失配靶点：
    #   - actuator_scale  : 直接乘 motorValue→thrust 线性段（绕开 sigmoid 饱和）。
    #   - s_surface_struct: 扰动 s_ratio 斜率与加性项权威（改变控制律结构）。
    #   - allocation_skew : 扰动 roll/pitch/yaw/depth 通道分配权重（推进器安装偏差）。
    # 默认 pid_gain 时所有 mismatch 缩放为恒等，零行为变更。
    ctrl_mismatch_mode = "pid_gain"

    # === M4: near-boundary effects (plug-and-play, default = off 完全旁路) ===
    # 见 boundary_effects.BoundaryEffectModels 与 ref/近边界效应.md。
    #   off                : 现状，无任何边界力（默认，零行为变更）。
    #   residual_buoyancy  : 微正浮力配平偏差 ΔB=frac·mg（机制一）。
    #   free_surface       : 自由液面分段浮力/阻尼 ×s(t) + 推进器吸气（机制二）。
    #   ground_effect      : 近底吸力 F=F_nom·(D/h)^γ（机制三）。
    #   nonlinear_restoring: 显式 COG/COB 偏置恢复力矩，强化 180° 失稳（机制一.2）。
    #   full               : 以上全部叠加。
    boundary_effect_mode = "off"
    boundary_residual_buoyancy_frac = 0.015  # +上浮，相对体重 m·g 的比例
    boundary_z_surface = 3.0                 # 世界 Z 水面高度
    boundary_z_bottom = 0.0                  # 世界 Z 池底高度
    boundary_vehicle_height = 0.3            # UUV 垂直高度 H
    boundary_ground_effect_coeff = 0.15      # F_nom = coeff·mg
    boundary_ground_effect_gamma = 2.0       # 近地效应指数 γ
    boundary_ground_effect_threshold = 0.5   # 触发近地效应的离底阈值 (m)
    boundary_r_cog = [0.0, 0.0, 0.0]         # B5 显式 COG body 偏置
    boundary_r_cob = [0.0, 0.0, 0.0]         # B5 显式 COB body 偏置

    # === Parametric meta-control (8-dim action) ===
    # 主开关；True 时必须配合 num_actions=8 的子 cfg / 任务 ID 使用。
    # 详见 gain_tuner.ParametricGainTuner 与 EasyUUVParametricEnvCfg。
    tune_gains = False
    gain_beta = 0.2                 # Bounded Safeguard 因子（±β 标称值波动）
    enable_pe = True
    pe_freq = 0.5                   # Hz, PE 注入正弦频率
    pe_amp = 0.05                   # PE 振幅（相对 ζ_nominal）
    pe_decay_gamma = 5.0            # 状态相关衰减强度 a(t)=pe_amp/(1+γ‖ω‖²)
    enable_deadzone = True
    deadzone_threshold = 0.02       # rad/s, 死区阈值（与 ‖ω_body‖ 同量纲）
    enable_param_lpf = True
    param_lpf_cutoff = 1.0          # Hz, 一阶 LPF 截止频率
    identity_init = False           # 幂等初始化（降难度，整个调制器旁路）

    class noise_cfg:
        enable_noise = True
        std_dev = 0.02
        correlation_coeff = 0.8
        # A4：仅作用于 obs 末 3 维 root_ang_vel_b 的额外白噪声（rad/s），
        # 用于模拟真实 IMU 陀螺噪声等级。默认 0.0 即等价旧行为（仅 std_dev 全局噪声）。
        # 仅当 obs_include_ang_vel=True 且该值 > 0 时生效。
        ang_vel_extra_std = 0.0


     # domain randomization
    # todo: isaaclabs has a built-in method somehow
    class domain_randomization:
        use_custom_randomization = True
        # com_to_cob_offset_radius = 0 # uniform from sphere around predicted com_to_cob_offset
        # volume_range = [0.022747843530591776, 0.022747843530591776] # uniform [lowerbound, upperbound]
        # mass_range = [2.2701e+0,2.2701e+0] # uniform [lowerbound, upperbound]
        com_to_cob_offset_radius = 0.05 # uniform from sphere around predicted com_to_cob_offset
        com_to_cob_offset_xyz_range = [0.0, 0.0, 0.0] # per-axis symmetric uniform range around the base offset
        volume_range = [0.019747843530591773, 0.02574784353059178] # uniform [loierbound, upperbound]
        mass_range = [2.2701e+01,2.2701e+01] # uniform [lowerbound, upperbound]
        mass_scale_range = [1.0, 1.0] # multiplicative mass randomization around the base embodiment
        inertia_scale_range = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]] # per-axis multiplicative randomization
        dyn_time_constant_scale_range = [1.0, 1.0] # multiplicative randomization for thruster first-order dynamics
        drag_multiplier_scale_range = [1.0, 1.0] # multiplicative randomization around the current embodiment drag
        thruster_com_offset_scale_range = [1.0, 1.0] # multiplicative scaling for thruster arm geometry
        PID_PWM_value = 0.6 # 后面再调整
        PID_scale_range = [[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]]
        PID_adjust_range = [[0.2 * 0.01/ PID_PWM_value, 0.06 * 0.01/ PID_PWM_value, 0 / PID_PWM_value], # x-axis rotation, Roll
                     [0.2 * 0.01/ PID_PWM_value, 0.06 * 0.01/ PID_PWM_value, 0 / PID_PWM_value], # y-axis rotation, Pitch
                     [0.3 * 0.01/ PID_PWM_value, 0.12 * 0.01/ PID_PWM_value, 0 / PID_PWM_value], # z-axis rotation, Yaw
                     [0.06 * 0.01/ PID_PWM_value, 0.05 * 0.01/ PID_PWM_value, 0 / PID_PWM_value]]

    # Embodiment configurations for cross-embodiment generalization experiments
    # 四种机型的物理参数配置
    embodiment_configs = {
        "base": {
            "mass": 2.2701e+01,
            "inertia_tensors": [0.37, 0.97, 1.19],
            "com_to_cob_offset": [0.0, 0.0, 0.01],
            "dyn_time_constant": 0.05,
            "drag_multiplier": 1.0,
        },
        "long_body": {
            "mass": 2.2701e+01,
            "inertia_tensors": [0.1, 2.5, 2.5],  # Roll 转动惯量减小，Pitch/Yaw 增大
            "com_to_cob_offset": [0.0, 0.0, 0.01],
            "dyn_time_constant": 0.05,
            "drag_multiplier": 1.0,
            "thruster_com_offset_scale": 1.2,  # 增大推进器力臂
        },
        "heavy_duty": {
            "mass": 2.2701e+01 * 5,  # 质量增加 5 倍
            "inertia_tensors": [0.37 * 5, 0.97 * 5, 1.19 * 5],  # 转动惯量按比例增加
            "com_to_cob_offset": [0.0, 0.0, 0.01],
            "dyn_time_constant": 0.2,  # 增大时间常数，模拟更重的响应
            "drag_multiplier": 5.0,  # 阻尼系数增加 5 倍
        },
        "heavy_moderate": {
            "mass": 2.2701e+01 * 2,  # 质量增加 2 倍
            "inertia_tensors": [0.37 * 2, 0.97 * 2, 1.19 * 2],  # 转动惯量按比例增加
            "com_to_cob_offset": [0.0, 0.0, 0.01],
            "dyn_time_constant": 0.1,  # 时间常数 ×2
            "drag_multiplier": 2.0,  # 阻尼系数增加 2 倍
        },
        "asymmetric": {
            "mass": 2.2701e+01,
            "inertia_tensors": [0.37, 0.97, 1.19],
            "com_to_cob_offset": [0.05, 0.05, 0.01],  # X 和 Y 轴增加随机偏移
            "dyn_time_constant": 0.05,
            "drag_multiplier": 1.0,
        },
        # === 多 embodiment 扩展（config 驱动 B⁺，opt-in，见 docs/guide/EMBODIMENT_ZOO.md）===
        # 每项在既有键之外声明可选键 num_thrusters/volume/action_lim_vec/thrust_allocation，
        # 触发 apply_embodiment_config 的 config 分配路径（_use_config_alloc=True）。
        # thrust_allocation.layout_specs 行 = [x, y, z, roll, pitch, yaw]（com->thruster 偏移 + rpy 朝向）。
        # 垂直推进器 pitch=-1.5708 (-90°) 承 roll/pitch/depth；水平推进器承 yaw。
        # ρ_body = mass/volume ≈ 990 < 997（水），微正浮力，满足近边界效应假设（D6）。
        "uuv6": {  # 全驱 6 推（4 垂直 + 2 水平纯 yaw 力偶），三控制轴正交
            "mass": 29.70,
            "inertia_tensors": [0.55, 1.45, 1.78],
            "com_to_cob_offset": [0.0, 0.0, 0.01],
            "dyn_time_constant": 0.05,
            "drag_multiplier": 1.0,
            "num_thrusters": 6,
            "volume": 0.030000,
            "action_lim_vec": [0.6, 0.6, 0.3, 1.0],
            "thrust_allocation": {
                "layout_specs": [
                    [0.129, 0.21, 0.03, 0.0, -1.5708, 0.0],    # 0 front_left_vertical
                    [0.129, -0.21, 0.03, 0.0, -1.5708, 0.0],   # 1 front_right_vertical
                    [-0.129, 0.21, 0.03, 0.0, -1.5708, 0.0],   # 2 rear_left_vertical
                    [-0.129, -0.21, 0.03, 0.0, -1.5708, 0.0],  # 3 rear_right_vertical
                    [0.0, 0.16125, -0.02, 0.0, 0.0, 0.0],      # 4 left_horizontal (+x)
                    [0.0, -0.16125, -0.02, 0.0, 0.0, 0.0],     # 5 right_horizontal (+x) -> 差分成 yaw 力偶
                ],
                "mode": "pinv",
                "controllable_dofs": ["heave", "roll", "pitch", "yaw"],
            },
        },
        "uuv4": {  # 欠驱动 4 推（仅 4 垂直），舍弃 yaw，只满足 roll/pitch/depth
            "mass": 21.78,
            "inertia_tensors": [0.36, 0.94, 1.15],
            "com_to_cob_offset": [0.0, 0.0, 0.01],
            "dyn_time_constant": 0.05,
            "drag_multiplier": 1.0,
            "num_thrusters": 4,
            "volume": 0.022000,
            "action_lim_vec": [0.6, 0.6, 0.3, 1.0],
            "thrust_allocation": {
                "layout_specs": [
                    [0.129, 0.21, 0.03, 0.0, -1.5708, 0.0],
                    [0.129, -0.21, 0.03, 0.0, -1.5708, 0.0],
                    [-0.129, 0.21, 0.03, 0.0, -1.5708, 0.0],
                    [-0.129, -0.21, 0.03, 0.0, -1.5708, 0.0],
                ],
                "mode": "wls",
                "controllable_dofs": ["heave", "roll", "pitch"],  # 屏蔽 yaw（Tz）
            },
        },
        "uuv6_angled": {  # 全驱 6 推非正交：垂直推进器带 ±8° roll 倾角、水平带 10° pitch 倾角
            "mass": 31.68,
            "inertia_tensors": [0.60, 1.55, 1.90],
            "com_to_cob_offset": [0.0, 0.0, 0.01],
            "dyn_time_constant": 0.05,
            "drag_multiplier": 1.0,
            "num_thrusters": 6,
            "volume": 0.032000,
            "action_lim_vec": [0.6, 0.6, 0.3, 1.0],
            "thrust_allocation": {
                "layout_specs": [
                    [0.129, 0.21, 0.03, 0.139626, -1.5708, 0.0],
                    [0.129, -0.21, 0.03, -0.139626, -1.5708, 0.0],
                    [-0.129, 0.21, 0.03, 0.139626, -1.5708, 0.0],
                    [-0.129, -0.21, 0.03, -0.139626, -1.5708, 0.0],
                    [0.0, 0.16125, -0.02, 0.0, 0.174533, 0.0],
                    [0.0, -0.16125, -0.02, 0.0, 0.174533, 0.0],
                ],
                "mode": "pinv",
                "controllable_dofs": ["heave", "roll", "pitch", "yaw"],
            },
        },
        "uuv4_angled": {  # 欠驱动 4 推非正交：垂直推进器带 ±8° roll 倾角，舍弃 yaw
            "mass": 23.76,
            "inertia_tensors": [0.40, 1.02, 1.25],
            "com_to_cob_offset": [0.0, 0.0, 0.01],
            "dyn_time_constant": 0.05,
            "drag_multiplier": 1.0,
            "num_thrusters": 4,
            "volume": 0.024000,
            "action_lim_vec": [0.6, 0.6, 0.3, 1.0],
            "thrust_allocation": {
                "layout_specs": [
                    [0.129, 0.21, 0.03, 0.139626, -1.5708, 0.0],
                    [0.129, -0.21, 0.03, -0.139626, -1.5708, 0.0],
                    [-0.129, 0.21, 0.03, 0.139626, -1.5708, 0.0],
                    [-0.129, -0.21, 0.03, -0.139626, -1.5708, 0.0],
                ],
                "mode": "wls",
                "controllable_dofs": ["heave", "roll", "pitch"],  # 屏蔽 yaw（Tz）
            },
        },
    }


class EasyUUVEnv(DirectRLEnv):
    cfg: EasyUUVEnvCfg

    def __init__(self, cfg: EasyUUVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode?
        self._debug = False

        # Initialize buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.cfg.goal_dims, device=self.device)
        self._default_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._completion_buffer = torch.zeros(self.num_envs, device=self.device)
        self._completed_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._default_env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_pos_w = self._default_env_origins # just for visualizations at the moment
        self._step_count = 0
        # sine_sweep 参考：逐 env 逐轴随机相位，使每个 episode 的连续目标各不相同。
        self._ref_phase = torch.zeros(self.num_envs, 3, device=self.device)
        self._ref_is_flip = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # SO(3) 控制：上一步参考四元数，用于有限差分得到参考角速度 ω_d（避免欧拉率奇异）。
        self._goal_prev = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_prev[:, 0] = 1.0  # 单位四元数初值
        # 动作平滑/jerk 阻尼用：上一步与上上步动作（reward 项消费）。
        self._prev_action = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._prev_prev_action = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.obs_noise_buffer = torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device)
        
        # Get thruster configurations
        self.thruster_com_offsets, self.thruster_quats = get_thruster_com_and_orientations(self.device)
        self.thruster_com_offsets = self.thruster_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
        self.thruster_quats = self.thruster_quats.repeat(self.num_envs, 1).to(self.device)
        self._base_thruster_quats = self.thruster_quats.clone()

        # 配置化推力分配（TAM）状态。默认走旧硬编码 8 推混合块（_use_config_alloc=False），
        # 仅当 embodiment config 声明 thrust_allocation 时切到 config B⁺ 路径（apply_embodiment_config）。
        self._num_thrusters = 8
        self._use_config_alloc = False
        self._alloc_B = None          # (6, N) 分配矩阵
        self._alloc_mode = "pinv"     # "pinv" | "wls"
        self._alloc_weight = None     # (6,) WLS 对角权重（欠驱动屏蔽不可控 DOF）

        # 新增：PID
        self.PID_args = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.old_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self.actions_i = torch.zeros(self.num_envs, 4, device=self.device) # 积分项不使用
        self.action_lim = torch.tensor(self.cfg.action_lim_vec, dtype=torch.float, device=self.device).reshape(1, 4)
        # SO(3) S 面：固定分配矩阵对 roll/yaw 反号，乘 restoring 符号向量修正（数值标定 [-1,+1,-1]）。
        self._geo_channel_sign = torch.tensor(
            getattr(self.cfg, "geo_channel_sign", [-1.0, 1.0, -1.0]),
            dtype=torch.float, device=self.device,
        ).reshape(1, 3)
        # A1：S 面 D 项 EMA 低通状态。仅在 d_filter_tau > 0 时使用。
        self._actions_d_filt = torch.zeros(self.num_envs, 4, device=self.device)
        self._actions_d_filt_prev = torch.zeros(self.num_envs, 4, device=self.device)

        # 记录数值
        self.log_MSE = torch.zeros(self.num_envs, dtype=torch.float, device=self.device) # 记录每个env的MSE

        torch.manual_seed(0)

        if self.cfg.eval_mode:
            print("Setting manual seed")
            torch.manual_seed(0)

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        if self._debug: print("mass: ", list(self._robot.root_physx_view._masses))

        # Get specific information about the AUV
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()

        # todo: get inertias from the model or physx view
        self.inertia_tensors = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float, requires_grad=False)

        # estimated inertial values from a solid rect. prism model (with estimated side lengths of 0.7m, 0.4m, and 0.2m):
        # fake inertial values for the AUV, based on I_ii = (1/12) * mass * (len_j**2 + len_k**2)
        self.inertia_tensors[:, 0] = 0.37
        self.inertia_tensors[:, 1] = 0.97
        self.inertia_tensors[:, 2] = 1.19

        if self.cfg.mass:
            self.masses = torch.full((self.num_envs, 1), self.cfg.mass, device=self.device)
        else:
            self.masses = self._robot.root_physx_view._masses

        # todo: cleaner way to handle this
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
            self.com_to_cob_offsets = torch.tensor(self.cfg.com_to_cob_offset).repeat(self.num_envs, 1).to(self.device)
        else:
            self.com_to_cob_offsets = self.cfg.com_to_cob_offset.copy()

        if type(self.cfg.volume) != torch.Tensor:
            self.volumes = torch.full((self.num_envs, 1), self.cfg.volume, device=self.device)
        else:
            self.volumes = self.cfg.volume.copy()

        self._base_water_rho = float(self.cfg.water_rho)

        # PID
        self.PID_args = torch.tensor(self.cfg.PID_init_args).reshape(1, 4, 3).repeat(self.num_envs, 1, 1).to(self.device)

        self.inertia_tensors_mean = self.inertia_tensors.mean(dim=1, keepdim=True) 

        # Thruster fault injection state
        self.thruster_efficiency_factors = torch.ones((self.num_envs, 8), device=self.device)
        self.fault_injection_enabled = False
        self.fault_start_sim_time = 0.0
        self.fault_thrusters = [4, 5]
        self.fault_rate_per_second = 0.02
        self.fault_profile = "rate"
        self.fault_target_efficiency = 0.0
        self.fault_ramp_duration_s = 0.0

        # === M3: controller mismatch persistent state (default = identity, 零行为变更) ===
        # mode 决定失配靶点；缩放因子默认 1.0（恒等）。apply_ctrl_mismatch() 写入这些 buffer，
        # _pid_control / _compute_dynamics 在对应 hook 处乘入。
        self.ctrl_mismatch_mode = str(getattr(self.cfg, "ctrl_mismatch_mode", "pid_gain"))
        # CM-B actuator_scale: 每推进器乘性失配（绕开 sigmoid 饱和，作用在 thrust 线性段）。
        self._mismatch_actuator_scale = torch.ones((self.num_envs, 8), device=self.device)
        # CM-C s_surface_struct: sigmoid 斜率与加性项权威缩放（标量）。
        self._mismatch_s_ratio_scale = 1.0
        self._mismatch_add_scale = 1.0
        # CM-D allocation_skew: 4 控制通道 (roll/pitch/yaw/depth) 分配权重缩放。
        self._mismatch_alloc_scale = torch.ones((self.num_envs, 4), device=self.device)

        # === M4: near-boundary effects (plug-and-play, default = off 完全旁路) ===
        # BoundaryEffectModels 仅返回加性 body-frame wrench 修正，在 _compute_dynamics
        # 合成口加入；推进器吸气返回每推进器 efficiency 乘子。默认 off → any_enabled=False
        # → 全部旁路，零行为变更。set_boundary_effect_mode() 可运行期切换。
        self.boundary_models = BoundaryEffectModels(
            num_envs=self.num_envs,
            device=self.device,
            residual_buoyancy_frac=float(getattr(self.cfg, "boundary_residual_buoyancy_frac", 0.015)),
            z_surface=float(getattr(self.cfg, "boundary_z_surface", 3.0)),
            z_bottom=float(getattr(self.cfg, "boundary_z_bottom", 0.0)),
            vehicle_height=float(getattr(self.cfg, "boundary_vehicle_height", 0.3)),
            ground_effect_coeff=float(getattr(self.cfg, "boundary_ground_effect_coeff", 0.15)),
            ground_effect_gamma=float(getattr(self.cfg, "boundary_ground_effect_gamma", 2.0)),
            ground_effect_threshold=float(getattr(self.cfg, "boundary_ground_effect_threshold", 0.5)),
            r_cog=tuple(getattr(self.cfg, "boundary_r_cog", [0.0, 0.0, 0.0])),
            r_cob=tuple(getattr(self.cfg, "boundary_r_cob", [0.0, 0.0, 0.0])),
        )
        self.boundary_models.apply_mode(str(getattr(self.cfg, "boundary_effect_mode", "off")))
        self._last_boundary_info: dict = {}

        self.thruster_angle_shift_enabled = False
        self.thruster_angle_shift_thrusters = [4]
        self.thruster_angle_shift_rad = 0.0
        self.thruster_angle_shift_axis = "yaw"
        self._torque_pulse_next_t = torch.zeros(self.num_envs, device=self.device)
        self._torque_pulse_end_t = torch.zeros(self.num_envs, device=self.device)
        self._torque_pulse_vec_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._torque_pulse_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            self._torque_pulse_generator = torch.Generator(device=self.device)
            self._torque_pulse_generator_device = torch.device(self.device)
        except TypeError:
            self._torque_pulse_generator = torch.Generator()
            self._torque_pulse_generator_device = torch.device("cpu")
        self._torque_pulse_generator.manual_seed(int(getattr(self.cfg, "torque_pulse_seed", 13)))
        self._schedule_next_torque_pulse(torch.arange(self.num_envs, device=self.device), initial=True)
        self._wave_manager = None
        self._wave_manager_signature = None

        # Initialize dynamics calculators
        self._init_thruster_dynamics()

        # === Parametric meta-control state（仅在 num_actions==8 时启用） ===
        self._tune_gains_enabled = bool(getattr(self.cfg, "tune_gains", False)) and self.cfg.num_actions == 8
        self._a_gain_buf = torch.zeros(self.num_envs, 4, device=self.device)
        self._zeta_nominal = torch.zeros(self.num_envs, 4, device=self.device)
        self._zeta_runtime = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_pe_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_deadzone_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._sim_time_s = torch.zeros(self.num_envs, device=self.device)
        if self._tune_gains_enabled:
            from .gain_tuner import ParametricGainTuner
            decision_dt = float(self.sim.cfg.dt) * float(self.cfg.decimation)
            self._gain_tuner = ParametricGainTuner(
                num_envs=self.num_envs,
                num_axes=4,
                dt=decision_dt,
                device=self.device,
                gain_beta=float(self.cfg.gain_beta),
                enable_pe=bool(self.cfg.enable_pe),
                pe_freq=float(self.cfg.pe_freq),
                pe_amp=float(self.cfg.pe_amp),
                pe_decay_gamma=float(self.cfg.pe_decay_gamma),
                enable_deadzone=bool(self.cfg.enable_deadzone),
                deadzone_threshold=float(self.cfg.deadzone_threshold),
                enable_param_lpf=bool(self.cfg.enable_param_lpf),
                param_lpf_cutoff=float(self.cfg.param_lpf_cutoff),
                identity_init=bool(self.cfg.identity_init),
            )
        else:
            self._gain_tuner = None

        self._refresh_domain_randomization_defaults()
        
        # Set initial goals
        self._reset_idx(self._robot._ALL_INDICES)

    def _init_thruster_dynamics(self):
        if type(self.cfg.com_to_cob_offset) != torch.Tensor:
          self.cfg.com_to_cob_offset = torch.tensor(self.cfg.com_to_cob_offset, device=self.device, dtype=torch.float32, requires_grad=False).reshape(1,3).repeat(self.num_envs, 1)

        # get force calculation functions and rotor dynamics models
        self.force_calculation_functions = HydrodynamicForceModels(self.num_envs, self.device, False)
        self.thruster_dynamics = DynamicsFirstOrder(self.num_envs, 8, self.cfg.dyn_time_constant, self.device)
        self.thruster_conversion = ConversionFunctionBasic(self.cfg.rotor_constant)

        # Embodiment type for cross-embodiment experiments
        self._embodiment_type = "base"
        self._drag_multiplier = 1.0
        self._drag_multiplier_per_env = torch.ones(self.num_envs, device=self.device)

    def apply_control_profile(self, control_profile: str) -> None:
        """Apply a control profile for cascade/direct PWM experiments."""
        normalized_profile = control_profile.strip().lower()
        if normalized_profile in {"direct_pwm", "direct", "rl_direct"}:
            self.cfg.cascade_control = False
            self.cfg.control_method = "DirectPWM"
            self.cfg.self_adapt = False
        elif normalized_profile in {"a-s-surface", "asurface", "adaptive_ssurface"}:
            self.cfg.cascade_control = True
            self.cfg.control_method = "Ssurface"
            self.cfg.self_adapt = True
        elif normalized_profile in {"s-surface", "ssurface"}:
            self.cfg.cascade_control = True
            self.cfg.control_method = "Ssurface"
            self.cfg.self_adapt = False
        elif normalized_profile == "pid":
            self.cfg.cascade_control = True
            self.cfg.control_method = "PID"
            self.cfg.self_adapt = False
        else:
            raise ValueError(
                f"Unknown control profile: {control_profile}. "
                "Supported profiles: direct_pwm, A-S-Surface, S-Surface, PID"
            )

        self.old_actions.zero_()
        self.actions_i.zero_()
        self.PID_args = torch.tensor(self.cfg.PID_init_args).reshape(1, 4, 3).repeat(self.num_envs, 1, 1).to(self.device)
        self._refresh_domain_randomization_defaults()
        print(f"Applied control profile: {control_profile} -> cascade={self.cfg.cascade_control}, "
              f"method={self.cfg.control_method}, self_adapt={self.cfg.self_adapt}")

    def set_thruster_fault(
        self,
        enabled: bool,
        start_sim_time: float = 20.0,
        fault_thrusters: list[int] | None = None,
        fault_rate_per_second: float = 0.02,
        fault_profile: str = "rate",
        target_efficiency: float = 0.0,
        ramp_duration_s: float = 0.0,
    ) -> None:
        """Configure a runtime thruster efficiency fault model."""
        self.fault_injection_enabled = enabled
        self.fault_start_sim_time = float(start_sim_time)
        self.fault_thrusters = list(fault_thrusters) if fault_thrusters is not None else [4, 5]
        self.fault_rate_per_second = float(fault_rate_per_second)
        profile = str(fault_profile).lower()
        if profile not in {"rate", "fixed", "ramp_to_target"}:
            raise ValueError(f"fault_profile must be rate/fixed/ramp_to_target, got {fault_profile!r}")
        self.fault_profile = profile
        self.fault_target_efficiency = float(max(0.0, min(1.0, target_efficiency)))
        self.fault_ramp_duration_s = float(max(0.0, ramp_duration_s))
        self.thruster_efficiency_factors.fill_(1.0)

    def get_thruster_efficiency_factors(self) -> torch.Tensor:
        return self.thruster_efficiency_factors.clone()

    def _thruster_angle_axis_vector(self, axis: str) -> torch.Tensor:
        axis = str(axis).lower()
        if axis == "roll":
            vec = [1.0, 0.0, 0.0]
        elif axis == "pitch":
            vec = [0.0, 1.0, 0.0]
        elif axis == "yaw":
            vec = [0.0, 0.0, 1.0]
        else:
            raise ValueError(f"thruster angle axis must be roll/pitch/yaw, got {axis!r}")
        return torch.tensor(vec, device=self.device, dtype=torch.float32)

    def set_thruster_angle_shift(
        self,
        enabled: bool,
        thrusters: list[int] | None = None,
        angle_rad: float = 0.0,
        axis: str = "yaw",
    ) -> None:
        """Apply a runtime thruster direction offset for parameter-ID diagnostics."""
        self.thruster_angle_shift_enabled = bool(enabled)
        self.thruster_angle_shift_thrusters = list(thrusters) if thrusters is not None else [4]
        self.thruster_angle_shift_rad = float(angle_rad) if enabled else 0.0
        self.thruster_angle_shift_axis = str(axis).lower()

        self.thruster_quats = self._base_thruster_quats.clone()
        if not self.thruster_angle_shift_enabled or abs(self.thruster_angle_shift_rad) <= 0.0:
            return

        axis_vec = self._thruster_angle_axis_vector(self.thruster_angle_shift_axis).reshape(1, 3)
        angle = torch.tensor([self.thruster_angle_shift_rad], device=self.device, dtype=torch.float32)
        delta_q = quat_from_angle_axis(angle, axis_vec).reshape(1, 4)
        quat_view = self.thruster_quats.reshape(self.num_envs, 8, 4)
        base_view = self._base_thruster_quats.reshape(self.num_envs, 8, 4)
        for thruster_idx in self.thruster_angle_shift_thrusters:
            idx = int(thruster_idx)
            if idx < 0 or idx >= 8:
                raise ValueError(f"thruster index out of range [0,7]: {idx}")
            quat_view[:, idx, :] = quat_mul(delta_q.expand(self.num_envs, 4), base_view[:, idx, :])

    def get_thruster_angle_shift_state(self) -> tuple[bool, float, str, list[int]]:
        return (
            bool(self.thruster_angle_shift_enabled),
            float(self.thruster_angle_shift_rad),
            str(self.thruster_angle_shift_axis),
            list(self.thruster_angle_shift_thrusters),
        )

    def _schedule_next_torque_pulse(self, env_ids: torch.Tensor, *, initial: bool = False) -> None:
        """Schedule the next runtime torque pulse for selected envs."""
        if env_ids.numel() == 0:
            return
        min_s = float(getattr(self.cfg, "torque_pulse_interval_min_s", 3.0))
        max_s = float(getattr(self.cfg, "torque_pulse_interval_max_s", 5.0))
        if max_s < min_s:
            max_s = min_s
        ctrl_dt = float(self.sim.cfg.dt) * float(getattr(self.cfg, "decimation", 1))
        now = self.episode_length_buf[env_ids].to(device=self.device, dtype=torch.float32) * ctrl_dt
        if initial:
            now = torch.zeros_like(now)
        if self._torque_pulse_generator_device == torch.device(self.device):
            jitter = torch.rand(env_ids.numel(), device=self.device, generator=self._torque_pulse_generator)
        else:
            jitter = torch.rand(env_ids.numel(), generator=self._torque_pulse_generator).to(self.device)
        self._torque_pulse_next_t[env_ids] = now + min_s + jitter * (max_s - min_s)
        self._torque_pulse_end_t[env_ids] = -1.0
        self._torque_pulse_vec_b[env_ids] = 0.0
        self._torque_pulse_active[env_ids] = False

    def _update_runtime_torque_pulse(self, torques: torch.Tensor) -> torch.Tensor:
        """Add intermittent body-frame torque pulses for disturbance validation."""
        level = float(getattr(self.cfg, "torque_pulse_level", 0.0))
        enabled = bool(getattr(self.cfg, "torque_pulse_enable", False)) and level > 0.0
        self._torque_pulse_active[:] = False
        if not enabled:
            self._torque_pulse_vec_b[:] = 0.0
            return torques

        ctrl_dt = float(self.sim.cfg.dt) * float(getattr(self.cfg, "decimation", 1))
        current_time = self.episode_length_buf.to(device=self.device, dtype=torch.float32) * ctrl_dt
        due = current_time >= self._torque_pulse_next_t
        if torch.any(due):
            ids = due.nonzero(as_tuple=False).flatten()
            axes = torch.as_tensor(getattr(self.cfg, "torque_pulse_axes", [1.0, 1.0, 1.0]), device=self.device, dtype=torch.float32).reshape(1, 3)
            if self._torque_pulse_generator_device == torch.device(self.device):
                direction = torch.randn((ids.numel(), 3), device=self.device, generator=self._torque_pulse_generator)
            else:
                direction = torch.randn((ids.numel(), 3), generator=self._torque_pulse_generator).to(self.device)
            direction = direction * axes
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
            self._torque_pulse_vec_b[ids] = direction * level
            self._torque_pulse_end_t[ids] = current_time[ids] + float(getattr(self.cfg, "torque_pulse_duration_s", 0.25))
            self._torque_pulse_next_t[ids] = float("inf")

        active = current_time < self._torque_pulse_end_t
        if torch.any(active):
            self._torque_pulse_active[active] = True
            torques = torques + self._torque_pulse_vec_b

        ended = (~active) & torch.isinf(self._torque_pulse_next_t)
        if torch.any(ended):
            self._schedule_next_torque_pulse(ended.nonzero(as_tuple=False).flatten())
        return torques

    def get_runtime_torque_pulse_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._torque_pulse_active.clone(), self._torque_pulse_vec_b.clone()

    def apply_pid_multipliers(self, zeta_updates: Mapping[str, float], env_ids: Sequence[int] | None = None) -> None:
        """Apply multiplicative updates to PID/S-surface gains."""
        if env_ids is None:
            target_envs = slice(None)
        else:
            target_envs = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        axis_lookup = {"roll": 0, "pitch": 1, "yaw": 2, "depth": 3}
        param_lookup = {"zeta1": 0, "zeta2": 1, "zeta3": 2}

        for key, multiplier in zeta_updates.items():
            if "_" not in key:
                continue
            axis_name, param_name = key.rsplit("_", 1)
            if axis_name not in axis_lookup or param_name not in param_lookup:
                continue
            axis_idx = axis_lookup[axis_name]
            param_idx = param_lookup[param_name]
            self.PID_args[target_envs, axis_idx, param_idx] *= float(multiplier)

    def apply_ctrl_mismatch(self, spec: Mapping[str, object] | None, env_ids: Sequence[int] | None = None) -> None:
        """M3: 安装非 pid_gain 失配靶点（plug-and-play，可拆装）。

        Args:
            spec: dict，至少含 ``mode`` ∈ {pid_gain, actuator_scale, s_surface_struct, allocation_skew}。
                  其余字段按 mode 解释：
                    actuator_scale : ``thrust_scale`` (float, 全推进器乘性) 或
                                     ``thruster_scales`` (list[num_thrusters])。
                    s_surface_struct: ``s_ratio_scale`` (float)、``add_scale`` (float)。
                    allocation_skew : ``alloc_scale`` (list[4], roll/pitch/yaw/depth) 或
                                      ``alloc_axis``+``alloc_value`` 单轴。
            env_ids: 仅作用于这些 env；None = 全部。
        本方法只写入持久 mismatch buffer，恒等默认不改变任何行为。pid_gain 模式为 no-op
        （仍走既有 --pid_multipliers 路径）。
        """
        if not spec:
            return
        mode = str(spec.get("mode", "pid_gain"))
        self.ctrl_mismatch_mode = mode
        if env_ids is None:
            target = slice(None)
        else:
            target = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if mode == "pid_gain":
            return  # 现状路径；增益缩放走 apply_pid_multipliers。

        if mode == "actuator_scale":
            if "thruster_scales" in spec:
                scales = torch.as_tensor(
                    spec["thruster_scales"], device=self.device, dtype=torch.float32
                ).reshape(1, self._num_thrusters)
                self._mismatch_actuator_scale[target] = scales
            else:
                s = float(spec.get("thrust_scale", 1.0))
                self._mismatch_actuator_scale[target] = s
        elif mode == "s_surface_struct":
            self._mismatch_s_ratio_scale = float(spec.get("s_ratio_scale", 1.0))
            self._mismatch_add_scale = float(spec.get("add_scale", 1.0))
        elif mode == "allocation_skew":
            if "alloc_scale" in spec:
                vec = torch.as_tensor(
                    spec["alloc_scale"], device=self.device, dtype=torch.float32
                ).reshape(1, 4)
                self._mismatch_alloc_scale[target] = vec
            else:
                axis_lookup = {"roll": 0, "pitch": 1, "yaw": 2, "depth": 3}
                axis = str(spec.get("alloc_axis", "yaw"))
                val = float(spec.get("alloc_value", 1.0))
                if axis in axis_lookup:
                    self._mismatch_alloc_scale[target, axis_lookup[axis]] = val
        else:
            print(f"[WARN] apply_ctrl_mismatch: unknown mode {mode!r}; ignored.")

    def apply_boundary_effect(self, spec: Mapping[str, object] | str | None) -> None:
        """M4: 配置近边界效应（plug-and-play，可拆装）。

        Args:
            spec: 字符串（直接当作 ``boundary_effect_mode`` 预设）或 dict：
                  {"mode": "free_surface", "z_surface": 1.5, "residual_buoyancy_frac": 0.02,
                   "r_cob": [0,0,0.02], ...}。dict 中除 ``mode`` 外的键按名覆盖
                  BoundaryEffectModels 的同名字段，再应用 mode 预设的开关。
        默认/未提供时保持 off，零行为变更。
        """
        if spec is None:
            return
        if isinstance(spec, str):
            self.boundary_models.apply_mode(spec)
            return
        if not isinstance(spec, Mapping):
            print(f"[WARN] apply_boundary_effect: unsupported spec {spec!r}; ignored.")
            return
        mode = str(spec.get("mode", "off"))
        for key, val in spec.items():
            if key == "mode":
                continue
            if hasattr(self.boundary_models, key):
                setattr(self.boundary_models, key, val)
            else:
                print(f"[WARN] apply_boundary_effect: unknown field {key!r}; ignored.")
        self.boundary_models.apply_mode(mode)

    def apply_embodiment_config(self, embodiment_type: str) -> None:
        """
        Apply a specific embodiment configuration to the environment.

        Args:
            embodiment_type: One of 'base', 'long_body', 'heavy_duty', 'asymmetric'
        """
        if embodiment_type not in self.cfg.embodiment_configs:
            raise ValueError(f"Unknown embodiment type: {embodiment_type}. "
                           f"Available types: {list(self.cfg.embodiment_configs.keys())}")

        config = self.cfg.embodiment_configs[embodiment_type]
        self._embodiment_type = embodiment_type

        # Apply mass
        if self.cfg.mass:
            self.masses = torch.full((self.num_envs, 1), config["mass"], device=self.device)
            # Also update the robot's mass in PhysX
            self._robot.root_physx_view.set_masses(self.masses.detach().cpu(), self._robot._ALL_INDICES.cpu())

        # Apply inertia tensors
        self.inertia_tensors[:, 0] = config["inertia_tensors"][0]
        self.inertia_tensors[:, 1] = config["inertia_tensors"][1]
        self.inertia_tensors[:, 2] = config["inertia_tensors"][2]

        # Apply COM to COB offset
        self.com_to_cob_offsets = torch.tensor(config["com_to_cob_offset"]).repeat(self.num_envs, 1).to(self.device)

        # Apply dynamics time constant
        self.cfg.dyn_time_constant = config["dyn_time_constant"]

        # === 推进器几何 + 分配路径 ===
        # config 声明 thrust_allocation -> config 驱动 B⁺ 路径（任意 N 推、支持非正交/欠驱动）；
        # 否则 -> 旧硬编码 8 推混合块路径（base/asym/long_body/heavy_*，零行为变更 A4）。
        alloc_cfg = config.get("thrust_allocation")
        if alloc_cfg is not None:
            layout = ThrusterLayout.from_specs(alloc_cfg["layout_specs"])
            self._num_thrusters = layout.num_thrusters
            self._use_config_alloc = True
            self.thruster_com_offsets = layout.positions.unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
            self.thruster_quats = layout.orientations.repeat(self.num_envs, 1).to(self.device)
            self._base_thruster_quats = self.thruster_quats.clone()
            self._alloc_B = build_wrench_matrix(layout).to(self.device)
            self._alloc_mode = alloc_cfg.get("mode", "pinv")
            self._alloc_weight = dof_weight_vector(alloc_cfg.get("controllable_dofs")).to(self.device)
        else:
            self._num_thrusters = 8
            self._use_config_alloc = False
            self._alloc_B = None
            self._alloc_weight = None
            # 旧几何重建路径（支持 long_body 的 thruster_com_offset_scale）。
            base_com_offsets, base_quats = get_thruster_com_and_orientations(self.device)
            if "thruster_com_offset_scale" in config:
                scale = config["thruster_com_offset_scale"]
                self.thruster_com_offsets = (base_com_offsets * scale).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
                # Note: thruster_quats remain unchanged
            else:
                self.thruster_com_offsets = base_com_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)

        # Reinitialize thruster dynamics with new time constant + thruster count
        self.thruster_dynamics = DynamicsFirstOrder(self.num_envs, self._num_thrusters, self.cfg.dyn_time_constant, self.device)

        # 推进器数量变化时，重建按推进器广播的运行期 buffer（默认 8 推时 N==8 恒等，零行为变更）。
        # 这两个 buffer 在 _compute_dynamics 里逐推进器乘 motorValues（shape N），必须与 N 对齐。
        if self.thruster_efficiency_factors.shape[1] != self._num_thrusters:
            self.thruster_efficiency_factors = torch.ones((self.num_envs, self._num_thrusters), device=self.device)
        if self._mismatch_actuator_scale.shape[1] != self._num_thrusters:
            self._mismatch_actuator_scale = torch.ones((self.num_envs, self._num_thrusters), device=self.device)

        # Apply drag multiplier for heavy-duty case
        self._drag_multiplier = config.get("drag_multiplier", 1.0)

        # config 化 volume（ρ_body 差异化，微正浮力）。
        if "volume" in config:
            self.volumes = torch.full((self.num_envs, 1), config["volume"], device=self.device)

        # config 化 action_lim（SO(3) 解封 roll/pitch）。
        if "action_lim_vec" in config:
            self.action_lim = torch.tensor(
                config["action_lim_vec"], dtype=torch.float, device=self.device
            ).reshape(1, 4)

        self._refresh_domain_randomization_defaults()

        print(f"Applied embodiment configuration: {embodiment_type}")
        print(f"  Mass: {config['mass']}")
        print(f"  Inertia tensors: {config['inertia_tensors']}")
        print(f"  COM to COB offset: {config['com_to_cob_offset']}")
        print(f"  Dynamics time constant: {config['dyn_time_constant']}")
        print(f"  Drag multiplier: {self._drag_multiplier}")
        print(f"  Thrusters: {self._num_thrusters}  (config_alloc={self._use_config_alloc}, mode={self._alloc_mode})")

    def _refresh_domain_randomization_defaults(self) -> None:
        """@dox _refresh_domain_randomization_defaults
        @brief Snapshot embodiment + PID gains as the per-episode DR baseline.
        @details Called from __init__ and apply_runtime_domain_shift. This is
                 also the **only** correct place to capture self._zeta_nominal
                 (= PID_args[:,:,0].clone()) so that meta-control's
                 ratio = zeta_runtime / zeta_nominal stays anchored on the
                 currently active embodiment/integrator state.
        """
        self._base_masses = self.masses.clone()
        self._base_inertia_tensors = self.inertia_tensors.clone()
        self._base_com_to_cob_offsets = self.com_to_cob_offsets.clone()
        self._base_volumes = self.volumes.clone()
        self._base_pid_args = self.PID_args.clone()
        self._base_drag_multiplier = torch.full((self.num_envs,), float(self._drag_multiplier), device=self.device)
        self._base_thruster_com_offsets = self.thruster_com_offsets.clone()
        self._base_dyn_time_constants = self.thruster_dynamics.tau.clone()
        self._drag_multiplier_per_env = self._base_drag_multiplier.clone()
        # ζ_nominal 快照：以 PID_args[:,:,0] 为 zeta1 标称值，配合 ParametricGainTuner 使用。
        if getattr(self, "_tune_gains_enabled", False):
            self._zeta_nominal = self.PID_args[:, :, 0].clone()
            self._zeta_runtime = self._zeta_nominal.clone()
            if self._gain_tuner is not None:
                self._gain_tuner.reset()

    def _sample_range(self, value_range, shape) -> torch.Tensor:
        range_tensor = torch.as_tensor(value_range, device=self.device, dtype=torch.float32)
        if range_tensor.ndim == 1 and range_tensor.numel() == 2:
            lower = torch.full(shape, float(range_tensor[0].item()), device=self.device)
            upper = torch.full(shape, float(range_tensor[1].item()), device=self.device)
        elif range_tensor.shape == tuple(shape):
            return range_tensor.to(device=self.device, dtype=torch.float32)
        elif len(shape) >= 1 and range_tensor.shape == tuple(shape[1:]):
            return range_tensor.unsqueeze(0).expand(shape).to(device=self.device, dtype=torch.float32)
        elif range_tensor.shape[-1] != 2:
            return range_tensor.expand(shape).to(device=self.device, dtype=torch.float32)
        else:
            lower = range_tensor[..., 0].expand(shape).to(device=self.device, dtype=torch.float32)
            upper = range_tensor[..., 1].expand(shape).to(device=self.device, dtype=torch.float32)
        return math_utils.sample_uniform(lower, upper, shape, self.device)

    def _set_masses(self, env_ids: torch.Tensor, masses: torch.Tensor) -> None:
        self.masses[env_ids] = masses
        env_ids_cpu = env_ids.detach().cpu()
        all_masses = self._robot.root_physx_view.get_masses()
        all_masses[env_ids_cpu] = self.masses[env_ids].detach().cpu()
        self._robot.root_physx_view.set_masses(all_masses, env_ids_cpu)

    def get_embodiment_type(self) -> str:
        """Get the current embodiment type."""
        return self._embodiment_type

    def get_current_fluid_velocity(self) -> torch.Tensor:
        """@dox get_current_fluid_velocity
        @brief Compute the per-env water-current velocity for hydrodynamics.
        @return Tensor (N, 3) [m/s] in world frame.
        @details Three modes via cfg.disturbance_cfg.mode:
                   - "constant" : returns base_vel.
                   - "sine"     : base_vel + sum_i amplitude_i * sin(2 pi f_i t).
                   - "jonswap"  : base_vel + JonswapWaveDisturbanceManager.get_wave_velocity(t).
                 The wave manager is cached by (hs, fp, depth, direction, seed) signature;
                 changing those via apply_runtime_domain_shift invalidates the cache.
        """
        disturbance_cfg = self.cfg.disturbance_cfg
        mode = getattr(disturbance_cfg, "mode", "none")

        def _vectorize(value):
            vector = torch.as_tensor(value, device=self.device, dtype=torch.float32).flatten()
            if vector.numel() == 1:
                vector = vector.repeat(3)
            return vector[:3].reshape(1, 3).repeat(self.num_envs, 1)

        base_vel = _vectorize(getattr(disturbance_cfg, "base_vel", [0.0, 0.0, 0.0]))

        if mode == "none":
            return torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        if mode == "constant":
            return base_vel
        if mode == "sine":
            amplitude = _vectorize(getattr(disturbance_cfg, "amplitude", [0.0, 0.0, 0.0]))
            frequency = _vectorize(getattr(disturbance_cfg, "frequency", [0.0, 0.0, 0.0]))
            current_time = self.episode_length_buf.to(device=self.device, dtype=torch.float32).unsqueeze(-1) * self.sim.cfg.dt
            return base_vel + amplitude * torch.sin(2.0 * math.pi * frequency * current_time)
        if mode == "jonswap":
            manager = self._get_wave_manager()
            current_time = self.episode_length_buf.to(device=self.device, dtype=torch.float32) * self.sim.cfg.dt
            return base_vel + manager.get_wave_velocity(self._robot.data.root_pos_w, current_time)

        raise ValueError(f"Unknown disturbance mode: {mode}")

    def _get_wave_manager(self) -> JonswapWaveDisturbanceManager:
        disturbance_cfg = self.cfg.disturbance_cfg
        signature = (
            float(getattr(disturbance_cfg, "jonswap_hs", 0.5)),
            float(getattr(disturbance_cfg, "jonswap_fp", 0.1)),
            float(getattr(disturbance_cfg, "jonswap_gamma", 3.3)),
            float(getattr(disturbance_cfg, "jonswap_depth", 30.0)),
            float(getattr(disturbance_cfg, "jonswap_direction", 0.0)),
            int(getattr(disturbance_cfg, "jonswap_seed", 7)),
        )
        if self._wave_manager is None or self._wave_manager_signature != signature:
            self._wave_manager = JonswapWaveDisturbanceManager(
                hs=signature[0],
                fp=signature[1],
                gamma=signature[2],
                depth=signature[3],
                direction=signature[4],
                seed=signature[5],
                device=self.device,
            )
            self._wave_manager_signature = signature
        return self._wave_manager

    def apply_runtime_domain_shift(
        self,
        *,
        volume_delta: float | None = None,
        volume_scale: float | None = None,
        mode: str | None = None,
        base_vel: Sequence[float] | torch.Tensor | None = None,
        amplitude: Sequence[float] | torch.Tensor | None = None,
        frequency: Sequence[float] | torch.Tensor | None = None,
        jonswap_hs: float | None = None,
        jonswap_fp: float | None = None,
        jonswap_gamma: float | None = None,
        jonswap_depth: float | None = None,
        jonswap_direction: float | None = None,
        jonswap_seed: int | None = None,
        noise_std: float | None = None,
        noise_corr: float | None = None,
        ang_vel_extra_std: float | None = None,
        d_filter_tau: float | None = None,
        d_filter_phase_lead_steps: float | None = None,
        d_filter_phase_lead_clip: float | None = None,
        water_density_scale: float | None = None,
        water_rho: float | None = None,
        torque_pulse_level: float | None = None,
        torque_pulse_interval_min_s: float | None = None,
        torque_pulse_interval_max_s: float | None = None,
        torque_pulse_duration_s: float | None = None,
        thruster_angle_shift_rad: float | None = None,
        thruster_angle_shift_thrusters: Sequence[int] | None = None,
        thruster_angle_shift_axis: str | None = None,
    ) -> None:
        """Apply a runtime domain shift without reaching into workflow-private state."""
        ## STDW INTEGRATION MARKER ##
        if volume_delta is not None:
            self.volumes.add_(volume_delta)
        if volume_scale is not None:
            self.volumes.mul_(volume_scale)

        disturbance_cfg = self.cfg.disturbance_cfg
        if mode is not None:
            disturbance_cfg.mode = mode
        if base_vel is not None:
            disturbance_cfg.base_vel = list(torch.as_tensor(base_vel, device=self.device, dtype=torch.float32).flatten()[:3].tolist())
        if amplitude is not None:
            disturbance_cfg.amplitude = list(torch.as_tensor(amplitude, device=self.device, dtype=torch.float32).flatten()[:3].tolist())
        if frequency is not None:
            disturbance_cfg.frequency = list(torch.as_tensor(frequency, device=self.device, dtype=torch.float32).flatten()[:3].tolist())
        if jonswap_hs is not None:
            disturbance_cfg.jonswap_hs = float(jonswap_hs)
        if jonswap_fp is not None:
            disturbance_cfg.jonswap_fp = float(jonswap_fp)
        if jonswap_gamma is not None:
            disturbance_cfg.jonswap_gamma = float(jonswap_gamma)
        if jonswap_depth is not None:
            disturbance_cfg.jonswap_depth = float(jonswap_depth)
        if jonswap_direction is not None:
            disturbance_cfg.jonswap_direction = float(jonswap_direction)
        if jonswap_seed is not None:
            disturbance_cfg.jonswap_seed = int(jonswap_seed)
        if any(v is not None for v in (jonswap_hs, jonswap_fp, jonswap_gamma, jonswap_depth, jonswap_direction, jonswap_seed)):
            self._wave_manager = None
            self._wave_manager_signature = None

        if noise_std is not None:
            self.cfg.noise_cfg.enable_noise = True
            self.cfg.noise_cfg.std_dev = noise_std
        if noise_corr is not None:
            self.cfg.noise_cfg.enable_noise = True
            self.cfg.noise_cfg.correlation_coeff = noise_corr
        if ang_vel_extra_std is not None:
            # A4：IMU 角速度专项噪声等级（rad/s）。>0 时同时确保噪声开启。
            self.cfg.noise_cfg.ang_vel_extra_std = float(ang_vel_extra_std)
            if float(ang_vel_extra_std) > 0.0:
                self.cfg.noise_cfg.enable_noise = True
        if d_filter_tau is not None:
            # A1：S 面 D 项一阶 EMA 低通时间常数（s）。0.0 即直通（旧行为）。
            self.cfg.d_filter_tau = float(d_filter_tau)
        if d_filter_phase_lead_steps is not None:
            self.cfg.d_filter_phase_lead_steps = float(d_filter_phase_lead_steps)
        if d_filter_phase_lead_clip is not None:
            self.cfg.d_filter_phase_lead_clip = float(d_filter_phase_lead_clip)
        if water_rho is not None:
            self.cfg.water_rho = float(water_rho)
        elif water_density_scale is not None:
            self.cfg.water_rho = float(self._base_water_rho) * float(water_density_scale)
        if torque_pulse_level is not None:
            self.cfg.torque_pulse_level = float(torque_pulse_level)
            self.cfg.torque_pulse_enable = float(torque_pulse_level) > 0.0
        if torque_pulse_interval_min_s is not None:
            self.cfg.torque_pulse_interval_min_s = float(torque_pulse_interval_min_s)
        if torque_pulse_interval_max_s is not None:
            self.cfg.torque_pulse_interval_max_s = float(torque_pulse_interval_max_s)
        if torque_pulse_duration_s is not None:
            self.cfg.torque_pulse_duration_s = float(torque_pulse_duration_s)
        if thruster_angle_shift_rad is not None:
            self.set_thruster_angle_shift(
                enabled=abs(float(thruster_angle_shift_rad)) > 0.0,
                thrusters=list(thruster_angle_shift_thrusters) if thruster_angle_shift_thrusters is not None else None,
                angle_rad=float(thruster_angle_shift_rad),
                axis=str(thruster_angle_shift_axis or getattr(self.cfg, "thruster_angle_shift_axis", "yaw")),
            )

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.starting_depth))
        self._robot = RigidObject(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))

        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """@dox _pre_physics_step
        @brief Split incoming actions into 4D ctrl and (optionally) 4D a_gain.
        @param actions Shape (N, 4) for baseline, (N, 8) for meta-control.
        @details When tune_gains=True and shape>=8, action[:, 4:8] is stored
                 in self._a_gain_buf for later consumption by _apply_action.
                 Action is clipped to [-1, 1] before split.
        """
        if self._debug: print("original actions vec: ", actions)
        if self._debug: print("concatenated actions shape: ", self._actions)

        # 阻尼项缓冲：在覆盖前把上一/上上步动作存好（reward 消费）。
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = self._actions

        self._actions[:] = actions
        self._actions[:] = torch.clip(self._actions, -1, 1).to(self.device)
        if self._tune_gains_enabled and self._actions.shape[1] >= 8:
            self._a_gain_buf[:] = self._actions[:, 4:8]

        # 连续参考模式：每个控制步按 episode 内时间平滑推进参考目标。
        if getattr(self.cfg, "reference_mode", "step") in {"sine_sweep", "flip360_sine", "mixed_sine_flip360"}:
            self._update_reference()

    def _apply_action(self) -> None:
        """@dox _apply_action
        @brief Drive thrusters from low-level ctrl actions; optionally route
               ParametricGainTuner output to PID gains.
        @details In 8D meta-control mode, this calls _gain_tuner.step(...) which
                 runs the 4-stage open-loop chain:
                   LPF -> deadzone -> Bounded Safeguard zeta_runtime = zeta_nom*(1 + beta*a_gain)
                   -> PE injection.
                 The ratio (zeta_runtime / zeta_nominal) is multiplied into PID_args
                 columns selected by cfg.gain_update_targets (default ["zeta1"], i.e. P).
                 PID_args is restored to base after _compute_dynamics so the change
                 only takes effect this physics tick.
        """
        # 8 维 meta-control：前 4 维 a_ctrl 进 _compute_dynamics；后 4 维 a_gain
        # 经 ParametricGainTuner 调制 ζ_runtime，写到 PID_args[:,:,0/1]（按 cfg.gain_update_targets）。
        if self._tune_gains_enabled and self._gain_tuner is not None:
            ang_vel = self._robot.data.root_ang_vel_b
            compound_error = torch.norm(ang_vel, dim=-1)
            self._sim_time_s += float(self.sim.cfg.dt) * float(self.cfg.decimation)
            zeta_runtime, debug = self._gain_tuner.step(
                a_gain_raw=self._a_gain_buf,
                zeta_nominal=self._zeta_nominal,
                body_ang_vel=ang_vel,
                compound_error=compound_error,
                sim_time_s=self._sim_time_s,
            )
            self._zeta_runtime = zeta_runtime
            self._last_pe_active = debug["pe_active"]
            self._last_deadzone_active = debug["deadzone_active"]
            # 路由到 PID_args 的列：zeta1 -> col 0 (P), zeta2 -> col 1 (D), zeta3 -> col 2 (I)
            col_lookup = {"zeta1": 0, "zeta2": 1, "zeta3": 2}
            targets = list(getattr(self.cfg, "gain_update_targets", ["zeta1"]))
            ratio = (zeta_runtime / torch.clamp(self._zeta_nominal, min=1e-8))
            for t in targets:
                col = col_lookup.get(t)
                if col is None:
                    continue
                self.PID_args[:, :, col] = self._base_pid_args[:, :, col] * ratio
            ctrl_actions = self._actions[:, :4]
        else:
            ctrl_actions = self._actions[:, :4] if self._actions.shape[1] >= 4 else self._actions

        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(ctrl_actions)
        self._robot.set_external_force_and_torque(self._thrust, self._moment)

        if self._tune_gains_enabled:
            # 还原 PID_args，避免穿透到下一步 / DR 重抽样
            self.PID_args[:] = self._base_pid_args

    def _get_observations(self) -> dict:
        #desired_pos_b = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._goal - self._robot.data.root_pos_w)
        offset_from_origin_b = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._default_env_origins - self._robot.data.root_pos_w)

        # Uniquefy and normalize all quaternions
        # goal = self._goal
        # root_quat_w = self._robot.data.root_quat_w
        # goal = math_utils.normalize(math_utils.quat_unique(self._goal))
        # root_quat_w = math_utils.normalize(math_utils.quat_unique(self._robot.data.root_quat_w))

        obs_components = [
            self._goal, # 4
            # offset_from_origin_b[:,2].unsqueeze(1),
            self._robot.data.root_pos_w[:,2].unsqueeze(1), # 1
            self._robot.data.root_quat_w,# 4
            # self._robot.data.root_lin_vel_b,
        ]
        if getattr(self.cfg, "obs_include_ang_vel", False):
            # A3：把 body 角速度喂回观测，便于策略学习高频抑振。
            obs_components.append(self._robot.data.root_ang_vel_b)  # 3
        obs = torch.cat(obs_components, dim=-1)

        if self.cfg.noise_cfg.enable_noise:
            new_random_noise = torch.randn_like(self.obs_noise_buffer) * self.cfg.noise_cfg.std_dev
            new_random_noise[:, : self.cfg.goal_dims] = 0.0
            alpha = self.cfg.noise_cfg.correlation_coeff
            self.obs_noise_buffer.mul_(alpha).add_((1.0 - alpha) * new_random_noise)
            self.obs_noise_buffer[:, : self.cfg.goal_dims] = 0.0

            obs = obs + self.obs_noise_buffer
            # 仅对 quaternion (goal_dims+1 起的 4 维) 做单位化；
            # 后续若拼接了 ang_vel 不应被卷入 normalize。
            quat_start = self.cfg.goal_dims + 1
            obs[:, quat_start:quat_start + 4] = normalize(obs[:, quat_start:quat_start + 4])

        # A4：IMU 级角速度专项白噪声，叠加在 obs 末 3 维 root_ang_vel_b 上。
        # 与上面的相关高斯噪声相互独立，用于噪声/滤波研究（默认 0.0 不生效）。
        ang_vel_extra = float(getattr(self.cfg.noise_cfg, "ang_vel_extra_std", 0.0))
        if getattr(self.cfg, "obs_include_ang_vel", False) and ang_vel_extra > 0.0:
            obs[:, -3:] = obs[:, -3:] + torch.randn_like(obs[:, -3:]) * ang_vel_extra

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        offsets_from_origin = quat_apply(quat_conjugate(self._robot.data.root_quat_w), self._default_env_origins - self._robot.data.root_pos_w)

        # 动作平滑 (rate) 与高频抖动 (jerk) 阻尼信号。
        action_rate = self._actions - self._prev_action
        action_jerk = self._actions - 2.0 * self._prev_action + self._prev_prev_action

        total_reward = _compute_rewards(
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_ang,
            self.cfg.rew_scale_lin_vel,
            self.cfg.rew_scale_ang_vel,
            self.cfg.rew_scale_actions,
            float(getattr(self.cfg, "rew_scale_action_rate", 0.0)),
            float(getattr(self.cfg, "rew_scale_action_jerk", 0.0)),
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self.reset_terminated,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._goal,
            offsets_from_origin,
            self._completed_envs,
            self._actions,
            action_rate,
            action_jerk,
        )

        # 获得数值
        ang_mse = math_utils.quat_error_magnitude(self._goal[:,:], self._robot.data.root_quat_w[:,:])
        self.log_MSE += torch.pow(ang_mse,2)

        # F4：难区间（接近倒置）容差整形。在 goal 倾角进入 band 时放宽姿态项惩罚斜率，
        # 避免始终在线的 rew_ang 对物理不可达/不稳定的倒置目标持续扣分而污染易控区。
        # 等效做法：对该区 env 把 rew_ang 的高斯指数除以 (1 + relax*w)，w∈[0,1] 平滑过渡。
        flip_relax = float(getattr(self.cfg, "flip_tol_relax", 0.0))
        if flip_relax > 0.0:
            # goal 倾角 = goal 体 z 轴与世界 z 轴夹角；用四元数把世界 +z 旋到 goal 体系，
            # 取其 z 分量 = cos(tilt)。goal 存 (w,x,y,z) 顺序。
            gq = self._goal[:, 0:4]
            w, x, y, z = gq[:, 0], gq[:, 1], gq[:, 2], gq[:, 3]
            # body_z 在世界系的 z 分量（旋转矩阵 R[2,2]）= 1 - 2(x^2 + y^2)
            cos_tilt = 1.0 - 2.0 * (x * x + y * y)
            tilt = torch.acos(torch.clamp(cos_tilt, -1.0, 1.0))
            lo = float(getattr(self.cfg, "flip_tol_band_lo", 2.094395))
            hi = float(getattr(self.cfg, "flip_tol_band_hi", math.pi))
            u = torch.clamp((tilt - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            smooth = u * u * (3.0 - 2.0 * u)  # smoothstep
            relax_factor = 1.0 + flip_relax * smooth  # >=1，band 内放宽
            base_ang = self.cfg.rew_scale_ang * torch.exp(-1.0 * ang_mse)
            shaped_ang = self.cfg.rew_scale_ang * torch.exp(-1.0 * ang_mse / relax_factor)
            total_reward = total_reward + (shaped_ang - base_ang)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs)

        self._step_count = self._step_count + 1

        if self.cfg.episode_length_before_reset:
            if self._step_count == self.cfg.episode_length_before_reset:
                time_out = torch.ones(self.num_envs)

        if self.cfg.use_boundaries:
            out_of_bounds = (
                (torch.abs(self._robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]) > self.cfg.max_auv_x) | 
                (torch.abs(self._robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]) > self.cfg.max_auv_y) | 
                (torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.starting_depth) > self.cfg.max_auv_z)
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if getattr(self.cfg, "surface_guard_enable", False):
            breach = self._robot.data.root_pos_w[:, 2] > (self.cfg.z_surface_guard - self.cfg.surface_margin)
            out_of_bounds = out_of_bounds.to(device=self.device, dtype=torch.bool) | breach

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """@dox _reset_idx
        @brief Reset selected envs and zero meta-control state.
        @details At the tail end this resets _gain_tuner internal LPF/PE state,
                 _sim_time_s, and _a_gain_buf for env_ids so a new episode does
                 not inherit stale gain dynamics.
        """
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.obs_noise_buffer[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._prev_prev_action[env_ids] = 0.0
        # A1：清零 D 项 EMA 状态，避免新 episode 继承上一段尾部姿态速度。
        if hasattr(self, "_actions_d_filt"):
            self._actions_d_filt[env_ids] = 0.0
        if hasattr(self, "_actions_d_filt_prev"):
            self._actions_d_filt_prev[env_ids] = 0.0
        if hasattr(self, "_torque_pulse_next_t"):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long) if not isinstance(env_ids, torch.Tensor) else env_ids
            self._schedule_next_torque_pulse(ids, initial=True)

        self._default_root_state[env_ids, :] = self._robot.data.default_root_state[env_ids]
        self._default_root_state[env_ids, :3] += self.scene.env_origins[env_ids]

        self._default_env_origins[env_ids, :] = self._default_root_state[env_ids, :3]

        if not self.cfg.eval_mode:
            # Randomize initial position relative to the origin
            self._default_root_state[env_ids, :3] += self._sample_from_sphere(len(env_ids), self.cfg.goal_spawn_radius)

            # Randomize initial orientation relative to the origin
            # self._default_root_state[env_ids, 3:7] = math_utils.random_orientation(len(env_ids), device=self.device)
            
            # Randomize initial linear and rotational velocities
            # self._default_root_state[env_ids, 7:13] = math_utils.sample_uniform(-self.cfg.init_vel_max, self.cfg.init_vel_max, (len(env_ids), 6), device=self.device)

        self._step_count = 0
        
        # Apply domain randomization
        self._reset_domain(env_ids)

        # Reset goals
        self._reset_goal(env_ids)

        if not self.cfg.eval_mode:
            # Apply guidance (set to goal position and orientation)
            envs_to_guide = math_utils.sample_uniform(0, 1, len(env_ids), self.device) < self.cfg.init_guidance_rate
            env_ids_to_guide = env_ids[envs_to_guide]
            self._default_root_state[env_ids_to_guide, :3] = self._default_env_origins[env_ids_to_guide, :3]
            self._default_root_state[env_ids_to_guide, 3:7] = self._goal[env_ids_to_guide, 0:4]

        self._robot.write_root_pose_to_sim(self._default_root_state[env_ids, :7], env_ids)
        self._robot.write_root_velocity_to_sim(self._default_root_state[env_ids, 7:], env_ids)

        # logging

        extras = dict()
        extras['Episode Reward / log MSE'] = torch.mean(self.log_MSE[env_ids]) / self.max_episode_length_s
        self.log_MSE[env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # ParametricGainTuner per-env state reset（LPF 状态 + sim_time 与 PE phase）
        if self._tune_gains_enabled and self._gain_tuner is not None:
            self._gain_tuner.reset(env_ids)
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long) if not isinstance(env_ids, torch.Tensor) else env_ids
            self._sim_time_s[ids] = 0.0
            self._a_gain_buf[ids] = 0.0


    # OVERRIDE THIS FUNC TO CHANGE GOAL
    def _reset_goal(self, env_ids: Sequence[int]):
        if getattr(self.cfg, "reference_mode", "step") in {"sine_sweep", "flip360_sine", "mixed_sine_flip360"}:
            # 平滑正弦参考：每个 env 逐轴抽随机相位，episode 内目标由 _update_reference 连续生成。
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long) \
                if not isinstance(env_ids, torch.Tensor) else env_ids
            self._ref_phase[ids] = math_utils.sample_uniform(
                0.0, 2.0 * math.pi, (ids.numel(), 3), self.device
            )
            if getattr(self.cfg, "reference_mode", "step") == "mixed_sine_flip360":
                flip_prob = float(getattr(self.cfg, "ref_mix_flip_prob", 0.5))
                self._ref_is_flip[ids] = torch.rand(ids.numel(), device=self.device) < flip_prob
            else:
                self._ref_is_flip[ids] = getattr(self.cfg, "reference_mode", "step") == "flip360_sine"
            self._update_reference(ids)
            # 新 episode 首步令 ω_d=0：参考"上一步"对齐到刚生成的新目标。
            self._goal_prev[ids] = self._goal[ids, 0:4].clone()
            return

        # Get random orientation (step 模式：每个 episode 一个随机姿态硬阶跃)
        self._goal[env_ids, 0:4] = math_utils.random_orientation(len(env_ids), device=self.device)
        # step 模式 goal 恒定，令 _goal_prev = goal ⇒ ω_d = 0（SO(3) 模式不引入虚假参考角速度）。
        self._goal_prev[env_ids] = self._goal[env_ids, 0:4].clone()

        # Get random yaw orientation with 0 pitch and roll
        # self._goal[env_ids,0:4] = math_utils.random_yaw_orientation(len(env_ids), device=self.device)

        # Get fix RPY
        # rs = torch.zeros(len(env_ids), device=self.device) + 0.0
        # ps = torch.zeros(len(env_ids), device=self.device) + 0.0
        # ys = torch.zeros(len(env_ids), device=self.device) + 0.0
        # self._goal[env_ids,0:4] = math_utils.quat_from_euler_xyz(rs, ps, ys)

    def _update_reference(self, env_ids: Sequence[int] | None = None):
        """按 episode 内时间为每个 env 生成连续 roll/pitch/yaw 目标四元数。"""
        reference_mode = getattr(self.cfg, "reference_mode", "step")
        if reference_mode not in {"sine_sweep", "flip360_sine", "mixed_sine_flip360"}:
            return
        if env_ids is None:
            ids = self._robot._ALL_INDICES
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long) \
                if not isinstance(env_ids, torch.Tensor) else env_ids

        # SO(3) 控制：在覆盖 _goal 之前快照上一步参考，供有限差分求 ω_d。
        self._goal_prev[ids] = self._goal[ids, 0:4].clone()
        hold_mask = None
        if getattr(self.cfg, "submerge_phase_enable", False):
            hold_steps = int(getattr(self.cfg, "submerge_hold_steps", 0))
            if hold_steps > 0:
                hold_mask = self.episode_length_buf[ids] < hold_steps

        if reference_mode == "flip360_sine":
            amp_value = getattr(self.cfg, "ref_sine_amp", [math.pi, math.pi, 0.0])
            freq_value = getattr(self.cfg, "ref_sine_freq", [0.05, 0.05, 0.0])
        elif reference_mode == "mixed_sine_flip360":
            flip_amp = torch.as_tensor(
                getattr(self.cfg, "ref_sine_amp", [math.pi, math.pi, 0.0]),
                device=self.device,
                dtype=torch.float32,
            ).reshape(1, 3)
            flip_freq = torch.as_tensor(
                getattr(self.cfg, "ref_sine_freq", [0.05, 0.05, 0.0]),
                device=self.device,
                dtype=torch.float32,
            ).reshape(1, 3)
            sine_amp = torch.as_tensor(
                getattr(self.cfg, "ref_mix_sine_amp", self.cfg.ref_sine_amp),
                device=self.device,
                dtype=torch.float32,
            ).reshape(1, 3)
            sine_freq = torch.as_tensor(
                getattr(self.cfg, "ref_mix_sine_freq", self.cfg.ref_sine_freq),
                device=self.device,
                dtype=torch.float32,
            ).reshape(1, 3)
            is_flip = self._ref_is_flip[ids].reshape(-1, 1)
            amp = torch.where(is_flip, flip_amp, sine_amp)
            freq = torch.where(is_flip, flip_freq, sine_freq)
            t = (self.episode_length_buf[ids].to(torch.float32) * float(self.sim.cfg.dt)).unsqueeze(-1)  # (M,1)
            rpy = amp * torch.sin(2.0 * math.pi * freq * t + self._ref_phase[ids])  # (M,3)
            self._goal[ids, 0:4] = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            if hold_mask is not None and torch.any(hold_mask):
                sub_ids = ids[hold_mask]
                self._goal[sub_ids, 0:4] = 0.0
                self._goal[sub_ids, 0] = 1.0
            return
        else:
            amp_value = self.cfg.ref_sine_amp
            freq_value = self.cfg.ref_sine_freq
        amp = torch.as_tensor(amp_value, device=self.device, dtype=torch.float32).reshape(1, 3)
        freq = torch.as_tensor(freq_value, device=self.device, dtype=torch.float32).reshape(1, 3)
        t = (self.episode_length_buf[ids].to(torch.float32) * float(self.sim.cfg.dt)).unsqueeze(-1)  # (M,1)
        rpy = amp * torch.sin(2.0 * math.pi * freq * t + self._ref_phase[ids])  # (M,3)
        self._goal[ids, 0:4] = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        if hold_mask is not None and torch.any(hold_mask):
            sub_ids = ids[hold_mask]
            self._goal[sub_ids, 0:4] = 0.0
            self._goal[sub_ids, 0] = 1.0

    def _reset_domain(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        self.masses[env_ids] = self._base_masses[env_ids]
        self.inertia_tensors[env_ids] = self._base_inertia_tensors[env_ids]
        self.inertia_tensors_mean[env_ids] = self._base_inertia_tensors[env_ids].mean(dim=1, keepdim=True)
        self.com_to_cob_offsets[env_ids] = self._base_com_to_cob_offsets[env_ids]
        self.volumes[env_ids] = self._base_volumes[env_ids]
        self.PID_args[env_ids] = self._base_pid_args[env_ids]
        self.thruster_com_offsets[env_ids] = self._base_thruster_com_offsets[env_ids]
        if hasattr(self, "_base_thruster_quats"):
            self.thruster_quats = self._base_thruster_quats.clone()
            self.thruster_angle_shift_enabled = False
            self.thruster_angle_shift_rad = 0.0
        self.thruster_dynamics.set_time_constants(env_ids, self._base_dyn_time_constants[env_ids])
        self.thruster_dynamics.reset(env_ids)
        self._set_masses(env_ids, self._base_masses[env_ids])
        base_drag = self._base_drag_multiplier[env_ids]
        self._drag_multiplier_per_env[env_ids] = base_drag

        # Randomize COM to COB offset
        if self.cfg.domain_randomization.use_custom_randomization:
            xyz_range = torch.as_tensor(
                getattr(self.cfg.domain_randomization, "com_to_cob_offset_xyz_range", [0.0, 0.0, 0.0]),
                device=self.device,
                dtype=torch.float32,
            ).reshape(1, 3)
            if torch.any(xyz_range > 0):
                offset_delta = math_utils.sample_uniform(
                    -xyz_range.expand(env_ids.numel(), 3),
                    xyz_range.expand(env_ids.numel(), 3),
                    (env_ids.numel(), 3),
                    self.device,
                )
            else:
                offset_delta = self._sample_from_sphere(env_ids.numel(), self.cfg.domain_randomization.com_to_cob_offset_radius)
            self.com_to_cob_offsets[env_ids] = self._base_com_to_cob_offsets[env_ids] + offset_delta

        # Randomize volume
        if self.cfg.domain_randomization.use_custom_randomization:
            vol_lower, vol_upper = self.cfg.domain_randomization.volume_range
            self.volumes[env_ids] = math_utils.sample_uniform(vol_lower, vol_upper, self.volumes[env_ids].shape, self.device)

            mass_scale_range = getattr(self.cfg.domain_randomization, "mass_scale_range", [1.0, 1.0])
            mass_scale = self._sample_range(mass_scale_range, self.masses[env_ids].shape)
            sampled_masses = self._base_masses[env_ids] * mass_scale
            mass_range = getattr(self.cfg.domain_randomization, "mass_range", None)
            if mass_range is not None and not np.allclose(np.asarray(mass_scale_range, dtype=float), [1.0, 1.0]):
                self._set_masses(env_ids, sampled_masses)
            elif mass_range is not None:
                mass_lower, mass_upper = mass_range
                sampled_masses = math_utils.sample_uniform(mass_lower, mass_upper, self.masses[env_ids].shape, self.device)
                self._set_masses(env_ids, sampled_masses)

            inertia_scale = self._sample_range(
                getattr(self.cfg.domain_randomization, "inertia_scale_range", [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
                self.inertia_tensors[env_ids].shape,
            )
            self.inertia_tensors[env_ids] = self._base_inertia_tensors[env_ids] * inertia_scale
            self.inertia_tensors_mean[env_ids] = self.inertia_tensors[env_ids].mean(dim=1, keepdim=True)

            drag_scale = self._sample_range(
                getattr(self.cfg.domain_randomization, "drag_multiplier_scale_range", [1.0, 1.0]),
                base_drag.shape,
            )
            sampled_drag = base_drag * drag_scale.reshape(-1)
            self._drag_multiplier = float(sampled_drag.mean().item()) if env_ids.numel() == self.num_envs else self._drag_multiplier
            self._drag_multiplier_per_env[env_ids] = sampled_drag.reshape(-1)

            tau_scale = self._sample_range(
                getattr(self.cfg.domain_randomization, "dyn_time_constant_scale_range", [1.0, 1.0]),
                self._base_dyn_time_constants[env_ids].shape,
            )
            self.thruster_dynamics.set_time_constants(env_ids, self._base_dyn_time_constants[env_ids] * tau_scale.reshape(-1))

            thruster_scale = self._sample_range(
                getattr(self.cfg.domain_randomization, "thruster_com_offset_scale_range", [1.0, 1.0]),
                (env_ids.numel(), 1, 1),
            )
            self.thruster_com_offsets[env_ids] = self._base_thruster_com_offsets[env_ids] * thruster_scale

            pid_scale = self._sample_range(
                getattr(self.cfg.domain_randomization, "PID_scale_range", [[1.0, 1.0, 1.0]] * 4),
                self.PID_args[env_ids].shape,
            )
            self.PID_args[env_ids] = self._base_pid_args[env_ids] * pid_scale
            PID_adjust_range = torch.Tensor(self.cfg.domain_randomization.PID_adjust_range).to(self.device)
            self.PID_args[env_ids] += math_utils.sample_uniform(
                -PID_adjust_range.expand(env_ids.numel(), -1, -1),
                PID_adjust_range.expand(env_ids.numel(), -1, -1),
                self.PID_args[env_ids].shape,
                self.device,
            )

    def _sample_from_circle(self, num_env_ids, r):
        sampled_radius = r * torch.sqrt(torch.rand((num_env_ids), device=self.device))
        sampled_theta = torch.rand((num_env_ids), device=self.device) * 2 * 3.14159
        sampled_x = sampled_radius * torch.cos(sampled_theta)
        sampled_y = sampled_radius * torch.sin(sampled_theta)
        return (sampled_x, sampled_y)

    def _sample_from_sphere(self, num_env_ids, r):
        coords = torch.randn((num_env_ids, 3), device=self.device)
        norms = torch.norm(coords, dim=1).unsqueeze(1)
        coords /= norms

        radii = r * torch.pow(torch.rand((num_env_ids, 1), device=self.device), 1/3)

        return radii * coords

    def _so3_attitude_error(self):
        """SO(3) 流形姿态误差（body frame），供 3D 向量化 S 面使用。

        放弃欧拉角逐轴解耦：直接在四元数流形上取 body-frame 姿态误差 e_R 与角速度
        误差 e_omega，规避 ±π 倒置区的万向节锁与通道耦合破产。

        约定「误差 = 目标 − 当前」，故 e_R 与 e_omega 都是**修正方向**，与
        `_pid_control` 中 `inner = ζ1·e_R + ζ2·e_omega` 的 + 号配合即为标准 PD
        （+e_R 角度纠偏，+e_omega 角速度阻尼）。注意：诊断报告里写的
        e_ω = ω_body − ω_desired 配 + 号会变成反阻尼（自激），这里取其相反数
        e_omega = ω_desired − ω_body 以保证纯解析基线（geo_residual_scale=0）稳定。

        Returns:
            e_R     (N, 3): body-frame 姿态误差 = 2·sgn(w)·vec(q_curr⁻¹ ⊗ q_goal)。
            e_omega (N, 3): body-frame 角速度误差 = ω_desired − ω_body；ω_desired 由
                            相邻两步 goal 四元数有限差分得到（首步 _goal_prev=goal → 0）。
        """
        q_curr = self._robot.data.root_quat_w   # (N, 4) wxyz, body->world
        q_goal = self._goal[:, 0:4]             # (N, 4) wxyz
        # body-frame 姿态误差：q_err = q_curr⁻¹ ⊗ q_goal，其虚部即 current→desired 的转轴。
        q_err = quat_mul(quat_conjugate(q_curr), q_goal)
        sgn_e = torch.where(
            q_err[:, 0:1] >= 0.0,
            torch.ones_like(q_err[:, 0:1]),
            -torch.ones_like(q_err[:, 0:1]),
        )  # 强制最短路径，且在 180° (w=0) 处不退化为 0
        e_R = 2.0 * sgn_e * q_err[:, 1:4]
        # 参考角速度 ω_desired：相邻两步 goal 四元数有限差分（慢参考下量级很小，主作用为阻尼）。
        ctrl_dt = float(self.sim.cfg.dt) * float(getattr(self.cfg, "decimation", 1))
        q_rel = quat_mul(quat_conjugate(self._goal_prev), q_goal)
        sgn_r = torch.where(
            q_rel[:, 0:1] >= 0.0,
            torch.ones_like(q_rel[:, 0:1]),
            -torch.ones_like(q_rel[:, 0:1]),
        )
        omega_d = 2.0 * sgn_r * q_rel[:, 1:4] / ctrl_dt
        e_omega = omega_d - self._robot.data.root_ang_vel_b
        return e_R, e_omega

    def _pid_control(self, actions, actions_d, actions_i) -> torch.Tensor:
        # 将action修改为PID控制，随后输出PWM波的正规化频率。
        motorValue = torch.zeros(self.num_envs, self._num_thrusters, device=self.device)
        if not self.cfg.cascade_control:
            # RL direct PWM duty cycle path: use policy outputs directly as the four control channels.
            PID_value = torch.clip(actions, -1, 1)
        else:
            # 约束action范围，用于对action的角度(rad)进行规范化,暂定roll,pitch,yaw,depth分别设置为0.15,0.15，0.3,1
            actions = actions * self.action_lim; actions_d = actions_d * self.action_lim; actions_i = actions_i * self.action_lim
            # PID_value = actions * self.PID_args[:,:,0] + actions_d * self.PID_args[:,:,1] + actions_i * self.PID_args[:,:,2]
            # 更改为S面控制
            if self.cfg.control_method == 'Ssurface':
                coeff_ratio = self.cfg.s_ratio
                # M3 CM-C: s_surface_struct 失配——缩放 sigmoid 斜率 (s_ratio) 与加性项权威。
                s_ratio_eff = self.cfg.s_ratio * float(getattr(self, "_mismatch_s_ratio_scale", 1.0))
                PID_value = 2 / (1 + torch.exp(-s_ratio_eff *  self.PID_args[:,:,0] * actions - s_ratio_eff * self.PID_args[:,:,1] * actions_d)) - 1
                if getattr(self.cfg, "self_adapt", False):
                    add_scale = float(getattr(self, "_mismatch_add_scale", 1.0))
                    PID_value_add = 30 * (1 / 60) * (actions + 1.00 * actions_d) * add_scale
                    PID_value_add = torch.clamp(PID_value_add, -0.35, 0.35)
                    PID_value += PID_value_add
                # SO(3) 流形 S 面：在四元数误差流形上重算 roll/pitch/yaw 三轴（覆盖 0:3），
                # depth（通道 3）保持上面的旧逐轴逻辑不变。规避欧拉奇异 + 通道解耦。
                if getattr(self.cfg, "attitude_error_mode", "euler") == "so3":
                    e_R, e_omega = self._so3_attitude_error()
                    geo_z1 = float(getattr(self.cfg, "geo_zeta1", 1.0))
                    geo_z2 = float(getattr(self.cfg, "geo_zeta2", 0.5))
                    res_scale = float(getattr(self.cfg, "geo_residual_scale", 0.0))
                    # 残差 RL：actions 已 ×action_lim（见上），让策略在解析基线上学修正（保 12 维 obs）。
                    inner = geo_z1 * e_R + geo_z2 * e_omega + res_scale * actions[:, 0:3]
                    s_surf = 2 / (1 + torch.exp(-s_ratio_eff * inner)) - 1
                    # 固定分配矩阵对 roll/yaw 通道反号，乘 restoring 符号向量修正（见 cfg geo_channel_sign）。
                    PID_value[:, 0:3] = s_surf * self._geo_channel_sign
            elif self.cfg.control_method == 'PID':
                PID_value = actions * self.PID_args[:,:,0] + actions_d * self.PID_args[:,:,1] + actions_i * self.PID_args[:,:,2]
            else:
                raise ValueError(f"Unknown control_method: {self.cfg.control_method}")
        '''将数值分配到推进器。控制顺序为roll,pitch,yaw,depth
        roll控制 => 0到3，当error为正(右侧下沉)，右侧两个(1、3)输出正，左侧两个(0、2)输出负
        pitch控制 => 0到3，当error为正(头部上升)，后侧两个(2、3)输出正，前侧两个(0、1)输出负
        yaw控制 => 4到7,当error为正，即希望AUV从上看逆时针旋转，左前和右后(4、7)输出正，右前和左后(5、6)输出负
        深度控制 => 0到3，要求下沉(我们定义error = depth_desire - depth_real,换言之error为正)输出PID为正，反之为负
        '''
        # === 分配路径分叉（Step 3d）===
        # 旧硬编码 8 推混合块（base/asym/long_body/heavy_*，_use_config_alloc=False，A4 零变更）
        # vs. config 驱动 B⁺（uuv6/uuv4/uuv6_angled/uuv4_angled，任意 N 推、非正交/欠驱动）。
        if not self._use_config_alloc:
            # M3 CM-D: allocation_skew 失配——逐控制通道缩放分配权重（模拟推进器安装/分配偏差）。仅旧路径生效。
            if self.cfg.cascade_control and self.ctrl_mismatch_mode == "allocation_skew":
                PID_value = PID_value * self._mismatch_alloc_scale
            motorValue[:,0] = -PID_value[:,0] - PID_value[:,1] + PID_value[:,3] # roll, pitch, depth
            motorValue[:,1] = PID_value[:,0] - PID_value[:,1] + PID_value[:,3]
            motorValue[:,2] = -PID_value[:,0] + PID_value[:,1] + PID_value[:,3]
            motorValue[:,3] = PID_value[:,0] + PID_value[:,1] + PID_value[:,3]
            motorValue[:,4] = PID_value[:,2];
            motorValue[:,5] = -PID_value[:,2]
            motorValue[:,6] = -PID_value[:,2]
            motorValue[:,7] = PID_value[:,2]
        else:
            # config B⁺ 路径：控制通道 [roll,pitch,yaw,depth] -> body wrench -> 伪逆分配到 N 推力。
            # 非正交/欠驱动耦合由 B⁺（+ wls 权重屏蔽不可控 DOF）吸收，控制律不改。
            wrench = control_channels_to_wrench(PID_value)
            motorValue = allocate_thrust(self._alloc_B, wrench, self._alloc_mode, self._alloc_weight)
        motorValue = torch.clip(motorValue, -1, 1).to(self.device) # clip to PWM values
        # STDW wrapper hook: cache low-level adaptation correction (PID_value_add) for pseudo-action.
        if 'PID_value_add' in locals():
            self._pid_value_add_buf = PID_value_add.detach().clone()
        else:
            self._pid_value_add_buf = torch.zeros_like(PID_value).detach()
        return motorValue

    def _compute_dynamics(self, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """@dox _compute_dynamics
        @brief Map 4D PWM-style actions to body-frame thrust + moment.
        @param actions Tensor (N, 4) in [-1, 1].
        @return (thrust, moment) each shape (N, 3) in body frame.
        @details Pipeline: actions -> first-order thruster lag -> wrench composition
                 -> hydrodynamic drag/added-mass via HydrodynamicForceModels ->
                 ground-truth body forces. JONSWAP wave velocity is injected here
                 through self.get_current_fluid_velocity().
        """
        # actions are -1 for full reverse thrust, 1 for full forward thrust - THESE REPRESENT PWM VALUES
        # BASED ON LINE 91 of https://gitlab.com/warplab/ros/warpauv/warpauv_simulation/-/blob/master/src/robot_sim_interface.py
        # Args:
        #     actions (torch.Tensor): Actions shape (num_envs, num_actions)
        # Returns:
        #     [torch.Tensor]: Forces sent to the simulation
        #     [torch.Tensor]: Torques sent to the simulation

        if self._debug: print("actions: ", actions)

        thruster_forces = torch.zeros((self.num_envs, self._num_thrusters, 3), device=self.device, dtype=torch.float)
        thruster_torques = torch.zeros((self.num_envs, self._num_thrusters, 3), device=self.device, dtype=torch.float)

        # 修改：action不再输出motorValue,而是输出PWM数值
        # A3：D 项使用真实 body 角速度（roll/pitch/yaw 三轴），depth 维保留动作差分。
        action_diff = actions - self.old_actions
        if getattr(self.cfg, "d_use_ang_vel", False):
            # 把误差对时间的微分近似为 -ω_body（设定值平稳时 ė ≈ -ω）。
            # 角速度量级 ~ rad/s，远大于 action_diff（动作 [-1,1] 单步差），
            # 这里用 ctrl 步长作归一化使其与原 action_diff 同量级。
            ctrl_dt = float(self.sim.cfg.dt) * float(self.cfg.decimation)
            ang_vel_b = self._robot.data.root_ang_vel_b  # (N, 3) roll/pitch/yaw
            actions_d = action_diff.clone()
            actions_d[:, 0:3] = -ang_vel_b * ctrl_dt
        else:
            actions_d = action_diff
        # A1：D 项一阶 EMA 低通。tau > 0 时启用。
        # alpha = dt / (tau + dt)；tau→0 即直通，tau→∞ 即完全静止。
        d_tau = float(getattr(self.cfg, "d_filter_tau", 0.0))
        if d_tau > 0.0:
            ctrl_dt = float(self.sim.cfg.dt) * float(self.cfg.decimation)
            alpha = ctrl_dt / (d_tau + ctrl_dt)
            filt_prev = self._actions_d_filt.clone()
            self._actions_d_filt = self._actions_d_filt + alpha * (actions_d - self._actions_d_filt)
            actions_d = self._actions_d_filt
            lead_steps = float(getattr(self.cfg, "d_filter_phase_lead_steps", 0.0))
            if lead_steps > 0.0:
                lead = lead_steps * (self._actions_d_filt - filt_prev)
                lead_clip = float(getattr(self.cfg, "d_filter_phase_lead_clip", 0.0))
                if lead_clip > 0.0:
                    lead = torch.clamp(lead, -lead_clip, lead_clip)
                actions_d = self._actions_d_filt + lead
            self._actions_d_filt_prev = self._actions_d_filt.detach().clone()
        motorValues = self._pid_control(actions, actions_d, self.actions_i) # motorValues (num_envs, 8)
        self._last_motor_values = motorValues.clone()  # expose for saturation analysis
        self.old_actions = actions.clone()
        # motorValues = torch.clone(actions) # at this point these are PWM commands between -1 and 1

        if self._debug: print("motorValues: ", motorValues)

        # convert the PWM commands to rad/s using method in https://gitlab.com/warplab/ros/warpauv/warpauv_simulation/-/blob/master/src/robot_sim_interface.py
        # lower the PWM threshold
        ctrl_dt = float(self.sim.cfg.dt) * float(getattr(self.cfg, "decimation", 1))
        current_sim_time = self.episode_length_buf.to(device=self.device, dtype=torch.float32) * ctrl_dt
        if self.fault_injection_enabled:
            self.thruster_efficiency_factors.fill_(1.0)
            active_fault_mask = current_sim_time >= self.fault_start_sim_time
            if torch.any(active_fault_mask):
                elapsed = (current_sim_time - self.fault_start_sim_time).clamp_min(0.0)
                if self.fault_profile == "fixed":
                    degradation = torch.full_like(elapsed.unsqueeze(-1), float(self.fault_target_efficiency))
                elif self.fault_profile == "ramp_to_target":
                    duration = max(float(self.fault_ramp_duration_s), 1.0e-6)
                    frac = (elapsed / duration).clamp(0.0, 1.0).unsqueeze(-1)
                    degradation = 1.0 + frac * (float(self.fault_target_efficiency) - 1.0)
                    degradation = torch.clamp(degradation, min=0.0, max=1.0)
                else:
                    degradation = torch.clamp(1.0 - self.fault_rate_per_second * elapsed.unsqueeze(-1), min=0.0, max=1.0)
                for thruster_idx in self.fault_thrusters:
                    self.thruster_efficiency_factors[:, thruster_idx] = degradation.squeeze(-1)
        else:
            self.thruster_efficiency_factors.fill_(1.0)

        # M3 CM-B: actuator_scale 失配——在 efficiency reset 之后乘入持久推力失配，
        # 作用于 motorValue→thrust 线性段，绕开 sigmoid 饱和（默认全 1.0 恒等）。
        if self.ctrl_mismatch_mode == "actuator_scale":
            self.thruster_efficiency_factors = self.thruster_efficiency_factors * self._mismatch_actuator_scale

        # M4 B3 推进器吸气——露出水面的推进器按浸没比例折减推力权威（默认 off → 全 1.0 恒等）。
        if self.boundary_models.enable_ventilation:
            vent = self.boundary_models.compute_ventilation_factor(
                root_pos_w=self._robot.data.root_pos_w,
                root_quat_w=self._robot.data.root_quat_w,
                thruster_com_offsets=self.thruster_com_offsets,
            )
            self.thruster_efficiency_factors = self.thruster_efficiency_factors * vent

        threshold = 0.02
        motorValues[torch.abs(motorValues) < threshold] = 0 
        motorValues[motorValues >= threshold] = -139.0 * (torch.pow(motorValues[motorValues >= threshold], 2.0)) + 500 * motorValues[motorValues >= threshold] + 8.28
        motorValues[motorValues <= -threshold] = 161.0 * (torch.pow(motorValues[motorValues <= -threshold], 2.0)) + 517.86 * motorValues[motorValues <= -threshold] - 5.72

        # get the current motor velocities using thruster dynamics
        # TODO: CHECK THAT SIM DT IS CORRECT HERE
        motorValues = self.thruster_dynamics.update(motorValues, self.episode_length_buf * self.sim.cfg.dt)

        # get thruster forces from their speeds using the thruster conversion function 
        motorValues = self.thruster_conversion.convert(motorValues)
        motorValues = motorValues * self.thruster_efficiency_factors

        # TODO: this could be taken out of the physics step
        thruster_forces[..., 0] = 1.0 # start with forces in the x direction
        thruster_forces = quat_apply(self.thruster_quats, thruster_forces) # rotate the forces into the thruster's frame

        # apply the force magnitudes to the thruster forces
        thruster_forces = thruster_forces * motorValues.unsqueeze(-1) # make motorValues shape (num_envs, 6, 1))

        # calculate the thruster torques 
        # T = r x F
        # T (num_envs, num_thrusters_per_env, 3)
        # r (num_thrusters_per_env, 3)
        # F (num_envs, num_thrusters_per_env, 3)
        # it should broadcast r to be (num_envs, num_thrusters_per_env, 3)
        thruster_torques = torch.cross(self.thruster_com_offsets, thruster_forces, dim=-1)

        # now sum together all the forces/torques on each robot
        thruster_forces = torch.sum(thruster_forces, dim=-2) # sum over the thruster indices
        thruster_torques = torch.sum(thruster_torques, dim=-2) # sum over the thruster indices

        ## Calculate hydrodynamics
        if self._debug: print("gravity magnitude: ", self._gravity_magnitude) 
        buoyancy_forces, buoyancy_torques = self.force_calculation_functions.calculate_buoyancy_forces(self._robot.data.root_quat_w, self.cfg.water_rho, self.volumes, abs(self._gravity_magnitude), self.com_to_cob_offsets)

        fluid_vel_w = self.get_current_fluid_velocity()
        density_forces, density_torques, viscosity_forces, viscosity_torques = self.force_calculation_functions.calculate_density_and_viscosity_forces(
          self._robot.data.root_quat_w, self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_w, self.inertia_tensors, self.inertia_tensors_mean, self.cfg.water_beta, self.cfg.water_rho, self.masses, fluid_vel_w
        )

        # Apply drag multiplier for embodiment configurations (e.g., heavy-duty)
        drag_multiplier = self._drag_multiplier_per_env.unsqueeze(-1)
        density_forces = density_forces * drag_multiplier
        density_torques = density_torques * drag_multiplier
        viscosity_forces = viscosity_forces * drag_multiplier
        viscosity_torques = viscosity_torques * drag_multiplier

        if self._debug: print("density forces: ", density_forces)
        if self._debug: print("density torques: ", density_torques)

        if self._debug: print("viscosity forces: ", viscosity_forces)
        if self._debug: print("viscosity torques: ", viscosity_torques)

        if self._debug: print("buoyancy forces: ", buoyancy_forces)
        if self._debug: print("buoyancy torques: ", buoyancy_torques)

        if self._debug: print("thruster forces: ", thruster_forces)
        if self._debug: print("thruster torques: ", thruster_torques)

        forces = density_forces + buoyancy_forces + viscosity_forces + thruster_forces
        torques = density_torques + buoyancy_torques + viscosity_torques + thruster_torques

        # M4 近边界效应——加性 body-frame wrench 修正（默认 off → any_enabled=False → 零修正）。
        if self.boundary_models.any_enabled:
            b_df, b_dt, b_info = self.boundary_models.compute_boundary_wrench(
                root_pos_w=self._robot.data.root_pos_w,
                root_quat_w=self._robot.data.root_quat_w,
                masses=self.masses,
                com_to_cob_offsets=self.com_to_cob_offsets,
                g_mag=abs(self._gravity_magnitude),
                buoyancy_forces_b=buoyancy_forces,
                buoyancy_torques_b=buoyancy_torques,
                drag_forces_b=density_forces + viscosity_forces,
                drag_torques_b=density_torques + viscosity_torques,
            )
            forces = forces + b_df
            torques = torques + b_dt
            self._last_boundary_info = b_info

        torques = self._update_runtime_torque_pulse(torques)

        if self._debug: print("final forces", forces)
        if self._debug: print("final torques", torques)

        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_ang_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_z_ang_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_z_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_z_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "x_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/x_b"
                self.x_b_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "z_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/z_b"
                self.z_b_visualizer = VisualizationMarkers(marker_cfg)
            
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.goal_ang_visualizer.set_visibility(True)
            self.goal_z_ang_visualizer.set_visibility(True)
            self.x_b_visualizer.set_visibility(True)
            self.z_b_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

            if hasattr(self, "goal_ang_visualizer"):
                self.goal_ang_visualizer.set_visibility(False)

            if hasattr(self, "goal_z_ang_visualizer"):
                self.goal_z_ang_visualizer.set_visibility(False)

            if hasattr(self, "x_b_visualizer"):
                self.x_b_visualizer.set_visibility(False)
            
            if hasattr(self, "z_b_visualizer"):
                self.z_b_visualizer.set_visibility(False)

    def _rotate_quat_by_euler_xyz(self, q: torch.tensor, x: float|torch.tensor, y: float|torch.tensor, z: float|torch.tensor, device=None):
        # Assumes q has shape [num_envs, 4]
        num_envs = q.shape[0]
        if device == None:
            device = self.device

        if type(x) == float:
            x = torch.zeros(num_envs, device=device) + x

        if type(y) == float:
            y = torch.zeros(num_envs, device=device) + y
        
        if type(z) == float:
            z = torch.zeros(num_envs, device=device) + z

        iq = math_utils.quat_from_euler_xyz(x, y, z)
        return math_utils.quat_mul(q, iq)


    def _debug_vis_callback(self, event):
        # Visualize the goal positions
        # self.goal_pos_visualizer.visualize(translations = self._default_env_origins)
        self.goal_pos_visualizer.visualize(translations = self._goal_pos_w)

        # Visualize goal orientations
        goal_quats_w = self._goal
        ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        ang_marker_scales[:, 0] = 1
        self.goal_ang_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=goal_quats_w, scales=ang_marker_scales)

        # Visualize goal orientations via another axis
        goal_z_quat = self._rotate_quat_by_euler_xyz(goal_quats_w, 0.0, -torch.pi/2, 0.0)
        ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        ang_marker_scales[:, 0] = 1
        self.goal_z_ang_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=goal_z_quat, scales=ang_marker_scales)

        # Visualize current X-direction
        x_w = self._robot.data.root_quat_w
        x_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        x_w_marker_scales[:, 0] = 1
        self.x_b_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=x_w, scales=x_w_marker_scales)

        # Visualize current Z-direction
        z_w_quat = self._rotate_quat_by_euler_xyz(self._robot.data.root_quat_w, 0.0, -torch.pi/2, 0.0)
        z_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        z_w_marker_scales[:, 0] = 1
        self.z_b_visualizer.visualize(translations=self._robot.data.root_pos_w, orientations=z_w_quat, scales=z_w_marker_scales)


@torch.jit.script
def quat_dist(q1, q2):
    return 1 - torch.sum(q1*q2, dim=-1)**2

@torch.jit.script
def _compute_rewards(
    rew_scale_pos: float,
    rew_scale_ang: float,
    rew_scale_lin_vel: float,
    rew_scale_ang_vel: float,
    rew_scale_actions: float,
    rew_scale_action_rate: float,
    rew_scale_action_jerk: float,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    goal: torch.Tensor,
    offsets_from_origin: torch.Tensor,
    completed_envs: torch.Tensor,
    actions: torch.Tensor,
    action_rate: torch.Tensor,
    action_jerk: torch.Tensor,
):

    # Reward position accuracy, todo: scale the gaussian std appropriately
    rew_pos = rew_scale_pos * torch.exp(-1 * torch.norm(offsets_from_origin, dim=1)**2)

    # rew_pos += rew_scale_pos * torch.exp(-1 * (torch.abs(offsets_from_origin[:,2]) ** 2)) # 只保留深度控制

    # Reward angular accuracy, todo: scale the gaussian std appropriately
    # Uniquefy and normalize all quaternions
    rew_ang = rew_scale_ang * torch.exp(-1 * math_utils.quat_error_magnitude(goal[:,:], root_quat[:,:]))

    # Penalize angular velocity (D1: 真负向二次惩罚，振荡越大越扣分，提供抑制极限环的负梯度)
    rew_ang_vel = -rew_scale_ang_vel * (torch.norm(ang_vel, dim=1) ** 2)

    # # Penalize energy consumption
    rew_action = rew_scale_actions * torch.exp(-1 * torch.norm(actions, dim=1) ** 0.5)

    # 阻尼项 (D1)：从平滑加分 exp(-‖·‖²) 改为负向二次惩罚 -‖·‖²，
    # 让大幅高频抖动产生持续负梯度而非仅趋零，实测对极限环的抑制更直接。
    rew_action_rate = -rew_scale_action_rate * (torch.norm(action_rate, dim=1) ** 2)
    rew_action_jerk = -rew_scale_action_jerk * (torch.norm(action_jerk, dim=1) ** 2)

    total_rew = rew_ang + rew_action + rew_pos + rew_ang_vel + rew_action_rate + rew_action_jerk


    return total_rew


# =============================================================================
# 8 维元控制 cfg：前 4 维 a_ctrl + 后 4 维 a_gain，由 ParametricGainTuner 处理。
# 与 4 维基线通过独立 task ID（EasyUUV-Direct-Parametric-v1）和独立
# experiment_name（easyuuv_parametric）共存，避免 checkpoint / 维度混淆。
# =============================================================================
@configclass
class EasyUUVParametricEnvCfg(EasyUUVEnvCfg):
    num_actions = 8
    # A3：观测拼接 root_ang_vel_b (3 维)，从 9 增到 12。
    num_observations = 12
    # A3：S 面 D 项与观测同步使用真实 body 角速度。
    d_use_ang_vel = True
    obs_include_ang_vel = True
    # A1 已验证负面：tau=0.05 在 A3 之上叠加 EMA 导致 MSE +20%（双重低通相位滞后），
    # 已回退为 0.0；代码实现保留，留待后续不同上下文复用。
    d_filter_tau = 0.0
    tune_gains = True
    # episode 延长到 6s：让正弦在单个 episode 内走完约 1 个周期，呈现真正的波形而非斜坡。
    episode_length_s = 6.0
    # 平滑正弦参考：幅度放大到 ~0.35 rad(~20°)、频率提高到 episode 内可见完整波形。
    reference_mode = "sine_sweep"
    ref_sine_amp = [0.35, 0.35, 0.35]
    ref_sine_freq = [0.18, 0.22, 0.15]
    # D1×1/4：把负向二次惩罚 scale 降到原来的 1/4，缓解 roll/pitch MSE 退步。
    rew_scale_ang_vel = 0.0125
    rew_scale_action_rate = 0.005
    rew_scale_action_jerk = 0.0025
