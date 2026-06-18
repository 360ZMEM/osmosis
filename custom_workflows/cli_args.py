# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument(
        "--workflow_config",
        type=str,
        default=None,
        help="Path, module, or module:object spec for workflow config overrides.",
    )
    arg_group.add_argument(
        "--logs_root",
        type=str,
        default=None,
        help="Override the root directory for training logs and checkpoints.",
    )
    arg_group.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Override the root directory for evaluation CSV outputs.",
    )
    arg_group.add_argument(
        "--artifacts_root",
        type=str,
        default=None,
        help="Override the root directory for exported policies and artifacts.",
    )
    arg_group.add_argument(
        "--mode",
        type=str,
        default="fixed",
        choices={"fixed", "llm", "expert", "stdw"},
        help="Evaluation mode: fixed policy, expert baseline, LLM-assisted adaptation, or STDW adaptation.",
    )
    arg_group.add_argument(
        "--llm_model_name",
        type=str,
        default=None,
        help="LLM model name used when --mode llm is enabled. Falls back to the shared LLM config when omitted.",
    )
    arg_group.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="OpenAI API key placeholder used when --mode llm is enabled.",
    )
    arg_group.add_argument(
        "--llm_base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL placeholder used when --mode llm is enabled.",
    )
    arg_group.add_argument(
        "--llm_interval_s",
        type=float,
        default=None,
        help="Minimum simulated seconds between LLM queries. Falls back to the shared LLM config when omitted.",
    )
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- play arguments
    arg_group.add_argument("--play_checkpoint", type=str, default=None, help="Checkpoint file to play from")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(
    task_name: str, args_cli: argparse.Namespace, config_overrides: dict[str, object] | None = None
) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    if config_overrides:
        from workflow_config import apply_config_overrides

        apply_config_overrides(rslrl_cfg, config_overrides)

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        rslrl_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        rslrl_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        rslrl_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        rslrl_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        rslrl_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        rslrl_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if rslrl_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        rslrl_cfg.wandb_project = args_cli.log_project_name
        rslrl_cfg.neptune_project = args_cli.log_project_name

    return rslrl_cfg
