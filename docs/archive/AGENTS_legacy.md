# AGENTS

本工作区是一个 Isaac Lab 的直接强化学习任务，目标对象是螺旋桨驱动的 AUV。代码里仍以 WarpAUV 作为主命名，而较新的 README 使用了 EasyUUV 品牌名。以代码和注册信息为准，不要把 README_final 当成当前可执行真相。

## 事实基线

- 已注册的 Gym 任务是 [__init__.py](__init__.py) 里的 `MOGA-WarpAUV-Direct-v1`。
- 默认 PPO 实验名是 [agents/rsl_rl_ppo_cfg.py](agents/rsl_rl_ppo_cfg.py) 里的 `warpauv_direct`。
- 规范的环境入口是 [warpauv_env.py](warpauv_env.py)；其他环境文件属于变体或后期实验版本。
- 和这个仓库最相关的可运行脚本都在 [custom_workflows/](custom_workflows/) 下，分析行为时优先看这里。

## 运行基线

- 训练入口是 [custom_workflows/train.py](custom_workflows/train.py)。默认任务名是 `MOGA-WarpAUV-Direct-v1`，默认并行环境数是 `512`；它还支持 `--video`、`--cpu`、`--disable_fabric`、`--resume`、`--logger` 和 `--max_iterations`。
- 两份 README 只是历史风格参考：[README_mit.md](README_mit.md) 仍然使用 `MOGA-WarpAUV-Direct-v1` 和 `2048` 环境；[README_final.md](README_final.md) 改成了 `EasyUUV-Direct-v1`、`1024` 环境和 `--headless`，但当前代码并没有注册 `EasyUUV-Direct-v1`，不能直接照搬执行。
- 推理和评估相关脚本主要看 [custom_workflows/play.py](custom_workflows/play.py)、[custom_workflows/play_eval.py](custom_workflows/play_eval.py)、[custom_workflows/play_poshold.py](custom_workflows/play_poshold.py)、[custom_workflows/play_controls.py](custom_workflows/play_controls.py)、[custom_workflows/play_pid.py](custom_workflows/play_pid.py)。
- 结果默认写到 `source/results/rsl_rl/<experiment_name>/<load_run>/<checkpoint>_play/`；[custom_workflows/plot_metrics.py](custom_workflows/plot_metrics.py) 会从这个目录读取 `output.csv`。

## 环境变体

- [warpauv_env.py](warpauv_env.py) 是基线版本，也是 Gym 注册真正指向的文件。
- [warpauv_env_new.py](warpauv_env_new.py)、[warpauv_env_real.py](warpauv_env_real.py)、[warpauv_env_highobs.py](warpauv_env_highobs.py) 是环境变体；[old/env_backups/warpauv_env.py.bak](old/env_backups/warpauv_env.py.bak) 和 [old/env_backups/warpauv_env.py.bak2](old/env_backups/warpauv_env.py.bak2) 是历史备份。除非确认 import 路径和注册关系，否则不要默认它们生效。
- 修改任何环境变体时，要一起检查观测维度、控制方式和设备放置；这些文件之间本来就有意做了分叉。

## 命名与文档规则

- 描述当前实现时，优先使用 `WarpAUV`、`warpauv_direct` 和 `MOGA-WarpAUV-Direct-v1`。
- [README_mit.md](README_mit.md) 和 [README_final.md](README_final.md) 只适合作为链接参考，不要把其中的完整安装步骤重复写进代理说明里。
- [README_final.md](README_final.md) 里提到的 `workflows/gen_policy.py`、`play_eval_task1.py`、`play_eval_task2.py`、`play_eval_step.py`、`play_controller.py` 在这个工作区里并不存在；对应的本地脚本都在 [custom_workflows/](custom_workflows/) 中。

## 常见坑

- 多个评估脚本硬编码了本地 checkpoint 路径，而且依赖 `wandb`；在把脚本当通用入口之前，一定先看目标脚本本身。
- 有些脚本默认依赖 CUDA 字符串或特定 GPU 编号；在 CPU-only 或不同显卡环境下要谨慎。
- URDF 到 USD 的流水线配置在 [data/warpauv/config.yaml](data/warpauv/config.yaml)；如果资产生成或导入路径发生变化，这个文件和资产引用要一起改。
- 任务名或实验名不要只改一处。如果改名，就要同步更新注册、CLI 默认值、日志路径和 checkpoint 查找逻辑。

## 先看什么

- 先看 [__init__.py](__init__.py)、[agents/rsl_rl_ppo_cfg.py](agents/rsl_rl_ppo_cfg.py)、[custom_workflows/train.py](custom_workflows/train.py) 和对应环境文件。
- 如果后续要继续整理文档，请继续把 MIT 时代的 WarpAUV 说明和后期 EasyUUV 说明分开，不要混写成一套。
