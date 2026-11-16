# Task registration for G1 velocity tracking.
# This file runs when `import src.tasks` is executed by train.py / play.py.
# register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls) stores the
# configs in mjlab's global registry; train.py later calls load_env_cfg(task_id),
# load_rl_cfg(task_id) and load_runner_cls(task_id) to retrieve them.
# Every other robot folder (go2, a2, as2, g1_23dof, h1_2, h2, r1) follows this pattern.
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-Rough",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Flat",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
