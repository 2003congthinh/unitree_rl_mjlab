import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  """PPO runner for velocity tasks. Same as MjlabOnPolicyRunner (rsl_rl OnPolicyRunner:
  collect rollouts -> PPO update -> log) but also exports an ONNX policy at every save."""

  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    """Save checkpoint + deployable ONNX.

    Inputs: path = ".../<run_dir>/model_<iter>.pt", infos (extra dict stored in the .pt).
    Output: None. Files written in <run_dir>:
      model_<iter>.pt  - torch checkpoint (actor, critic, optimizer, iteration)
      policy.onnx      - actor network only, input "obs" -> output "actions",
                         with metadata (joint names, default pose, gains, obs terms ...)
                         attached; this is the file copied to deploy/robots/<robot>/config/...
    Uploads policy.onnx to W&B when the W&B logger is used.
    """
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
