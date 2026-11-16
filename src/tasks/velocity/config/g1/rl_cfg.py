"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task.

  Input: none. Output: RslRlOnPolicyRunnerCfg with
    actor / critic : MLP 512-256-128, ELU, observation normalisation;
                     actor outputs a Gaussian (mean = action, learnable scalar std).
    algorithm      : PPO (clip 0.2, gamma 0.99, GAE lambda 0.95, 5 epochs x 4 mini-batches,
                     adaptive LR targeting KL 0.01, entropy bonus 0.01).
    num_steps_per_env=24 -> each iteration collects 24 steps x num_envs transitions.
    experiment_name="g1_velocity" -> logs/rsl_rl/g1_velocity/...
    save_interval=100, max_iterations=10001.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
