"""Convert a retargeted motion CSV into the .npz reference used by tracking tasks.

ENTRY POINT of the motion-preprocessing step (run BEFORE training a tracking task).

  python scripts/csv_to_npz.py --robot g1 --input-file src/assets/motions/g1/dance1_subject2.csv \
         --output-name dance1_subject2.npz --input-fps 30 --output-fps 50

Data in  (CSV, one row per frame, no header):
  [0:3]  root position xyz (m, world)
  [3:7]  root quaternion in xyzw order (converted to wxyz here)
  [7:]   joint angles (rad) in the order of `joint_names` (29 for g1, 23 for g1_23dof)

Data out (.npz dict, T = number of output frames, J = all robot joints, B = all robot bodies):
  "fps":            [output_fps]
  "joint_pos":      (T, J)     joint angles
  "joint_vel":      (T, J)     joint velocities
  "body_pos_w":     (T, B, 3)  world position of every body link
  "body_quat_w":    (T, B, 4)  world orientation (wxyz) of every body link
  "body_lin_vel_w": (T, B, 3)  world linear velocity of every body link
  "body_ang_vel_w": (T, B, 3)  world angular velocity of every body link

Call graph: main() -> builds Scene + Simulation -> run_sim() -> MotionLoader (resample + velocities)
            -> for each frame: write state to sim, forward kinematics, record bodies -> np.savez.
"""

from typing import Any

import numpy as np
import torch
import tyro
import os
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from src.tasks.tracking.config.g1_23dof.env_cfgs import unitree_g1_23dof_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
  """Loads a CSV motion, resamples it to a new frame rate and derives velocities.

  After construction it holds (T_out = number of output frames, D = joints in CSV):
    motion_base_poss     (T_out, 3)  root position
    motion_base_rots     (T_out, 4)  root quaternion wxyz
    motion_base_lin_vels (T_out, 3)  root linear velocity  (finite difference)
    motion_base_ang_vels (T_out, 3)  root angular velocity (SO(3) finite difference)
    motion_dof_poss      (T_out, D)  joint positions
    motion_dof_vels      (T_out, D)  joint velocities
  and streams them frame by frame through get_next_state().
  """

  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    """Inputs: CSV path, input/output fps, torch device, optional (first, last) CSV line range.
    Output: None (runs _load_motion -> _interpolate_motion -> _compute_velocities)."""
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file.

    Input:  self.motion_file (CSV), optional self.line_range.
    Output: None. Sets motion_base_poss_input (N,3), motion_base_rots_input (N,4 wxyz),
            motion_dof_poss_input (N,D), input_frames = N, duration (s).
    How: np.loadtxt -> torch float32; columns 3:7 are re-ordered xyzw -> wxyz.
    """
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps.

    Input:  the *_input tensors from _load_motion.
    Output: None. Sets motion_base_poss, motion_base_rots, motion_dof_poss (T_out rows).
    How: builds output timestamps 0, 1/out_fps, ... < duration; for each timestamp finds
         the two surrounding input frames + blend factor (_compute_frame_blend), then uses
         linear interpolation for positions/joints and SLERP for the root quaternion.
    """
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors.

    Inputs: a, b (T, K) tensors; blend (T, 1) in [0, 1].
    Output: (T, K) tensor = a * (1 - blend) + b * blend.
    """
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions.

    Inputs: a, b (T, 4) wxyz quaternions; blend (T,) in [0, 1].
    Output: (T, 4) quaternions, computed row by row with mjlab's quat_slerp.
    """
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion.

    Input:  times (T_out,) output timestamps in seconds.
    Output: (index_0, index_1, blend), each (T_out,):
            index_0 = input frame just before t, index_1 = the frame after (clamped),
            blend   = fractional position of t between them.
    """
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion.

    Output: None. Sets motion_base_lin_vels (T,3), motion_dof_vels (T,D) via
            torch.gradient (central differences), and motion_base_ang_vels (T,3)
            via _so3_derivative.
    """
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).

    How: q_rel = q[t+1] * conj(q[t-1]) is the rotation over 2*dt; converting it to
    axis-angle and dividing by 2*dt gives the angular velocity (central difference).
    The first/last samples are copied so the output keeps length B.
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion.

    Output: ((base_pos (1,3), base_rot (1,4), base_lin_vel (1,3), base_ang_vel (1,3),
              dof_pos (1,D), dof_vel (1,D)), reset_flag)
            reset_flag is True when the last frame was just returned (index wraps to 0).
    """
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_path,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
):
  """Replay the motion kinematically in MuJoCo and record full-body states to .npz.

  Inputs:
    sim, scene:  a 1-env mjlab Simulation + Scene containing the robot.
    joint_names: list[str] = joint order of the CSV columns 7: .
    input_file, input_fps, output_fps, line_range: forwarded to MotionLoader.
    output_path: where the .npz is written.
    render, renderer: optionally render frames offscreen (frames are collected only).
  Output:
    None. Writes the .npz described in the module docstring.

  How it works (per output frame):
    1. Get the next root pose/velocity + joint pos/vel from MotionLoader.
    2. Write them directly into the simulator (no physics, pure kinematics).
    3. sim.forward() computes forward kinematics -> every body link's world pose/velocity.
    4. Append robot.data.{joint_pos, joint_vel, body_link_*_w} to the `log` dict.
    5. When the motion wraps (reset_flag), stack the lists to arrays and np.savez(log).
  """
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)
        np.savez(output_path, **log)  # type: ignore[arg-type]


def main(
  robot: str,
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """Replay motion from CSV file and output to npz file.

  Output: None. The .npz is saved to ./src/assets/motions/<robot>/<output_name>.

  How it works: picks the scene (from the tracking env cfg) and the CSV joint order for
  `robot` ("g1" = 29 DoF, "g1_23dof" = 23 DoF), compiles a single-env Simulation whose
  timestep equals 1/output_fps, optionally creates an OffscreenRenderer, then calls run_sim().

  Args:
    robot: "g1" or "g1_23dof".
    input_file: Path to the input CSV file.
    output_name: Path to the output npz file.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
    line_range: Range of lines to process from the CSV file.
  """
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  if robot == "g1":    # 29 Dof
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    joint_names=[
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ]
    output_dir = "./src/assets/motions/g1"
  elif robot == "g1_23dof":
    scene = Scene(unitree_g1_23dof_flat_tracking_env_cfg().scene, device=device)
    joint_names=[    # 23 Dof
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
    ]
    output_dir = "./src/assets/motions/g1_23dof"
  else:
    raise ValueError(f"Unsupported robot: {robot}")

  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()
  os.makedirs(output_dir, exist_ok=True)
  if not output_name.endswith(".npz"):
    output_name += ".npz"
  output_path = os.path.join(output_dir, output_name)

  run_sim(
    sim=sim,
    scene=scene,
    joint_names=joint_names,
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_path=output_path,
    render=render,
    line_range=line_range,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
