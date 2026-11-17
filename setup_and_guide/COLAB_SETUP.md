# Running unitree_rl_mjlab on Google Colab

Yes, partly. The **training side** (Python) can run on Colab. The **deployment side** (C++ controller and `simulate/`) cannot.

## What can run on Colab

| Part | On Colab? | Why |
|---|---|---|
| `scripts/train.py` | Yes | It needs Linux and an NVIDIA GPU, which Colab provides. That actually suits it better than a Windows machine, since the setup guide recommends Ubuntu. |
| `scripts/csv_to_npz.py` | Yes | Headless, no window needed. |
| `scripts/list_envs.py` | Yes | |
| `scripts/play.py` | Awkward | Colab has no display. It falls back to the Viser web viewer on port 8080, which Colab can't show without port forwarding. Record a video instead (see the caveats below). |
| `scripts/visualize_terrain.py` | Awkward | Same Viser/port problem. |
| `deploy/` (`g1_ctrl`) | No | It must talk to the robot over Ethernet (DDS) and read a gamepad. |
| `simulate/` (`unitree_mujoco`) | No | It needs a desktop window and a gamepad. |

The usual workflow: **train on Colab → download `policy.onnx` (plus the `.npz` for mimic) → deploy from a local Linux PC.**

## Setup steps

**1. Pick a GPU runtime.** Go to Runtime → Change runtime type → GPU. The free T4 works but is slow. L4 or A100 (paid) is much better for 4096 environments.

**2. Get the code and install it:**

```python
!nvidia-smi
!git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git   # or upload your local copy / use Drive
%cd unitree_rl_mjlab
!pip install -e .        # pulls mjlab==1.2.0 and mujoco-warp==3.5.0
```

You can skip the `apt install libyaml-cpp-dev ...` step in `doc/setup_en.md`. Those packages are only for building the C++ code. The guide recommends Python 3.11 and Colab ships 3.12. That's probably fine, but if `pip install` fails on a version conflict, the Python version is the first thing to check.

**3. Enable offscreen rendering** (needed for videos; this is the standard MuJoCo-on-Colab setup):

```python
!mkdir -p /usr/share/glvnd/egl_vendor.d
!printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
%env MUJOCO_GL=egl
```

**4. Save logs to Google Drive.** Colab deletes local files when the session ends, and free sessions disconnect after a few hours. A full run is 10,001 iterations for velocity and 30,001 for tracking.

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/rl_logs && ln -s /content/drive/MyDrive/rl_logs logs
```

**5. Logging.** mjlab probably uses Weights & Biases by default (not confirmed). If training asks for a W&B key, run `!wandb login`, or try `--agent.logger=tensorboard`.

**6. Train:**

```python
!python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

If the T4 runs out of memory, lower the count to `--env.scene.num-envs=2048` or `1024`.

**7. Resume after a disconnect.** `train.py` already supports this:

```python
!python scripts/train.py Unitree-G1-Flat --agent.resume=True --agent.load-run=<run_folder_name>
```

## Caveats

- **Watching the result:** `play.py --video` records an MP4 to `logs/rsl_rl/<exp>/<run>/videos/play/`. But `play.py` then keeps running the Viser viewer forever, so the cell never finishes on its own. Wait for the video file to appear, then stop the cell. A cleaner fix is a small headless rollout script that only records the video and exits.
- **Motion imitation:** run `csv_to_npz.py` on Colab too, and keep the `.npz` on Drive. It must be copied to `deploy/robots/g1/config/policy/mimic/...` later.
- **Speed:** free T4 sessions will be noticeably slower than a desktop RTX card. Expect long runs and plan to resume them several times.
