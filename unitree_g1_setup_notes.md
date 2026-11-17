# Unitree G1 Motion Tracking — Setup & Command Log

Project: train an RL motion-tracking policy (PPO, BeyondMimic-style) for the Unitree G1 using `unitree_rl_mjlab`, with reference motions from BONES-SEED.

Platform: Windows 11, `cmd`, Anaconda env `unitree_rl_mjlab`, NVIDIA RTX 4060 (8 GB).

---

## 1. Working library versions

| Package | Version | Note |
|---|---|---|
| Python | 3.11 | Recommended by the repo setup guide |
| mjlab | 1.2.0 | Pinned by `unitree_rl_mjlab`'s `setup.py` |
| mujoco | **3.5.0** | 4.x breaks `mujoco-warp` 3.5.0 (see Fix 1) |
| mujoco-warp | 3.5.0 | GPU physics backend |
| warp-lang | **1.12.0** | 1.19 too new, 1.11 too old (see Fixes 4–5) |
| torch | **CUDA build** | Default Windows pip install is CPU-only (see Fix 3) |
| scipy | 1.17.1 | Missing dependency (see Fix 2) |
| huggingface_hub | 2.4.0 | CLI command is `hf`, not `huggingface-cli` |
| pyarrow | latest | Needed by pandas to read `.parquet` |
| tensorboard | (installed with mjlab) | Used instead of wandb (see Fix 6) |
| numpy | 2.4.6 | |

Hardware/driver as reported by Warp: CUDA Toolkit 12.9, Driver 13.3, RTX 4060 8 GiB (sm_89).

Full snapshot of the working environment:

```bat
pip freeze > requirements-working.txt
```
Records every installed package with its exact version. Restore later with `pip install -r requirements-working.txt`.

---

## 2. Bug-fix summary

| # | Error | Cause | Fix |
|---|---|---|---|
| 0 | `'huggingface-cli' is not recognized` | `huggingface_hub` 2.x renamed the CLI | Use `hf` |
| 0b | `Unable to find a usable engine; tried 'pyarrow'` | pandas can't read parquet alone | `pip install pyarrow` |
| 0c | Filter returned 0 rows | `content_props` uses `0` for "no props", not empty | Filter on `content_props.astype(str) == '0'` |
| 1 | `mjtEnableBit has no attribute 'mjENBL_MULTICCD'` | mujoco 4.0.1 installed; mujoco-warp 3.5.0 expects 3.5 | Uninstall mujoco 4 packages, install `mujoco==3.5.0` |
| 2 | `No module named 'scipy'` | Missing dependency | `pip install scipy` |
| 3 | `Torch not compiled with CUDA enabled` | CPU-only torch | Install torch from the PyTorch CUDA index |
| 4 | `module 'warp' has no attribute 'context'` | warp-lang 1.19 removed `wp.context` | Downgrade warp-lang |
| 5 | `unsupported operand type(s) for \|: 'type' and 'function'` | warp-lang 1.11 is below mjlab's minimum (≥1.12) | `pip install "warp-lang==1.12.0"` |
| 6 | `ValidationError ... start_method ... Extra inputs are not permitted` | wandb too new for rsl_rl's logger | Train with `--agent.logger=tensorboard` |
| 7 | `IndexError: list index out of range` in `runner.py` line 119 | Path split on `/`; Windows uses `\` | Normalize slashes before splitting (code below) |

---

## 3. Commands in chronological order

### 3.1 Hugging Face login

```bat
huggingface-cli login
```
**Failed.** The old CLI name no longer exists in `huggingface_hub` 2.x.

```bat
pip show huggingface_hub
```
Checks whether the package is installed and which version. Result: 2.4.0, installed in the conda env.

```bat
hf auth login
hf auth whoami
```
Logs in with a Hugging Face **Read** token (input is hidden when pasting), then confirms which account is logged in. The BONES-SEED license must be accepted on the dataset page first, or downloads return 403.

Fallback if `hf` is not on PATH:
```bat
python -c "from huggingface_hub import login; login()"
```

### 3.2 Download BONES-SEED metadata

```bat
hf download bones-studio/seed --repo-type dataset --include "metadata/*" --local-dir ..\bones-seed
```
Downloads only the metadata folder (not the 23.5 GB motion archives) to a folder **outside** the git repo, so large data never gets committed.

### 3.3 Inspect the metadata

```bat
python -c "import pandas as pd, glob; f=glob.glob(r'..\bones-seed\metadata\*.parquet')[0]; df=pd.read_parquet(f); print(df.shape); print(df.columns.tolist())"
```
Loads the metadata table and prints its size and column names. **First run failed** (no parquet engine).

```bat
pip install pyarrow
```
Adds parquet support to pandas. Rerunning the command gave `(142220, 51)`: 142,220 motions, 51 columns.

Useful columns: `move_duration_frames`, `category`, `content_type_of_movement`, `content_body_position`, `content_props`, `is_mirror`, `is_neutral`, `content_short_description`, `move_g1_path`.

### 3.4 `explore.py` — value counts of key columns

```python
import pandas as pd, glob
df = pd.read_parquet(glob.glob(r'..\bones-seed\metadata\*.parquet')[0])

for c in ['package', 'category', 'content_type_of_movement',
          'content_body_position', 'content_props', 'is_mirror', 'is_neutral']:
    print(f'\n== {c} ==')
    print(df[c].value_counts(dropna=False).head(15))

print('\nmean frames:', df.move_duration_frames.mean())
```
Shows what values each column actually contains, so filters can be written correctly.

Findings:
- `content_props == 0` means "no props".
- `content_body_position == 'standing'` covers 122k motions.
- Mean length is 875.8 frames, consistent with **120 fps** for the SOMA data (288 h × 3600 × 120 / 142,220 ≈ 875).

### 3.5 `filter.py` — pick short, simple clips

First version returned 0 rows because it treated `content_props` as empty for "no props". Corrected version:

```python
import pandas as pd, glob
df = pd.read_parquet(glob.glob(r'..\bones-seed\metadata\*.parquet')[0])

sel = df[
    (df.is_mirror.astype(str).str.lower() == 'false')          # skip mirrored duplicates
    & (df.content_props.astype(str) == '0')                    # 0 = no props
    & (df.content_body_position == 'standing')                 # upright only
    & (df.is_neutral == 1)                                     # neutral style, no acting
    & (df.move_duration_frames.between(360, 720))              # 3-6 s at 120 fps
    & (df.content_type_of_movement.isin(['walking', 'gesture', 'standing idle']))
]
print('matches:', len(sel))

cols = ['move_name', 'move_duration_frames', 'content_type_of_movement',
        'content_short_description', 'move_g1_path']
print(sel[cols].sample(min(20, len(sel)), random_state=0).to_string())
```
Result: 6,984 matching clips. Chosen first candidate: `walk_ff_loop_180_R_003__A067` ("strolling forward", 523 frames), path `g1/csv/221117/walk_ff_loop_180_R_003__A067.csv`.

`move_g1_path` is relative to the dataset root and points **inside `g1.tar.gz`**; the file isn't on disk until the archive is downloaded and extracted (see 3.12).

### 3.6 First conversion attempt (example clip)

```bat
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance1_subject2.csv --output-name dance1_subject2.npz --input-fps 30 --output-fps 50 --robot g1
```
Converts the repo's bundled example motion (30 fps CSV of G1 joint angles) into the 50 Hz `.npz` format the trainer uses, computing body positions/velocities via forward kinematics. Used first because it is already in the correct format, so any error comes from the environment, not the data.

Windows `cmd` note: `\` line continuations from Linux docs don't work in `cmd`; write commands on one line (or use `^`).

**Failed** with `mjENBL_MULTICCD` → Fix 1.

### 3.7 Fix 1 — MuJoCo version mismatch

```bat
pip show mujoco mujoco-warp mjlab warp-lang
pip check
python -c "import importlib.metadata as m; print([r for r in m.requires('mjlab') if 'mujoco' in r.lower() or 'warp' in r.lower()])"
```
Shows installed versions, checks for declared conflicts, and prints what mjlab requires. Result: mujoco 4.0.1 vs mujoco-warp 3.5.0; mjlab only requires `mujoco>=3.5.0` (no upper bound), so pip had installed 4.x.

```bat
pip uninstall -y mujoco mujoco-core mujoco-studio
pip install "mujoco==3.5.0"
python -c "import mujoco, mujoco_warp; print(mujoco.__version__, 'ok')"
```
Removes MuJoCo 4 (which is split into three packages), installs 3.5.0, and checks both libraries import. Result: `3.5.0 ok`.

### 3.8 Fix 2 — missing scipy

```bat
pip install scipy
```
mjlab's terrain module imports `scipy.interpolate`, which wasn't installed.

### 3.9 Fix 3 — CPU-only PyTorch

```bat
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
Shows the GPU and max CUDA version the driver supports, and whether torch can use the GPU.

```bat
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cuXXX
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Replaces the CPU build with a CUDA build (take the exact `cuXXX` from pytorch.org → Get Started, at or below the driver's CUDA version). The check should print `True` and `NVIDIA GeForce RTX 4060`.

### 3.10 Fixes 4 & 5 — Warp version

warp-lang 1.19.0 → `no attribute 'context'` (too new).

```bat
python -c "import importlib.metadata as m; print([r for r in m.requires('mujoco-warp') if 'warp' in r.lower()])"
```
Prints the Warp version that `mujoco-warp` needs, to choose a version that satisfies both packages.

warp-lang 1.11.0 → `unsupported operand type(s) for |` (below mjlab's minimum of 1.12).

```bat
pip install "warp-lang==1.12.0"
```
Installs the version that works with both mjlab 1.2.0 and mujoco-warp 3.5.0. The `warp.context ... will soon be removed` DeprecationWarning is harmless.

After this, conversion succeeded and produced `src/assets/motions/g1/dance1_subject2.npz`.

### 3.11 Training

```bat
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=1024
```
Trains a PPO tracking policy on the motion with 1024 parallel simulated robots (the README's 4096 doesn't fit well in 8 GB VRAM). The first run spends a few minutes compiling GPU kernels.

**Failed** at logger setup (wandb) → Fix 6:

```bat
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=1024 --agent.logger=tensorboard
```
Same training, logging to TensorBoard (no account needed) instead of wandb.

**Failed** when saving the first checkpoint → Fix 7. In `src\tasks\tracking\rl\runner.py`, line 119:

```python
# before (breaks on Windows paths with backslashes)
filename = policy_path.split("/")[-2] + ".onnx"

# after
filename = policy_path.replace("\\", "/").split("/")[-2] + ".onnx"
```
Converts Windows `\` to `/` before splitting, so the ONNX file is named after the run folder as on Linux.

Check for other path splits with the same problem:
```bat
findstr /s /n /c:"split(" src\*.py scripts\*.py
```

After this, training runs. Checkpoints go to `logs\rsl_rl\g1_tracking\<date_time>\model_<iter>.pt`, and `policy.onnx` is exported for deployment.

What the training printout shows (useful for the report):
- **Action (29):** target joint positions for the G1's 29 joints.
- **Actor observations (154):** reference command (58), anchor orientation, angular velocity, joint pos/vel, last action. What the real robot can sense.
- **Critic observations (286):** actor inputs plus simulator-only body positions/orientations and linear velocity (asymmetric actor-critic).
- **Rewards:** root and body position/orientation/velocity tracking (weights 0.5–1.0), penalties for action rate (−0.1), joint limits (−10), self-collisions (−10).
- **Terminations:** timeout, or drifting too far from the reference (anchor position/orientation, end-effector position).
- **Domain randomization:** random pushes every 1–3 s, center-of-mass offset, encoder bias, foot friction.
- **Networks:** actor and critic MLPs, 512 → 256 → 128, ELU.

`num-envs` is the number of parallel robot copies, not a dataset to cover. Fewer environments means fewer samples per iteration, not missing data.

### 3.12 Monitoring (second terminal, while training)

```bat
tensorboard --logdir logs\rsl_rl\g1_tracking
```
Serves training curves at http://localhost:6006. Watch mean reward and mean episode length; stop training (Ctrl+C) when they plateau.

```bat
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --help | findstr /i iter
```
Finds the flag that sets the maximum number of training iterations.

### 3.13 Stopping and resuming training

**Where checkpoints are stored.** Each run gets its own timestamped folder:

```
logs\rsl_rl\g1_tracking\<date_time>\model_<iter>.pt
```
e.g. `logs\rsl_rl\g1_tracking\2026-11-17_22-15-40\model_1500.pt`.

```bat
dir /s /b logs\rsl_rl\g1_tracking\*.pt
```
Lists every saved checkpoint across all runs. The highest-numbered file in a folder is that run's latest policy.

Checkpoints are saved **every N iterations** (the config's save interval), not continuously. After Ctrl+C, progress since the last saved `model_<iter>.pt` is lost.

```bat
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --help | findstr /i "save resume load"
```
Shows the save interval and the exact names of the resume flags.

**Resuming.** The README lists `--agent.resume`. In rsl_rl-style configs it works with:
- `resume`: turn resuming on
- `load-run`: which run folder to load (default usually the latest run)
- `load-checkpoint`: which file in it (default usually the latest `model_*.pt`)

Resume from the latest run and checkpoint:
```bat
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=1024 --agent.logger=tensorboard --agent.resume=True
```

Resume from a specific run and checkpoint:
```bat
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=1024 --agent.logger=tensorboard --agent.resume=True --agent.load-run=2026-11-17_22-15-40 --agent.load-checkpoint=model_1500.pt
```
`load-run` and `load-checkpoint` are best guesses from similar configs; use the spelling shown by `--help`.

**Keep the same when resuming:**
- **Motion file.** A new clip should be a new training run, not a resume.
- **Task name**, so the network input/output shapes match.
- `num-envs` can change, but keeping it the same makes curves easier to compare.

**What to expect:** resuming usually creates a **new** timestamped folder that continues the iteration count from the loaded checkpoint; TensorBoard shows both runs, with the curve picking up where it stopped. A path error during resume is likely the same Windows `/` vs `\` issue as Fix 7.

---

## 4. Next steps (not yet run)

### 4.1 Evaluate the policy

```bat
dir /s /b logs\rsl_rl\g1_tracking\*.pt
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/<run_folder>/model_<iter>.pt
```
Lists saved checkpoints, then replays the trained policy in a MuJoCo viewer. May hit the same `split("/")` Windows bug.

### 4.2 Get a BONES-SEED G1 clip

```bat
hf download bones-studio/seed g1/csv/221117/walk_ff_loop_180_R_003__A067.csv --repo-type dataset --local-dir ..\bones-seed
```
Tries to download one CSV directly. If it's not available as a single file:

```bat
hf download bones-studio/seed g1.tar.gz --repo-type dataset --local-dir ..\bones-seed
cd ..\bones-seed
tar -tzf g1.tar.gz | findstr walk_ff_loop_180_R_003__A067
tar -xzf g1.tar.gz g1/csv/221117/walk_ff_loop_180_R_003__A067.csv
```
Downloads the 23.5 GB archive (check disk space first), finds the exact path inside it, and extracts only the chosen clip.

### 4.3 Check the CSV format against the example

```python
import pandas as pd
a = pd.read_csv(r'src\assets\motions\g1\dance1_subject2.csv', header=None, nrows=3)
b = pd.read_csv(r'src\assets\motions\g1\walk_ff_loop_180_R_003__A067.csv', header=None, nrows=3)
print('example:', a.shape); print(a.iloc[:, :8])
print('bones:  ', b.shape); print(b.iloc[:, :8])
print('bones total rows:', len(pd.read_csv(r'src\assets\motions\g1\walk_ff_loop_180_R_003__A067.csv', header=None)))
```
Compares column count, header and value layout. If the BONES row count equals `move_duration_frames` (523), the CSV is 120 fps. If layouts differ, a conversion script is needed before `csv_to_npz.py`.

### 4.4 Convert and train the BONES clip

```bat
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/walk_ff_loop_180_R_003__A067.csv --output-name walk_ff_loop_180_R_003__A067.npz --input-fps 120 --output-fps 50 --robot g1
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/walk_ff_loop_180_R_003__A067.npz --env.scene.num-envs=1024 --agent.logger=tensorboard
```
Same pipeline as the example; only the file name and `--input-fps` change (confirm the FPS in 4.3 first).

---

## 5. Git hygiene

Add to `.gitignore`:
```gitignore
logs/
*.npz
```
Keep `bones-seed/` outside the repo folder. Large files committed by accident make pushes fail (GitHub rejects files over 100 MB).
