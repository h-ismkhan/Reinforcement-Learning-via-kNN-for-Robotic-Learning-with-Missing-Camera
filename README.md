# RL-BFS + Fusion Head on LIBERO

> Setup guide for running RL-BFS with a Fusion Head on the LIBERO benchmark (Spatial, Goal, or Object tasks).

---

## Tested Environment

| Component | Version |
|-----------|---------|
| OS | Linux (Ubuntu) |
| Python | 3.12 |
| GPU | NVIDIA (CUDA) |
| Conda env | py12 |

---

## Table of Contents

1. [LIBERO Installation](#1-libero-installation)
2. [Additional pip Packages](#2-additional-pip-packages)
3. [Manual File Edits in LIBERO Source](#3-manual-file-edits-in-libero-source)
4. [Script Dependencies](#4-script-dependencies)
5. [Running the Script](#5-running-the-script)
6. [Common Errors and Fixes](#6-common-errors-and-fixes)

---

## 1. LIBERO Installation

**Clone the repository and install in editable mode:**

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

**Download the dataset** (spatial, goal, object, etc.):

```bash
python benchmark_scripts/download_libero_datasets.py
```

When prompted for a custom path, enter something like:

```
/home/omega/Documents/py-ipynb/4-Libero/datasets
```

> Update the `DATASET_ROOT` variable in your script to match this path.

---

## 2. Additional pip Packages

These are not installed automatically by LIBERO:

```bash
pip install bddl
pip install cloudpickle
pip install gymnasium
pip install gym==0.26.2
```

> **Note on `gym`:** `gym==0.21.0` does not build on Python 3.12. Use `0.26.2` instead. A deprecation warning will appear at runtime — this is harmless.

> **Note on `robosuite`:** Version `1.4.1` is required. Check your installed version:
> ```bash
> pip show robosuite | grep Version
> ```
> If it is `1.5` or higher, downgrade:
> ```bash
> pip install robosuite==1.4.1
> ```

---

## 3. Manual File Edits in LIBERO Source

PyTorch 2.6 changed the default behaviour of `torch.load()` to require `weights_only=True`. LIBERO was written before this change, so without these edits every environment creation will fail with a `"Weights only load failed"` error.

You need to edit **4 files** inside your LIBERO installation (e.g. `/home/omega/LIBERO/libero/`).

### Option A — Apply with `sed` (recommended)

Run from inside your LIBERO folder:

```bash
sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g' \
    libero/libero/benchmark/__init__.py \
    libero/lifelong/metric.py \
    libero/lifelong/evaluate.py

sed -i 's/model_dict = torch.load(model_path, map_location=map_location)/model_dict = torch.load(model_path, map_location=map_location, weights_only=False)/' \
    libero/lifelong/utils.py
```

Verify the changes applied:

```bash
grep -n "torch.load" libero/libero/benchmark/__init__.py
# Should show weights_only=False on the relevant line
```

### Option B — Edit manually

<details>
<summary>Click to expand manual edit instructions</summary>

**File 1:** `libero/libero/benchmark/__init__.py` (~line 164)
```python
# Before
init_states = torch.load(init_states_path)
# After
init_states = torch.load(init_states_path, weights_only=False)
```

**File 2:** `libero/lifelong/utils.py` (~line 59)
```python
# Before
model_dict = torch.load(model_path, map_location=map_location)
# After
model_dict = torch.load(model_path, map_location=map_location, weights_only=False)
```

**File 3:** `libero/lifelong/metric.py` (~line 107)
```python
# Before
init_states = torch.load(init_states_path)
# After
init_states = torch.load(init_states_path, weights_only=False)
```

**File 4:** `libero/lifelong/evaluate.py` (~line 254)
```python
# Before
init_states = torch.load(init_states_path)
# After
init_states = torch.load(init_states_path, weights_only=False)
```

</details>

---

## 4. Script Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install scikit-learn
pip install numpy
pip install h5py
pip install Pillow
```

Most of these will already be present in a standard PyTorch + Hugging Face environment.

---

## 5. Running the Script

```bash
python [file-name].py
```

The script runs in three phases:

| Phase | Description |
|-------|-------------|
| **Phase 1** | Extracts CLIP embeddings; trains prediction RL policy and fusion head on full modalities |
| **Phase 2** | Trains per-camera imputation RL policies and soft imputation heads (for camera-dropout evaluation) |
| **Phase 3** | Runs live rollouts in the LIBERO simulator; reports success rates for Tables 1, 3, and 4 |

### Checkpoints

The following checkpoint files are saved during training:

```
best_rl_libero_with_unn.pt
best_fusion_libero_with_unn.pt
best_imp_libero_with_unn_mod0.pt
best_imp_libero_with_unn_mod1.pt
best_soft_imp_libero_with_unn_mod0.pt
best_soft_imp_libero_with_unn_mod1.pt
```

### Skipping Training

If checkpoints already exist and you only want to rerun the simulator evaluation, set the following flag in your script and rerun:

```python
SKIP_TRAINING = True
```

### Results

Results are printed to the terminal and saved to:

```
results_[dataset-name]_with_unn.json
```

---

## 6. Common Errors and Fixes

| Error / Warning | Fix |
|-----------------|-----|
| `No module named 'bddl'` | `pip install bddl` |
| `No module named 'cloudpickle'` | `pip install cloudpickle` |
| `No module named 'gym'` | `pip install gym==0.26.2` |
| `No module named 'robosuite.environments.manipulation.single_arm_env'` | `pip install robosuite==1.4.1` |
| `Weights only load failed ... numpy.core.multiarray._reconstruct` | Apply the file edits in [Part 3](#3-manual-file-edits-in-libero-source) |
| `No module named 'robomimic'` | This import is not needed — ensure you are using the latest version of the script, which does not import `robomimic` |
| ⚠️ `Gym has been unmaintained since 2022` | Warning only — safe to ignore |
| ⚠️ `No private macro file found` (robosuite) | Warning only — safe to ignore |
| ⚠️ `Using a slow image processor` (Hugging Face CLIP) | Warning only — results are not affected |
