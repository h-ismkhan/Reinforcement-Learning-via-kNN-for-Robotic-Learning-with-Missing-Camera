"""
rl_libero_spatial.py
====================
RL-BFS + Soft Fusion Head pipeline applied to LIBERO-Spatial.

Faithfully implements the method from RL-4.txt for robot IL:

  Modalities (3, no audio):
    • agent-cam   : CLIP ViT-B/32 vision  → 512-dim  (mod 0)
    • in-hand-cam : CLIP ViT-B/32 vision  → 512-dim  (mod 1)
    • language    : CLIP text encoder     → 512-dim  (mod 2, always present)

  Missing-modality handling (Tables 3/4 — camera dropout):
    NEVER zero-fill. Always use the two-stage imputation pipeline:
      1. Per-modality RL policy (BFS over training donors with that camera)
         selects top-K' donors via learned scoring.
      2. SoftImputationHead (cross-attention over top-K' donors)
         produces the imputed embedding block.
    This is identical to the imputation described in RL-4.txt.

  Prediction:
    Always via FusionHead (soft cross-attention over top-K' RL-ranked
    candidates) — hard argmax is NEVER used at inference.

  Output:
    Retrieved training demo's action sequence replayed open-loop in sim.
    Success rate = % rollouts completing the task within 260 steps.
    Protocol: 3 seeds × 25 rollouts × 10 tasks (matches DisDP paper).

═══════════════════════════════════════════════════════════════════════════
ALL TUNEABLE PARAMETERS ARE IN THE "PARAMETERS" BLOCK BELOW.
═══════════════════════════════════════════════════════════════════════════
"""

# ── standard library ──────────────────────────────────────────────────────────
import os, copy, heapq, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# ── Libero ─────────────────────────────────────────────────────────────────
from libero.libero.benchmark import get_benchmark_dict

# ── CLIP ───────────────────────────────────────────────────────────────────
from transformers import CLIPProcessor, CLIPModel

# =============================================================================
# PARAMETERS
# =============================================================================

DATASET_ROOT = "/.../datasets"
LIBERO_SUITE = "libero_spatial"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modality dims (CLIP ViT-B/32)
CAM0_DIM  = 512
CAM1_DIM  = 512
LANG_DIM  = 512
N_MODS    = 3
MOD_NAMES = ["agent_cam", "inhand_cam", "language"]
MOD_DIMS  = [CAM0_DIM, CAM1_DIM, LANG_DIM]

NUM_FRAMES = 8       # frames sampled per demo for CLIP vision

VAL_FRAC   = 0.15
SEED       = 42
VARIANT    = "with"  # 'with' | 'just_scaling' | 'without'

# ── RL / BFS ─────────────────────────────────────────────────────────────────
STRATEGY      = "knn"
K_APPROX      = 20
K_SEED_KNN    = 5
K_GRAPH       = 5
MAX_BFS_DEPTH = 6
MAX_NODE2_RL  = 200

PPO_EPOCHS     = 30
PPO_CLIP       = 0.2
LR             = 3e-4
ENT_COEF       = 0.0001
MINIBATCH_SIZE = 128
RL_CKPT        = f"best_rl_libero_{VARIANT}_{STRATEGY}.pt"

# ── Prediction Fusion Head (soft, always used — no hard argmax) ──────────────
FUSION_TOPK    = 32
FUSION_HIDDEN  = 128
FUSION_HEADS   = 4
FUSION_EPOCHS  = 30
FUSION_LR      = 3e-4
FUSION_BATCH   = 32
FUSION_CKPT    = f"best_fusion_libero_{VARIANT}_{STRATEGY}.pt"

# ── Imputation RL + Soft Imputation Head ─────────────────────────────────────
IMP_PPO_EPOCHS      = 50
IMP_MINIBATCH_SIZE  = 64
IMP_LR              = 3e-4
IMP_ENT_COEF        = 0.0001
IMP_CKPT_PREFIX     = f"best_imp_libero_{VARIANT}_{STRATEGY}"

IMP_SOFT_TOPK       = 32
IMP_SOFT_HIDDEN     = 64
IMP_SOFT_HEADS      = 2
IMP_SOFT_EPOCHS     = 30
IMP_SOFT_LR         = 3e-4
IMP_SOFT_BATCH      = 32
IMP_SOFT_CKPT_PREFIX = f"best_soft_imp_libero_{VARIANT}_{STRATEGY}"

# ── Skip training and load saved checkpoints (use after robosuite fix) ────────
# Set True to jump straight to Phase 3 rollout evaluation using saved .pt files
SKIP_TRAINING = False

# ── Ablation: set False to use hard RL argmax instead of soft fusion head ─────
# When False: prediction uses best RL-scored candidate directly (no FusionHead)
#             imputation uses hard argmax donor (no SoftImputationHead)
# When True : both use soft cross-attention heads (full RL4IL method)
USE_SOFT_FUSION = True

# ── Rollout evaluation ────────────────────────────────────────────────────────
N_ROLLOUT_SEEDS = 3
N_ROLLOUTS_TASK = 25
MAX_STEPS       = 260

# Evaluation configs: (label, cam0_present, cam1_present)
# Language is ALWAYS present — never dropped.
EVAL_CONFIGS = [
    ("none",   True,  True),   # Table 1
    ("mask_0", False, True),   # Table 3/4: agent cam missing
    ("mask_1", True,  False),  # Table 3/4: in-hand cam missing
]

# =============================================================================
# FROZEN CLIP ENCODER
# =============================================================================

class FrozenCLIPEncoder(nn.Module):
    """Single CLIP ViT-B/32 model for both vision and text."""
    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, frames_list):
        """frames_list: list of B items, each a list of HxWx3 uint8 arrays.
        Returns (B, 512) on DEVICE."""
        device = next(self.model.parameters()).device
        all_feats = []
        for frames in frames_list:
            if len(frames) == 0:
                all_feats.append(torch.zeros(512, device=device))
                continue
            if len(frames) >= NUM_FRAMES:
                idxs = np.linspace(0, len(frames)-1, NUM_FRAMES, dtype=int)
            else:
                repeat = (NUM_FRAMES + len(frames) - 1) // len(frames)
                idxs = np.tile(np.arange(len(frames)), repeat)[:NUM_FRAMES]
            pil_frames = [Image.fromarray(frames[i].astype("uint8")) for i in idxs]
            inputs = self.processor(images=pil_frames, return_tensors="pt",
                                    padding=True).to(device)
            out = self.model.get_image_features(
                pixel_values=inputs["pixel_values"])   # (NUM_FRAMES, 512)
            all_feats.append(out.mean(dim=0))
        return torch.stack(all_feats)   # (B, 512)

    @torch.no_grad()
    def encode_text(self, texts_list):
        """texts_list: list of B strings. Returns (B, 512) on DEVICE."""
        device = next(self.model.parameters()).device
        inputs = self.processor(text=texts_list, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=77).to(device)
        return self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"])


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_task_demos(hdf5_path, task_lang, rng):
    """Load all demos for one task. Returns list of demo dicts."""
    demos = []
    with h5py.File(hdf5_path, "r") as f:
        for dk in sorted(f["data"].keys()):
            demo_grp = f["data"][dk]
            actions  = demo_grp["actions"][:]
            obs_grp  = demo_grp["obs"]

            # agent camera — try all known key names
            cam0 = None
            for key in ("agentview_rgb", "agentview_image", "agentview",
                        "frontview_rgb", "frontview_image", "frontview"):
                if key in obs_grp:
                    cam0 = obs_grp[key][:]
                    break
            if cam0 is None:
                # last resort: any key containing 'agent' or 'front'
                for key in obs_grp.keys():
                    if "agent" in key.lower() or "front" in key.lower():
                        cam0 = obs_grp[key][:]
                        break
            if cam0 is None:
                cam0 = np.zeros((1, 128, 128, 3), dtype=np.uint8)

            # in-hand camera
            cam1 = None
            for key in ("robot0_eye_in_hand_rgb", "eye_in_hand_image",
                        "robot0_eye_in_hand", "wrist_rgb", "wrist_image"):
                if key in obs_grp:
                    cam1 = obs_grp[key][:]
                    break
            if cam1 is None:
                for key in obs_grp.keys():
                    if "hand" in key.lower() or "wrist" in key.lower():
                        cam1 = obs_grp[key][:]
                        break
            if cam1 is None:
                cam1 = np.zeros((1, 128, 128, 3), dtype=np.uint8)

            demos.append({
                "demo_key":    dk,
                "task_lang":   task_lang,
                "cam0_frames": [cam0[t] for t in range(len(cam0))],
                "cam1_frames": [cam1[t] for t in range(len(cam1))],
                "actions":     actions.astype(np.float32),
                "label":       1,
            })
    rng.shuffle(demos)
    return demos


def _find_hdf5(dataset_root, suite_name, task_name):
    """
    Robustly locate the HDF5 file for a task.

    Libero stores files as:
        <suite_name>/<task_name>_demo.hdf5   (most common)
        <suite_name>/<task_name>.hdf5
        <suite_name>/demo_<task_name>.hdf5
        <suite_name>/<task_name>/demo.hdf5   (nested)

    If none of those match, scan the directory for any .hdf5 file whose
    stem contains the task_name as a substring (case-insensitive), or
    whose task_name is a substring of its stem — whichever is best.
    Returns the path string or None.
    """
    suite_dir = os.path.join(dataset_root, suite_name)

    # 1. Explicit candidates
    explicit = [
        os.path.join(suite_dir, f"{task_name}_demo.hdf5"),
        os.path.join(suite_dir, f"{task_name}.hdf5"),
        os.path.join(suite_dir, f"demo_{task_name}.hdf5"),
        os.path.join(suite_dir, task_name, "demo.hdf5"),
    ]
    for p in explicit:
        if os.path.exists(p):
            return p

    # 2. Scan directory — fuzzy match on stem
    if not os.path.isdir(suite_dir):
        return None
    all_hdf5 = [f for f in os.listdir(suite_dir) if f.endswith(".hdf5")]

    # exact substring match (task_name inside filename stem)
    tn_lower = task_name.lower()
    for fname in all_hdf5:
        stem = fname.replace(".hdf5", "").replace("_demo", "").lower()
        if tn_lower == stem or tn_lower in stem or stem in tn_lower:
            return os.path.join(suite_dir, fname)

    # 3. Word-overlap score as last resort
    tn_words = set(tn_lower.split("_"))
    best_score, best_path = 0, None
    for fname in all_hdf5:
        stem  = fname.replace(".hdf5","").replace("_demo","").lower()
        words = set(stem.split("_"))
        score = len(tn_words & words)
        if score > best_score:
            best_score = score
            best_path  = os.path.join(suite_dir, fname)

    # only accept if more than half the task words matched
    if best_score >= max(1, len(tn_words) // 2):
        return best_path
    return None


def load_libero_spatial(dataset_root, suite_name,
                        val_frac=VAL_FRAC, seed=SEED):
    benchmark_dict = get_benchmark_dict()
    task_suite     = benchmark_dict[suite_name]()
    n_tasks        = task_suite.n_tasks
    print(f"\nLoading {suite_name}: {n_tasks} tasks …")

    # show what HDF5 files are actually present so user can debug naming
    suite_dir = os.path.join(dataset_root, suite_name)
    if os.path.isdir(suite_dir):
        found = [f for f in os.listdir(suite_dir) if f.endswith(".hdf5")]
        print(f"  HDF5 files found in {suite_dir}:")
        for f in sorted(found):
            print(f"    {f}")
        # print obs keys from the first file found so user can verify
        if found:
            try:
                with h5py.File(os.path.join(suite_dir, found[0]), "r") as f:
                    demo_keys = list(f["data"].keys())
                    if demo_keys:
                        obs_keys = list(f["data"][demo_keys[0]]["obs"].keys())
                        print(f"  Obs keys in first file: {obs_keys}")
            except Exception:
                pass
    else:
        print(f"  [WARN] Suite directory not found: {suite_dir}")

    rng = np.random.RandomState(seed)
    all_train, all_val, all_test = [], [], []

    for task_id in range(n_tasks):
        task      = task_suite.get_task(task_id)
        task_lang = task.language

        hdf5_path = _find_hdf5(dataset_root, suite_name, task.name)
        if hdf5_path is None:
            print(f"  [WARN] HDF5 not found for task: {task.name}")
            continue

        demos   = _load_task_demos(hdf5_path, task_lang, rng)
        n       = len(demos)
        n_val   = max(1, int(n * val_frac))
        val_d   = demos[:n_val]
        train_d = demos[n_val:]
        test_d  = val_d

        all_train.extend(train_d)
        all_val.extend(val_d)
        all_test.extend(test_d)
        print(f"  Task {task_id:02d}  {task.name:<60s}  "
              f"train={len(train_d):>3d}  val={len(val_d):>3d}  "
              f"[{os.path.basename(hdf5_path)}]")

    print(f"  Total — train={len(all_train)}  val={len(all_val)}  "
          f"test={len(all_test)}")

    if len(all_train) == 0:
        raise RuntimeError(
            f"\nNo demos loaded! Check that the dataset path is correct:\n"
            f"  DATASET_ROOT = {dataset_root}\n"
            f"  Suite dir    = {suite_dir}\n"
            f"Files present: {os.listdir(suite_dir) if os.path.isdir(suite_dir) else 'directory missing'}\n"
            f"Task names expected by Libero:\n"
            + "\n".join(f"  {task_suite.get_task(i).name}"
                        for i in range(n_tasks))
        )
    print()
    return all_train, all_val, all_test, task_suite


# =============================================================================
# EMBEDDING EXTRACTION & NORMALISATION
# =============================================================================

@torch.no_grad()
def extract_embeddings(demos, clip_enc, batch_size=8):
    """Returns cam0_emb(N,512), cam1_emb(N,512), lang_emb(N,512), labels(N,)."""
    N = len(demos)
    c0l, c1l, ltl, lbl = [], [], [], []
    nb = (N + batch_size - 1) // batch_size
    for b in range(nb):
        print(f"\r    batch {b+1}/{nb}", end="", flush=True)
        batch = demos[b*batch_size:(b+1)*batch_size]
        z0 = clip_enc.encode_image([d["cam0_frames"] for d in batch]).cpu().numpy()
        z1 = clip_enc.encode_image([d["cam1_frames"] for d in batch]).cpu().numpy()
        zt = clip_enc.encode_text( [d["task_lang"]   for d in batch]).cpu().numpy()
        c0l.append(z0); c1l.append(z1); ltl.append(zt)
        lbl.extend([d["label"] for d in batch])
    print()
    return (np.concatenate(c0l).astype(np.float32),
            np.concatenate(c1l).astype(np.float32),
            np.concatenate(ltl).astype(np.float32),
            np.array(lbl, dtype=np.float32))


def compute_norm_stats(c0, c1, lt):
    """Per-modality z-score stats from training set (numpy arrays)."""
    stats = {}
    for m, Z in enumerate([c0, c1, lt]):
        mu    = Z.mean(axis=0)
        sigma = Z.std(axis=0, ddof=1)
        sigma = np.where(sigma < 1e-8, np.ones_like(sigma), sigma)
        stats[m] = (mu, sigma)
    return stats


def build_partial_embedding(raw_mods, present_mask, stats, variant):
    """
    Normalise and concatenate modality blocks.
    Missing blocks are zero-filled here ONLY as a placeholder — at
    inference time missing blocks are ALWAYS replaced by the soft
    imputation head output before this function is called.
    Mirrors RL-4.txt modality-fair normalisation with M' present mods.
    """
    M_pres = max(1, int(present_mask.sum()))
    parts  = []
    for m, (Z_row, dm) in enumerate(zip(raw_mods, MOD_DIMS)):
        Z = Z_row.copy().astype(np.float32)
        if not present_mask[m]:
            parts.append(np.zeros(dm, dtype=np.float32))
            continue
        mu, sigma = stats[m]
        if variant == "with":
            Z = (Z - mu) / sigma
            Z = Z / (dm * M_pres) ** 0.5
        elif variant == "just_scaling":
            Z = Z / (dm * M_pres) ** 0.5
        parts.append(Z)
    return np.concatenate(parts).astype(np.float32)


def _build_full_emb(raw_list, stats):
    N = raw_list[0].shape[0]
    full_mask = np.ones(N_MODS, dtype=bool)
    return np.stack([
        build_partial_embedding([raw_list[m][i] for m in range(N_MODS)],
                                full_mask, stats, VARIANT)
        for i in range(N)
    ]).astype(np.float32)


def build_all_embeddings(clip_enc, train_demos, val_demos, test_demos):
    print("Extracting embeddings …")
    print("  TRAIN …")
    tr_c0,tr_c1,tr_lt,tr_labels = extract_embeddings(train_demos, clip_enc)
    print("  VAL   …")
    va_c0,va_c1,va_lt,va_labels = extract_embeddings(val_demos,   clip_enc)
    print("  TEST  …")
    te_c0,te_c1,te_lt,te_labels = extract_embeddings(test_demos,  clip_enc)

    stats  = compute_norm_stats(tr_c0, tr_c1, tr_lt)
    tr_raw = [tr_c0, tr_c1, tr_lt]
    va_raw = [va_c0, va_c1, va_lt]
    te_raw = [te_c0, te_c1, te_lt]

    print("  Building full embeddings …")
    tr_emb = _build_full_emb(tr_raw, stats)
    va_emb = _build_full_emb(va_raw, stats)
    te_emb = _build_full_emb(te_raw, stats)

    print(f"  Embedding dim: {tr_emb.shape[1]}  (variant='{VARIANT}')\n")
    return (tr_emb, tr_labels, va_emb, va_labels, te_emb, te_labels,
            tr_raw, va_raw, te_raw, stats)


# =============================================================================
# RL / BFS CORE  (mirrors mosi pipeline exactly)
# =============================================================================

def _build_knn_graph(emb, k=K_GRAPH):
    N = emb.shape[0]; kq = min(k+1, N)
    idx = NearestNeighbors(n_neighbors=kq, algorithm="auto",
                           metric="euclidean", n_jobs=-1).fit(emb)
    dists, idxs = idx.kneighbors(emb)
    adj = [[] for _ in range(N)]
    for i in range(N):
        for p in range(kq):
            j = idxs[i,p]; d = dists[i,p]
            if j != i and d > 1e-12:
                adj[i].append((j, float(d)))
    return adj


def _build_knn_index(emb, k=K_APPROX):
    kq = min(k+1, emb.shape[0])
    idx = NearestNeighbors(n_neighbors=kq, algorithm="auto",
                           metric="euclidean", n_jobs=-1).fit(emb)
    return idx


def get_seeds_knn(knn_idxs, knn_dists, k=K_SEED_KNN):
    return [(int(knn_idxs[i]), float(knn_dists[i]))
            for i in range(min(k, len(knn_idxs)))]


def get_seeds_unn(q_vec, tr_emb, knn_idxs, knn_dists):
    nb_vecs = tr_emb[knn_idxs]
    order   = np.argsort(knn_dists)
    useful_idx, useful_vecs = [], []
    for pos in order:
        d_ij   = knn_dists[pos]; ni_vec = nb_vecs[pos]
        if not any(np.linalg.norm(jv - ni_vec) < d_ij for jv in useful_vecs):
            useful_idx.append(pos); useful_vecs.append(ni_vec)
    return [(int(knn_idxs[p]), float(knn_dists[p])) for p in useful_idx]


def _get_seeds(q_vec, tr_emb, knn_idxs, knn_dists):
    return (get_seeds_unn(q_vec, tr_emb, knn_idxs, knn_dists)
            if STRATEGY == "unn" else get_seeds_knn(knn_idxs, knn_dists))


def _bfs(seeds, adj, target_sz=None,
         max_depth=MAX_BFS_DEPTH, max_nodes=MAX_NODE2_RL):
    visited = {}; heap = []
    for idx, dist in seeds:
        if idx not in visited:
            visited[idx] = (dist, 0)
            heapq.heappush(heap, (dist, idx, 0))
    sequence = []
    while heap:
        g_dist, node, depth = heapq.heappop(heap)
        if g_dist > visited.get(node,(float("inf"),))[0] + 1e-9:
            continue
        sequence.append((node, g_dist, depth))
        if target_sz is not None and len(sequence) >= target_sz:
            break
        for nb, ew in adj[node]:
            nd = g_dist + ew
            if nd < visited.get(nb,(float("inf"),))[0]:
                visited[nb] = (nd, depth+1)
                heapq.heappush(heap, (nd, nb, depth+1))
        if target_sz is None and depth >= max_depth and len(sequence) >= max_nodes:
            break
    return sequence


def build_candidate_set(seeds, target_label, tr_labels, adj, rng=None):
    sequence = _bfs(seeds, adj)
    if not sequence:
        return None, None
    oracle_pos = min(range(len(sequence)),
                     key=lambda p: abs(float(tr_labels[sequence[p][0]])
                                       - float(target_label)))
    if rng is None:
        rng = np.random.RandomState()
    perm       = rng.permutation(len(sequence))
    sequence   = [sequence[p] for p in perm]
    oracle_pos = int(np.where(perm == oracle_pos)[0][0])
    return sequence, oracle_pos


def build_all_sets(emb, labels, adj):
    N = emb.shape[0]; kq = min(K_APPROX+1, N)
    index = NearestNeighbors(n_neighbors=kq, algorithm="auto",
                             metric="euclidean", n_jobs=-1).fit(emb)
    knn_dists_all, knn_idxs_all = index.kneighbors(emb)
    all_sets, all_oracles = [], []
    set_sizes = np.zeros(N, dtype=np.float32)
    rng = np.random.RandomState(SEED)
    for i in range(N):
        mask  = knn_dists_all[i] > 1e-12
        idxs  = knn_idxs_all[i][mask]; dists = knn_dists_all[i][mask]
        seeds = _get_seeds(emb[i], emb, idxs, dists)
        if not seeds:
            all_sets.append(None); all_oracles.append(None); continue
        cset, oracle = build_candidate_set(seeds, labels[i], labels, adj, rng=rng)
        all_sets.append(cset); all_oracles.append(oracle)
        if cset is not None:
            set_sizes[i] = float(len(cset))
    v = set_sizes[set_sizes > 0]
    print(f"  Sets: {sum(s is not None for s in all_sets)}/{N}"
          f"  | ≥2: {sum(s is not None and len(s)>=2 for s in all_sets)}/{N}"
          f"  | mean size: {v.mean():.1f}")
    return all_sets, all_oracles, set_sizes


def build_features(q_vec, candidate_set, tr_emb, tr_labels):
    """State (D+2) and candidate features (D+4) — from RL-4.txt."""
    n = len(candidate_set)
    cand_labels = np.array([float(tr_labels[idx]) for idx,_,_ in candidate_set])
    mean_lbl  = cand_labels.mean(); var_lbl = cand_labels.var()
    max_depth = max(dep for _,_,dep in candidate_set) + 1e-9
    max_dist  = max(d   for _,d,_   in candidate_set) + 1e-9
    rank_map  = {orig: r for r,(orig,_) in
                 enumerate(sorted(enumerate(candidate_set), key=lambda x:x[1][1]))}
    state = np.concatenate([q_vec,[var_lbl],[float(n)]]).astype(np.float32)
    cand_feats = []
    for pos,(idx,dist,depth) in enumerate(candidate_set):
        cand_feats.append(np.concatenate([
            tr_emb[idx],
            [dist/max_dist], [depth/max_depth],
            [rank_map[pos]/max(n-1,1)],
            [float(tr_labels[idx])-mean_lbl],
        ]).astype(np.float32))
    return state, np.stack(cand_feats)


# =============================================================================
# SCORING MLP  (shared by prediction RL and imputation RL)
# =============================================================================

class ScoringMLP(nn.Module):
    def __init__(self, state_dim, cand_dim, hidden=256):
        super().__init__()
        self.qe = nn.Sequential(nn.Linear(state_dim,hidden),nn.ReLU(),
                                nn.Linear(hidden,hidden))
        self.ce = nn.Sequential(nn.Linear(cand_dim, hidden),nn.ReLU(),
                                nn.Linear(hidden,hidden))
        self.sh = nn.Sequential(nn.Linear(3*hidden,hidden),nn.ReLU(),
                                nn.Linear(hidden,1))
    def forward(self, state, cand_feats):
        sq = state.dim() == 1
        if sq: state=state.unsqueeze(0); cand_feats=cand_feats.unsqueeze(0)
        B,K,_ = cand_feats.shape
        h_o   = self.qe(state)
        h_oe  = h_o.unsqueeze(1).expand(-1,K,-1)
        h_i   = self.ce(cand_feats)
        s     = self.sh(torch.cat([h_oe,h_i,h_oe*h_i],dim=-1)).squeeze(-1)
        return s.squeeze(0) if sq else s


# =============================================================================
# PPO UPDATE
# =============================================================================

def ppo_update(policy, optimizer, rollout, clip=PPO_CLIP, ent_coef=ENT_COEF):
    states,cands_l,actions,old_lps,advantages = rollout
    losses = []
    for i in range(len(states)):
        scores  = policy(states[i], cands_l[i])
        log_pi  = F.log_softmax(scores, dim=-1)
        pi      = log_pi.exp()
        ratio   = torch.exp(log_pi[actions[i]] - old_lps[i].detach())
        adv     = advantages[i].detach()
        entropy = -(pi * log_pi).sum()
        losses.append(-torch.min(ratio*adv,
                                 torch.clamp(ratio,1-clip,1+clip)*adv)
                      - ent_coef*entropy)
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return loss.item()


# =============================================================================
# PREDICTION FUSION HEAD  (soft cross-attention — always used, no hard argmax)
# =============================================================================

class FusionHead(nn.Module):
    """
    Attends over top-K' RL-ranked candidates.
    Query → K projection, weighted label sum + MLP refinement.
    Outputs a scalar prediction (binary success probability here).
    """
    def __init__(self, emb_dim, hidden=FUSION_HIDDEN,
                 n_heads=FUSION_HEADS, topk=FUSION_TOPK):
        super().__init__()
        self.topk   = topk
        self.n_heads = n_heads
        self.d_head  = hidden // n_heads
        assert hidden % n_heads == 0
        self.q_proj  = nn.Linear(emb_dim, hidden, bias=False)
        self.k_proj  = nn.Linear(emb_dim, hidden, bias=False)
        self.scale   = self.d_head ** -0.5
        self.ctx_proj = nn.Linear(hidden, hidden)
        self.refine  = nn.Sequential(nn.Linear(hidden+1,hidden),nn.ReLU(),
                                     nn.Linear(hidden,1))

    def forward(self, q_emb, cand_embs, cand_labels):
        K=cand_embs.shape[0]; H=self.n_heads; Dh=self.d_head
        q = self.q_proj(q_emb).view(H,Dh)
        k = self.k_proj(cand_embs).view(K,H,Dh)
        w = F.softmax(torch.einsum("hd,khd->hk",q,k)*self.scale,
                      dim=-1).mean(0)                       # (K,)
        att_lbl = (w * cand_labels).sum().unsqueeze(0)      # (1,)
        k_flat  = k.reshape(K,H*Dh)
        ctx     = F.relu(self.ctx_proj((w.unsqueeze(-1)*k_flat).sum(0)))
        return self.refine(torch.cat([ctx,att_lbl],-1)).squeeze(-1)


@torch.no_grad()
def _topk_from_sequence(q_vec, sequence, tr_emb, tr_labels, policy, topk):
    """Run RL policy over sequence, return top-topk demo indices + arrays."""
    if not sequence:
        return None, None, None
    state, cfs = build_features(q_vec, sequence, tr_emb, tr_labels)
    scores = policy(torch.tensor(state,device=DEVICE),
                    torch.tensor(cfs,  device=DEVICE))
    top = scores.argsort(descending=True)[:topk].cpu().numpy()
    embs   = np.stack([tr_emb[sequence[j][0]]          for j in top])
    labels = np.array([float(tr_labels[sequence[j][0]]) for j in top])
    didxs  = [sequence[j][0] for j in top]
    return embs, labels, didxs


def _infer_one_fusion(q_vec, knn_idxs, knn_dists,
                      tr_emb, tr_labels, adj, tr_set_sizes, policy, head):
    """
    Full soft inference: BFS → RL scoring → FusionHead attended prediction.
    Returns (best_train_demo_idx, predicted_score).
    """
    mask  = knn_dists > 1e-12
    idxs  = knn_idxs[mask]; dists = knn_dists[mask]
    if len(idxs) == 0:
        return 0, float(tr_labels[0])
    seeds = _get_seeds(q_vec, tr_emb, idxs, dists)
    if not seeds:
        return int(idxs[0]), float(tr_labels[idxs[0]])

    target_sz = max(1, int(max(float(tr_set_sizes[idx]) for idx,_ in seeds)))
    sequence  = _bfs(seeds, adj, target_sz=target_sz)
    if not sequence:
        return int(idxs[0]), float(tr_labels[idxs[0]])

    # hard argmax path (ablation: USE_SOFT_FUSION=False)
    if not USE_SOFT_FUSION or head is None:
        state, cfs = build_features(q_vec, sequence, tr_emb, tr_labels)
        sc   = policy(torch.tensor(state, device=DEVICE),
                      torch.tensor(cfs,   device=DEVICE))
        best = int(sc.argmax().item())
        return sequence[best][0], float(tr_labels[sequence[best][0]])

    # soft fusion path (full RL4IL, USE_SOFT_FUSION=True)
    ce, cl, didxs = _topk_from_sequence(
        q_vec, sequence, tr_emb, tr_labels, policy, FUSION_TOPK)
    if ce is None:
        return sequence[0][0], float(tr_labels[sequence[0][0]])

    q_t  = torch.tensor(q_vec, dtype=torch.float32, device=DEVICE)
    ce_t = torch.tensor(ce,    dtype=torch.float32, device=DEVICE)
    cl_t = torch.tensor(cl,    dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        score = head(q_t, ce_t, cl_t).item()
    return didxs[0], score


def _eval_fusion_acc(head, policy, emb, labels, tr_emb, tr_labels, adj, sizes):
    head.eval()
    idx = _build_knn_index(tr_emb)
    kd, ki = idx.kneighbors(emb)
    preds = []
    with torch.no_grad():
        for qi in range(len(labels)):
            _, s = _infer_one_fusion(emb[qi],ki[qi],kd[qi],
                                     tr_emb,tr_labels,adj,sizes,policy,head)
            preds.append(s)
    preds = np.array(preds, dtype=np.float32)
    return float(((preds >= 0.5).astype(int) == np.round(labels).astype(int)).mean())


def train_policy(tr_emb, tr_labels, tr_sets, tr_oracles,
                 va_emb, va_labels, adj, set_sizes):
    D = tr_emb.shape[1]
    policy    = ScoringMLP(D+2, D+4).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    best_acc, best_state = -1.0, None

    valid_tr = [i for i in range(len(tr_emb))
                if tr_sets[i] is not None and len(tr_sets[i])>=2
                and tr_oracles[i] is not None]
    print(f"  Valid training points: {len(valid_tr)}")
    print(f"Training PPO ({PPO_EPOCHS} epochs) …")

    for epoch in range(PPO_EPOCHS):
        ep_rng = np.random.RandomState(SEED+epoch)
        # reshuffle candidate sets
        epoch_sets,epoch_oracles = {},{}
        for i in valid_tr:
            cset = tr_sets[i]; perm = ep_rng.permutation(len(cset))
            nc   = [cset[p] for p in perm]
            onode = cset[tr_oracles[i]][0]
            epoch_sets[i]    = nc
            epoch_oracles[i] = int(np.where(
                np.array([n for n,_,_ in nc])==onode)[0][0])
        ep_rng.shuffle(valid_tr)

        # collect rollout
        states,cands_l,actions,old_lps,advantages = [],[],[],[],[]
        policy.eval()
        with torch.no_grad():
            for i in valid_tr:
                cset=epoch_sets[i]; oracle=epoch_oracles[i]
                state,cfs = build_features(tr_emb[i],cset,tr_emb,tr_labels)
                s_t=torch.tensor(state,device=DEVICE)
                cf_t=torch.tensor(cfs,device=DEVICE)
                scores=policy(s_t,cf_t)
                log_pi=F.log_softmax(scores,dim=-1)
                action=int(torch.multinomial(log_pi.exp(),1).item())
                old_lp=log_pi[action]
                K=len(cset); y_s=float(tr_labels[i])
                if K<=1: reward=0.0
                else:
                    errs=[abs(float(tr_labels[cset[j][0]])-y_s) for j in range(K)]
                    si=np.argsort(errs); rm={int(si[r]):r for r in range(K)}
                    reward=float(rm[oracle]-rm[action])/(K-1)
                states.append(s_t); cands_l.append(cf_t)
                actions.append(action); old_lps.append(old_lp.detach())
                advantages.append(torch.tensor(reward,dtype=torch.float32,device=DEVICE))

        # PPO updates
        policy.train(); M=len(states); perm=ep_rng.permutation(M)
        ep_loss,nb = 0.0,0
        for start in range(0,M,MINIBATCH_SIZE):
            mb=perm[start:start+MINIBATCH_SIZE]
            ep_loss+=ppo_update(policy,optimizer,
                ([states[j] for j in mb],[cands_l[j] for j in mb],
                 [actions[j] for j in mb],[old_lps[j] for j in mb],
                 [advantages[j] for j in mb])); nb+=1

        # val: build a temporary fusion head for evaluation is expensive;
        # instead evaluate RL argmax accuracy as proxy
        policy.eval()
        idx = _build_knn_index(tr_emb)
        kd,ki = idx.kneighbors(va_emb)
        preds=[]
        with torch.no_grad():
            for qi in range(len(va_labels)):
                mask=kd[qi]>1e-12; ids=ki[qi][mask]; ds=kd[qi][mask]
                if not len(ids): preds.append(0.5); continue
                seeds=_get_seeds(va_emb[qi],tr_emb,ids,ds)
                if not seeds: preds.append(float(tr_labels[ids[0]])); continue
                tsz=max(1,int(max(float(set_sizes[ix]) for ix,_ in seeds)))
                seq=_bfs(seeds,adj,target_sz=tsz)
                if not seq: preds.append(float(tr_labels[ids[0]])); continue
                state,cfs=build_features(va_emb[qi],seq,tr_emb,tr_labels)
                sc=policy(torch.tensor(state,device=DEVICE),
                          torch.tensor(cfs,device=DEVICE))
                best=int(sc.argmax().item())
                preds.append(float(tr_labels[seq[best][0]]))
        val_acc=float(((np.array(preds)>=0.5).astype(int)==
                        np.round(va_labels).astype(int)).mean())
        marker=" ← best" if val_acc>best_acc else ""
        print(f"  Epoch {epoch+1:>3d}/{PPO_EPOCHS}  "
              f"loss={ep_loss/max(nb,1):.4f}  val_acc={val_acc:.4f}{marker}")
        if val_acc>best_acc:
            best_acc=val_acc; best_state=copy.deepcopy(policy.state_dict())
            torch.save(best_state, RL_CKPT)

    policy.load_state_dict(best_state)
    print(f"  Best val acc: {best_acc:.4f}  →  {RL_CKPT}\n")
    return policy


def train_fusion(policy, tr_emb, tr_labels, tr_sets,
                 va_emb, va_labels, adj, set_sizes):
    print("=" * 60)
    print("PREDICTION FUSION HEAD training")
    print("=" * 60)
    D    = tr_emb.shape[1]
    head = FusionHead(emb_dim=D).to(DEVICE)
    opt  = torch.optim.Adam(head.parameters(), lr=FUSION_LR)
    policy.eval()
    valid       = [i for i in range(len(tr_emb))
                   if tr_sets[i] is not None and len(tr_sets[i])>=1]
    best_acc, best_state = -1.0, None

    for epoch in range(FUSION_EPOCHS):
        head.train()
        rng  = np.random.RandomState(SEED+1000+epoch)
        perm = rng.permutation(len(valid))
        ep_loss,nb = 0.0,0
        for start in range(0,len(valid),FUSION_BATCH):
            mb=[valid[perm[j]]
                for j in range(start,min(start+FUSION_BATCH,len(valid)))]
            bl=[]
            for i in mb:
                ce,cl,_ = _topk_from_sequence(
                    tr_emb[i],tr_sets[i],tr_emb,tr_labels,policy,FUSION_TOPK)
                if ce is None: continue
                q_t  = torch.tensor(tr_emb[i],dtype=torch.float32,device=DEVICE)
                ce_t = torch.tensor(ce,        dtype=torch.float32,device=DEVICE)
                cl_t = torch.tensor(cl,        dtype=torch.float32,device=DEVICE)
                pred = head(q_t,ce_t,cl_t)
                tgt  = torch.tensor(float(tr_labels[i]),dtype=torch.float32,device=DEVICE)
                bl.append(F.mse_loss(pred,tgt))
            if not bl: continue
            loss=torch.stack(bl).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.0)
            opt.step(); ep_loss+=loss.item(); nb+=1

        val_acc=_eval_fusion_acc(head,policy,va_emb,va_labels,tr_emb,tr_labels,adj,set_sizes)
        marker=" ← best" if val_acc>best_acc else ""
        print(f"  Epoch {epoch+1:>3d}/{FUSION_EPOCHS}  "
              f"loss={ep_loss/max(nb,1):.4f}  val_acc={val_acc:.4f}{marker}")
        if val_acc>best_acc:
            best_acc=val_acc; best_state=copy.deepcopy(head.state_dict())
            torch.save(best_state, FUSION_CKPT)

    head.load_state_dict(best_state)
    print(f"  Best val acc: {best_acc:.4f}  →  {FUSION_CKPT}\n")
    return head


# =============================================================================
# IMPUTATION RL + SOFT IMPUTATION HEAD  (from RL-4.txt section)
# =============================================================================

class SoftImputationHead(nn.Module):
    """
    Cross-attention over top-K' RL-ranked donors.
    Projects partial query + donor embeddings → weighted sum → MLP → d_m output.
    Exactly as described in RL-4.txt "Soft Imputation via Attended Donor Aggregation".
    """
    def __init__(self, d_full, d_m, hidden=IMP_SOFT_HIDDEN, n_heads=IMP_SOFT_HEADS):
        super().__init__()
        self.d_m=d_m; self.n_heads=n_heads; self.d_head=hidden//n_heads
        assert hidden%n_heads==0
        self.q_proj = nn.Linear(d_full, hidden, bias=False)
        self.k_proj = nn.Linear(d_m,    hidden, bias=False)
        self.scale  = self.d_head**-0.5
        self.refine = nn.Sequential(nn.Linear(d_m+hidden,hidden),nn.ReLU(),
                                    nn.Linear(hidden,d_m))

    def forward(self, q_partial, donor_embs):
        """q_partial: (D_full,), donor_embs: (K',d_m) → (d_m,)"""
        K=donor_embs.shape[0]; H=self.n_heads; Dh=self.d_head
        q=self.q_proj(q_partial).view(H,Dh)
        k=self.k_proj(donor_embs).view(K,H,Dh)
        w=F.softmax(torch.einsum("hd,khd->hk",q,k)*self.scale,dim=-1).mean(0)
        attended  = (w.unsqueeze(-1)*donor_embs).sum(0)          # (d_m,)
        k_flat    = k.reshape(K,H*Dh)
        context   = F.relu((w.unsqueeze(-1)*k_flat).sum(0))      # (hidden,)
        return self.refine(torch.cat([attended,context],-1))      # (d_m,)


def _imp_bfs(seeds, valid_emb_m, adj_imp, gt_emb_m=None, rng=None):
    """BFS for imputation. Oracle = closest donor in modality-m L2 space."""
    sequence = _bfs(seeds, adj_imp)
    if not sequence: return None, None
    if gt_emb_m is not None:
        oracle_pos = min(range(len(sequence)),
            key=lambda p: float(np.sum(
                (valid_emb_m[sequence[p][0]] - gt_emb_m)**2)))
    else:
        oracle_pos = 0
    if rng is None: rng = np.random.RandomState()
    perm       = rng.permutation(len(sequence))
    sequence   = [sequence[p] for p in perm]
    oracle_pos = int(np.where(perm==oracle_pos)[0][0])
    return sequence, oracle_pos


def _imp_features(q_partial_emb, q_emb_m, candidate_set, valid_emb_m):
    """
    Imputation state and candidate features (RL-4.txt):
      state  = [q_partial_emb ; mean_cand_m ; Var(dist_m) ; |B|]
      cand   = [donor_emb_m   ; norm_dist_m ; norm_depth  ; norm_rank]
    """
    n = len(candidate_set)
    cand_embs_m = np.stack([valid_emb_m[c[0]] for c in candidate_set])
    dists_m     = np.linalg.norm(cand_embs_m - q_emb_m[np.newaxis,:], axis=1)
    max_dist    = float(dists_m.max()) + 1e-9
    max_dep     = max(dep for _,_,dep in candidate_set) + 1e-9
    sr          = np.argsort(dists_m)
    rank_map    = {int(sr[r]):r for r in range(n)}

    state = np.concatenate([q_partial_emb, cand_embs_m.mean(0),
                             [float(dists_m.var())], [float(n)]]).astype(np.float32)
    cfs   = []
    for pos,(li,g_dist,depth) in enumerate(candidate_set):
        cfs.append(np.concatenate([
            valid_emb_m[li],
            [dists_m[pos]/max_dist],
            [depth/max_dep],
            [rank_map[pos]/max(n-1,1)],
        ]).astype(np.float32))
    return state, np.stack(cfs)


def train_imp_policy(mod_idx, tr_raw, va_raw, stats):
    """Train per-modality imputation RL policy (RL-4.txt imputation section)."""
    dm=MOD_DIMS[mod_idx]; m_name=MOD_NAMES[mod_idx]; N_tr=tr_raw[0].shape[0]
    print(f"\n{'='*60}\nIMPUTATION POLICY — {m_name}\n{'='*60}")

    tr_raw_m  = tr_raw[mod_idx]
    # All training demos have all modalities; we simulate mod as missing.
    valid_idx = np.arange(N_tr)
    valid_emb_m = tr_raw_m                          # (N_tr, d_m)
    kq = min(K_APPROX+1, N_tr)
    imp_index = NearestNeighbors(n_neighbors=kq, algorithm="auto",
                                 metric="euclidean", n_jobs=-1).fit(valid_emb_m)
    # BFS adj over donor embeddings
    adj_imp = _build_knn_graph(valid_emb_m, k=K_GRAPH)

    D_full    = sum(MOD_DIMS)
    policy    = ScoringMLP(D_full+dm+2, dm+3).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=IMP_LR)
    train_items = list(range(N_tr))
    best_l2, best_state = float("inf"), None

    for epoch in range(IMP_PPO_EPOCHS):
        ep_rng = np.random.RandomState(SEED+200+mod_idx*100+epoch)
        ep_rng.shuffle(train_items)
        policy.eval()
        states,cands_l,actions,old_lps,advantages = [],[],[],[],[]

        for i in train_items:
            pm = np.ones(N_MODS, dtype=bool); pm[mod_idx]=False
            q_partial = build_partial_embedding(
                [tr_raw[m][i] for m in range(N_MODS)], pm, stats, VARIANT)
            gt_emb_m = tr_raw_m[i]

            _,knn_loc = imp_index.kneighbors(gt_emb_m.reshape(1,-1))
            knn_loc   = knn_loc[0]
            knn_d_m   = np.linalg.norm(valid_emb_m[knn_loc]-gt_emb_m, axis=1)
            seeds = _get_seeds(gt_emb_m, valid_emb_m, knn_loc, knn_d_m)
            if not seeds: continue

            cset,oracle = _imp_bfs(seeds, valid_emb_m, adj_imp, gt_emb_m, ep_rng)
            if cset is None or len(cset)<2: continue

            state,cfs = _imp_features(q_partial, gt_emb_m, cset, valid_emb_m)
            s_t  = torch.tensor(state,dtype=torch.float32,device=DEVICE)
            cf_t = torch.tensor(cfs,  dtype=torch.float32,device=DEVICE)
            with torch.no_grad():
                scores=policy(s_t,cf_t)
                log_pi=F.log_softmax(scores,dim=-1)
                action=int(torch.multinomial(log_pi.exp(),1).item())
                old_lp=log_pi[action]
            K=len(cset)
            if K<=1: reward=0.0
            else:
                l2s=[float(np.sum((valid_emb_m[cset[j][0]]-gt_emb_m)**2)) for j in range(K)]
                si=np.argsort(l2s); rm={int(si[r]):r for r in range(K)}
                reward=float(rm[oracle]-rm[action])/(K-1)
            states.append(s_t); cands_l.append(cf_t)
            actions.append(action); old_lps.append(old_lp.detach())
            advantages.append(torch.tensor(reward,dtype=torch.float32,device=DEVICE))

        if not states: continue
        policy.train(); M=len(states); perm=ep_rng.permutation(M)
        ep_loss,nb=0.0,0
        for start in range(0,M,IMP_MINIBATCH_SIZE):
            mb=perm[start:start+IMP_MINIBATCH_SIZE]
            ep_loss+=ppo_update(policy,optimizer,
                ([states[j] for j in mb],[cands_l[j] for j in mb],
                 [actions[j] for j in mb],[old_lps[j] for j in mb],
                 [advantages[j] for j in mb]),ent_coef=IMP_ENT_COEF); nb+=1

        val_l2 = _eval_imp_policy(policy,mod_idx,va_raw,stats,
                                   valid_emb_m,imp_index,adj_imp)
        marker=" ← best" if val_l2<best_l2 else ""
        print(f"  Epoch {epoch+1:>3d}/{IMP_PPO_EPOCHS}  "
              f"loss={ep_loss/max(nb,1):.4f}  val_l2={val_l2:.4f}{marker}")
        if val_l2<best_l2:
            best_l2=val_l2; best_state=copy.deepcopy(policy.state_dict())
            torch.save(best_state,f"{IMP_CKPT_PREFIX}_mod{mod_idx}.pt")

    policy.load_state_dict(best_state)
    print(f"  Best val L2: {best_l2:.4f}")
    return policy, imp_index, valid_idx, valid_emb_m, adj_imp


def _eval_imp_policy(policy, mod_idx, raw, stats,
                     valid_emb_m, imp_index, adj_imp):
    errs=[]; rng=np.random.RandomState(SEED+999); policy.eval()
    for i in range(raw[0].shape[0]):
        pm=np.ones(N_MODS,dtype=bool); pm[mod_idx]=False
        q_partial=build_partial_embedding(
            [raw[m][i] for m in range(N_MODS)],pm,stats,VARIANT)
        gt_emb_m=raw[mod_idx][i]
        _,kl=imp_index.kneighbors(gt_emb_m.reshape(1,-1)); kl=kl[0]
        kd=np.linalg.norm(valid_emb_m[kl]-gt_emb_m,axis=1)
        seeds=_get_seeds(gt_emb_m,valid_emb_m,kl,kd)
        if not seeds: continue
        cset,_=_imp_bfs(seeds,valid_emb_m,adj_imp,gt_emb_m,rng)
        if not cset: continue
        state,cfs=_imp_features(q_partial,gt_emb_m,cset,valid_emb_m)
        with torch.no_grad():
            sc=policy(torch.tensor(state,device=DEVICE),
                      torch.tensor(cfs,device=DEVICE))
        best=int(sc.argmax().item())
        errs.append(float(np.sum((valid_emb_m[cset[best][0]]-gt_emb_m)**2)))
    return float(np.mean(errs)) if errs else float("inf")


def train_soft_imp_head(mod_idx, imp_policy,
                        imp_index, valid_emb_m, adj_imp,
                        tr_raw, va_raw, stats):
    """
    Train SoftImputationHead for modality mod_idx.
    Frozen imp_policy ranks donors; head learns to aggregate them via MSE.
    Matches RL-4.txt "Soft Imputation via Attended Donor Aggregation".
    """
    D_full=sum(MOD_DIMS); dm=MOD_DIMS[mod_idx]; m_name=MOD_NAMES[mod_idx]
    N_tr=tr_raw[0].shape[0]
    print(f"  [Soft Imputation Head — {m_name}]")
    head  = SoftImputationHead(D_full, dm).to(DEVICE)
    opt   = torch.optim.Adam(head.parameters(), lr=IMP_SOFT_LR)
    imp_policy.eval()
    train_items = list(range(N_tr))
    best_mse, best_state = float("inf"), None

    for epoch in range(IMP_SOFT_EPOCHS):
        head.train()
        rng  = np.random.RandomState(SEED+3000+mod_idx*100+epoch)
        perm = rng.permutation(len(train_items))
        ep_loss,nb = 0.0,0

        for start in range(0,len(train_items),IMP_SOFT_BATCH):
            mb=[train_items[perm[j]]
                for j in range(start,min(start+IMP_SOFT_BATCH,len(train_items)))]
            bl=[]
            for i in mb:
                pm=np.ones(N_MODS,dtype=bool); pm[mod_idx]=False
                q_partial=build_partial_embedding(
                    [tr_raw[m][i] for m in range(N_MODS)],pm,stats,VARIANT)
                gt_emb_m=tr_raw[mod_idx][i]
                _,kl=imp_index.kneighbors(gt_emb_m.reshape(1,-1)); kl=kl[0]
                kd=np.linalg.norm(valid_emb_m[kl]-gt_emb_m,axis=1)
                seeds=_get_seeds(gt_emb_m,valid_emb_m,kl,kd)
                if not seeds: continue
                cset,_=_imp_bfs(seeds,valid_emb_m,adj_imp,gt_emb_m,rng)
                if not cset: continue
                # get top-K' donors from frozen imp policy
                state,cfs=_imp_features(q_partial,gt_emb_m,cset,valid_emb_m)
                with torch.no_grad():
                    sc=imp_policy(torch.tensor(state,device=DEVICE),
                                  torch.tensor(cfs,device=DEVICE))
                top=sc.argsort(descending=True)[:IMP_SOFT_TOPK].cpu().numpy()
                donors=np.stack([valid_emb_m[cset[j][0]] for j in top])
                q_t = torch.tensor(q_partial,dtype=torch.float32,device=DEVICE)
                d_t = torch.tensor(donors,   dtype=torch.float32,device=DEVICE)
                pred = head(q_t, d_t)
                gt_t = torch.tensor(gt_emb_m,dtype=torch.float32,device=DEVICE)
                bl.append(F.mse_loss(pred, gt_t))
            if not bl: continue
            loss=torch.stack(bl).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.0)
            opt.step(); ep_loss+=loss.item(); nb+=1

        val_mse=_eval_soft_imp(head,imp_policy,mod_idx,
                                imp_index,valid_emb_m,adj_imp,va_raw,stats)
        marker=" ← best" if val_mse<best_mse else ""
        print(f"    Epoch {epoch+1:>3d}/{IMP_SOFT_EPOCHS}  "
              f"loss={ep_loss/max(nb,1):.4f}  val_mse={val_mse:.4f}{marker}")
        if val_mse<best_mse:
            best_mse=val_mse; best_state=copy.deepcopy(head.state_dict())
            torch.save(best_state,f"{IMP_SOFT_CKPT_PREFIX}_mod{mod_idx}.pt")

    head.load_state_dict(best_state)
    print(f"    Best val MSE: {best_mse:.4f}")
    return head


def _eval_soft_imp(head, imp_policy, mod_idx,
                   imp_index, valid_emb_m, adj_imp, raw, stats):
    head.eval(); errs=[]; rng=np.random.RandomState(SEED+4444)
    for i in range(raw[0].shape[0]):
        pm=np.ones(N_MODS,dtype=bool); pm[mod_idx]=False
        q_partial=build_partial_embedding(
            [raw[m][i] for m in range(N_MODS)],pm,stats,VARIANT)
        gt_emb_m=raw[mod_idx][i]
        _,kl=imp_index.kneighbors(gt_emb_m.reshape(1,-1)); kl=kl[0]
        kd=np.linalg.norm(valid_emb_m[kl]-gt_emb_m,axis=1)
        seeds=_get_seeds(gt_emb_m,valid_emb_m,kl,kd)
        if not seeds: continue
        cset,_=_imp_bfs(seeds,valid_emb_m,adj_imp,gt_emb_m,rng)
        if not cset: continue
        state,cfs=_imp_features(q_partial,gt_emb_m,cset,valid_emb_m)
        with torch.no_grad():
            sc=imp_policy(torch.tensor(state,device=DEVICE),
                          torch.tensor(cfs,device=DEVICE))
        top=sc.argsort(descending=True)[:IMP_SOFT_TOPK].cpu().numpy()
        donors=np.stack([valid_emb_m[cset[j][0]] for j in top])
        q_t=torch.tensor(q_partial,dtype=torch.float32,device=DEVICE)
        d_t=torch.tensor(donors,   dtype=torch.float32,device=DEVICE)
        with torch.no_grad():
            pred=head(q_t,d_t).cpu().numpy()
        errs.append(float(np.mean((pred-gt_emb_m)**2)))
    return float(np.mean(errs)) if errs else float("inf")


def impute_one_sample(raw_mods_i, present_mask, stats,
                      imp_policies, soft_heads):
    """
    For one sample: run soft imputation for each missing modality.
    Returns list of 3 numpy arrays (raw modality embeddings, imputed where missing).
    NEVER zero-fills — always uses SoftImputationHead.
    """
    row = [raw_mods_i[m].copy() for m in range(N_MODS)]

    for mod_idx, item in enumerate(imp_policies):
        if present_mask[mod_idx] or item is None:
            continue   # present or language (never imputed)
        policy, imp_index, valid_idx, valid_emb_m, adj_imp = item
        head = soft_heads[mod_idx]

        pm = present_mask.copy()
        q_partial = build_partial_embedding(row, pm, stats, VARIANT)

        # At test time the missing modality's ground truth is unavailable.
        # Use zero cold-start for BFS seeding (RL-4.txt §"test time").
        zero_m = np.zeros(MOD_DIMS[mod_idx], dtype=np.float32)
        _,kl = imp_index.kneighbors(zero_m.reshape(1,-1)); kl=kl[0]
        kd   = np.linalg.norm(valid_emb_m[kl]-zero_m, axis=1)
        seeds = _get_seeds(zero_m, valid_emb_m, kl, kd)

        if not seeds:
            # fallback: mean of training donors (rare edge case)
            row[mod_idx] = valid_emb_m.mean(axis=0)
            continue

        rng  = np.random.RandomState(SEED+8888+mod_idx)
        cset,_ = _imp_bfs(seeds, valid_emb_m, adj_imp, rng=rng)
        if not cset:
            row[mod_idx] = valid_emb_m.mean(axis=0)
            continue

        # soft imputation via SoftImputationHead
        state, cfs = _imp_features(q_partial, zero_m, cset, valid_emb_m)
        with torch.no_grad():
            sc  = policy(torch.tensor(state,dtype=torch.float32,device=DEVICE),
                         torch.tensor(cfs,  dtype=torch.float32,device=DEVICE))
        top = sc.argsort(descending=True)[:IMP_SOFT_TOPK].cpu().numpy()
        donors = np.stack([valid_emb_m[cset[j][0]] for j in top])
        q_t = torch.tensor(q_partial,dtype=torch.float32,device=DEVICE)
        d_t = torch.tensor(donors,   dtype=torch.float32,device=DEVICE)
        head.eval()
        with torch.no_grad():
            imp_vec = head(q_t, d_t).cpu().numpy()
        row[mod_idx] = imp_vec

    return row


def build_imputed_emb(raw_list, present_mask_all,
                      stats, imp_policies, soft_heads):
    """
    Build normalised embeddings for all N samples given a camera availability mask.
    present_mask_all: (3,) bool — same mask applied to every sample.
    """
    N = raw_list[0].shape[0]
    out = []
    for i in range(N):
        row = impute_one_sample(
            [raw_list[m][i] for m in range(N_MODS)],
            present_mask_all, stats, imp_policies, soft_heads)
        # after imputation all modalities are present → full normalisation
        full_mask = np.ones(N_MODS, dtype=bool)
        out.append(build_partial_embedding(row, full_mask, stats, VARIANT))
    return np.stack(out).astype(np.float32)


# =============================================================================
# ROLLOUT EVALUATION
# =============================================================================

def _make_libero_env(task, resolution=128):
    """
    Create a Libero OffScreenRenderEnv for one task using the correct API.
    Returns (env, init_states).
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights":  resolution,
        "camera_widths":   resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    return env


def _replay_demo(env, init_state, actions, max_steps=MAX_STEPS):
    """
    Reset env to init_state, replay action sequence open-loop.
    Returns True if task succeeds within max_steps.
    """
    env.reset()
    env.set_init_state(init_state)
    for t, action in enumerate(actions):
        if t >= max_steps:
            break
        _, _, done, info = env.step(action)
        if done or info.get("success", False):
            return True
    return False


def evaluate_rollouts(policy, head, train_demos,
                      tr_emb, tr_labels, adj, tr_set_sizes,
                      task_suite, te_emb_cfg, test_demos, config_label):
    """
    Query RL-BFS + FusionHead → retrieve best training demo → replay in sim.
    Uses task_suite.get_task_init_states(task_id) for fixed initial states,
    matching the DisDP evaluation protocol exactly.
    """
    print(f"\n  Evaluating rollouts [{config_label}] …")
    idx = _build_knn_index(tr_emb)
    kd_all, ki_all = idx.kneighbors(te_emb_cfg)

    # group test demos by task language for query lookup
    task_demos = {}
    for qi, demo in enumerate(test_demos):
        task_demos.setdefault(demo["task_lang"], []).append((qi, demo))

    all_successes = []

    for task_id in range(task_suite.n_tasks):
        task      = task_suite.get_task(task_id)
        task_lang = task.language
        tlist     = task_demos.get(task_lang, [])
        if not tlist:
            continue

        # create env and fetch fixed initial states
        try:
            env         = _make_libero_env(task)
            init_states = task_suite.get_task_init_states(task_id)
        except Exception as e:
            print(f"    [WARN] env creation failed for task {task_id}: {e}")
            continue

        task_successes = []
        for seed_idx in range(N_ROLLOUT_SEEDS):
            rng   = np.random.RandomState(SEED + seed_idx * 1000 + task_id)
            # sample N_ROLLOUTS_TASK query indices
            qidxs = rng.choice(len(tlist), size=N_ROLLOUTS_TASK, replace=True)
            # cycle through fixed init states for reproducibility
            ok = 0
            for rollout_i, qi_local in enumerate(qidxs):
                qi, _ = tlist[qi_local]

                # retrieve best training demo via RL-BFS + FusionHead
                best_tr_idx, _ = _infer_one_fusion(
                    te_emb_cfg[qi], ki_all[qi], kd_all[qi],
                    tr_emb, tr_labels, adj, tr_set_sizes, policy, head)
                actions    = train_demos[best_tr_idx]["actions"]
                init_state = init_states[rollout_i % len(init_states)]

                try:
                    if _replay_demo(env, init_state, actions):
                        ok += 1
                except Exception:
                    pass

            task_successes.append(ok / N_ROLLOUTS_TASK)

        mean_sr = float(np.mean(task_successes))
        std_sr  = float(np.std(task_successes))
        print(f"    Task {task_id:02d}  {task_lang[:50]:<50s}  "
              f"SR={mean_sr:.3f} ± {std_sr:.3f}")
        all_successes.extend(task_successes)
        env.close()

    mean_all = float(np.mean(all_successes)) if all_successes else 0.0
    std_all  = float(np.std(all_successes))  if all_successes else 0.0
    return mean_all, std_all


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device   : {DEVICE}")
    print(f"Variant  : {VARIANT}")
    print(f"Strategy : {STRATEGY}\n")

    # ── load data ─────────────────────────────────────────────────────────────
    train_demos, val_demos, test_demos, task_suite = load_libero_spatial(
        DATASET_ROOT, LIBERO_SUITE)

    # ── extract embeddings ────────────────────────────────────────────────────
    print("Loading frozen CLIP encoder …")
    clip_enc = FrozenCLIPEncoder().to(DEVICE)
    print("  Done.\n")

    (tr_emb, tr_labels, va_emb, va_labels, te_emb, te_labels,
     tr_raw, va_raw, te_raw, stats) = build_all_embeddings(
        clip_enc, train_demos, val_demos, test_demos)

    del clip_enc; torch.cuda.empty_cache()
    print("  CLIP encoder freed.\n")

    # ── PHASE 1: Prediction RL + Fusion Head (full modalities) ────────────────
    print("=" * 60)
    print("PHASE 1 — Prediction RL-BFS + Fusion Head")
    print("=" * 60)
    adj       = _build_knn_graph(tr_emb)
    print(f"Building BFS candidate sets (strategy='{STRATEGY}') …")
    all_sets, all_oracles, set_sizes = build_all_sets(tr_emb, tr_labels, adj)
    print()

    if SKIP_TRAINING and os.path.exists(RL_CKPT) and os.path.exists(FUSION_CKPT):
        print(f"  SKIP_TRAINING=True — loading checkpoints …")
        D = tr_emb.shape[1]
        policy = ScoringMLP(D+2, D+4).to(DEVICE)
        policy.load_state_dict(torch.load(RL_CKPT, map_location=DEVICE))
        policy.eval()
        head = FusionHead(emb_dim=D).to(DEVICE)
        head.load_state_dict(torch.load(FUSION_CKPT, map_location=DEVICE))
        head.eval()
        print(f"  Loaded {RL_CKPT} and {FUSION_CKPT}\n")
    else:
        policy = train_policy(tr_emb, tr_labels, all_sets, all_oracles,
                              va_emb, va_labels, adj, set_sizes)
        head   = train_fusion(policy, tr_emb, tr_labels, all_sets,
                              va_emb, va_labels, adj, set_sizes)

    # ── PHASE 2: Imputation RL + Soft Heads (for camera-dropout configs) ──────
    print("\n" + "=" * 60)
    print("PHASE 2 — Imputation RL + Soft Imputation Heads")
    print("=" * 60)
    imp_policies = []
    soft_heads   = []
    for mod_idx in range(N_MODS):
        if mod_idx == 2:          # language always present — no imputation
            imp_policies.append(None); soft_heads.append(None); continue

        rl_ckpt_m   = f"{IMP_CKPT_PREFIX}_mod{mod_idx}.pt"
        soft_ckpt_m = f"{IMP_SOFT_CKPT_PREFIX}_mod{mod_idx}.pt"
        dm          = MOD_DIMS[mod_idx]
        D_full      = sum(MOD_DIMS)

        # always rebuild the kNN index / adj (cheap, no training)
        tr_raw_m    = tr_raw[mod_idx]
        valid_idx_m = np.arange(tr_raw_m.shape[0])
        valid_emb_m = tr_raw_m
        kq          = min(K_APPROX+1, len(valid_idx_m))
        imp_index_m = NearestNeighbors(n_neighbors=kq, algorithm="auto",
                                       metric="euclidean", n_jobs=-1).fit(valid_emb_m)
        adj_imp_m   = _build_knn_graph(valid_emb_m, k=K_GRAPH)

        if SKIP_TRAINING and os.path.exists(rl_ckpt_m) and os.path.exists(soft_ckpt_m):
            print(f"  Modality {MOD_NAMES[mod_idx]}: loading from checkpoints …")
            policy_m = ScoringMLP(D_full+dm+2, dm+3).to(DEVICE)
            policy_m.load_state_dict(torch.load(rl_ckpt_m, map_location=DEVICE))
            policy_m.eval()
            sh = SoftImputationHead(D_full, dm).to(DEVICE)
            sh.load_state_dict(torch.load(soft_ckpt_m, map_location=DEVICE))
            sh.eval()
            result = (policy_m, imp_index_m, valid_idx_m, valid_emb_m, adj_imp_m)
        else:
            result = train_imp_policy(mod_idx, tr_raw, va_raw, stats)
            policy_m, imp_index_m, valid_idx_m, valid_emb_m, adj_imp_m = result
            sh = train_soft_imp_head(mod_idx, policy_m,
                                     imp_index_m, valid_emb_m, adj_imp_m,
                                     tr_raw, va_raw, stats)

        imp_policies.append(result)
        soft_heads.append(sh)

    # ── PHASE 3: Rollout evaluation across all configs ─────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 — Rollout Evaluation (Tables 1 / 3 / 4)")
    print("=" * 60)

    results = {}
    for config_label, cam0_present, cam1_present in EVAL_CONFIGS:
        present_mask = np.array([cam0_present, cam1_present, True], dtype=bool)

        if cam0_present and cam1_present:
            # Table 1: all modalities present — no imputation needed
            te_emb_cfg = te_emb
            tr_emb_cfg = tr_emb
            adj_cfg    = adj
            sizes_cfg  = set_sizes
            policy_cfg = policy
            head_cfg   = head
        else:
            # Tables 3/4: one camera missing — impute via RL + SoftHead
            print(f"\n  Building imputed embeddings for [{config_label}] …")
            # impute training set (for building the retrieval graph)
            tr_emb_cfg = build_imputed_emb(
                tr_raw, present_mask, stats, imp_policies, soft_heads)
            # impute test queries
            te_emb_cfg = build_imputed_emb(
                te_raw, present_mask, stats, imp_policies, soft_heads)

            # rebuild BFS on imputed training embeddings
            adj_cfg = _build_knn_graph(tr_emb_cfg)
            sets_cfg, oracles_cfg, sizes_cfg = build_all_sets(
                tr_emb_cfg, tr_labels, adj_cfg)

            # retrain prediction policy + fusion head on imputed space
            policy_cfg = train_policy(
                tr_emb_cfg, tr_labels, sets_cfg, oracles_cfg,
                va_emb, va_labels, adj_cfg, sizes_cfg)
            head_cfg = train_fusion(
                policy_cfg, tr_emb_cfg, tr_labels, sets_cfg,
                va_emb, va_labels, adj_cfg, sizes_cfg)

        mean_sr, std_sr = evaluate_rollouts(
            policy_cfg, head_cfg, train_demos,
            tr_emb_cfg, tr_labels, adj_cfg, sizes_cfg,
            task_suite, te_emb_cfg, test_demos, config_label)
        results[config_label] = (mean_sr, std_sr)

    # ── Final summary ──────────────────────────────────────────────────────────
    sep = "═" * 70
    print(f"\n{sep}")
    fusion_label = "Soft Fusion" if USE_SOFT_FUSION else "Hard Argmax"
    print(f"  RESULTS — {LIBERO_SUITE}  (RL-BFS + {fusion_label})")
    print(f"  variant='{VARIANT}'  strategy='{STRATEGY}'")
    print(sep)

    disdp_ref = {
        "libero_object": {
            "none":   "0.816 ± 0.02",
            "mask_0": "0.295 ± 0.04",
            "mask_1": "0.226 ± 0.03",
        },
        "libero_spatial": {
            "none":   "0.701 ± 0.04",
            "mask_0": "0.144 ± 0.02",
            "mask_1": "0.112 ± 0.00",
        },
    }
    ref = disdp_ref.get(LIBERO_SUITE, {})

    cam_labels = {
        "none":   "All cams",
        "mask_0": "Cam0 masked (agent)",
        "mask_1": "Cam1 masked (in-hand)",
    }
    print(f"  {'Config':<24s}  {'Ours':>16s}  {'DisDP (ref)':>14s}  {'Delta':>8s}")
    print(f"  {'-'*24}  {'-'*16}  {'-'*14}  {'-'*8}")
    for cfg, _c0, _c1 in EVAL_CONFIGS:
        if cfg in results:
            m, s   = results[cfg]
            ours   = f"{m:.3f} ± {s:.3f}"
            disdp  = ref.get(cfg, "n/a")
            if disdp != "n/a":
                disdp_m = float(disdp.split()[0])
                delta   = f"{m - disdp_m:+.3f}"
            else:
                delta = "n/a"
            print(f"  {cam_labels[cfg]:<24s}  {ours:>16s}  {disdp:>14s}  {delta:>8s}")
    print(sep)
    print()
    print("  Delta = Ours - DisDP.  Positive = our method wins.")
    print(sep + "\n")

    fusion_tag = "soft" if USE_SOFT_FUSION else "hard"
    out_json = f"results_{LIBERO_SUITE}_{VARIANT}_{STRATEGY}_{fusion_tag}.json"
    with open(out_json,"w") as f:
        json.dump({k:list(v) for k,v in results.items()}, f, indent=2)
    print(f"  Results saved → {out_json}\n")


if __name__ == "__main__":
    main()