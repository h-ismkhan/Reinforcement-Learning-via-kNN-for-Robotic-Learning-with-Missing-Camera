
"""
topk_study.py
=============
Top-K ablation for RL4IL on a single LIBERO suite.

For each value in TOPK_SET, both FUSION_TOPK (prediction fusion head)
and IMP_SOFT_TOPK (soft imputation head) are set to that value.
All other hyperparameters remain fixed.

Results are written to a CSV:
  <suite>_<variant>_<strategy>_sf<0|1>_e<epochs>_topk.csv
  Columns: top_k, sr_mask0, sr_mask1

═══════════════════════════════════════════════════════════════════
ALL TUNEABLE PARAMETERS ARE IN THE BLOCK BELOW.
═══════════════════════════════════════════════════════════════════
"""

import os, copy, heapq, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from libero.libero.benchmark import get_benchmark_dict
from transformers import CLIPProcessor, CLIPModel

# =============================================================================
# PARAMETERS
# =============================================================================

DATASET_ROOT = "/.../datasets"
LIBERO_SUITE = "libero_spatial"   # change to libero_object / libero_spatial /libero_goal

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Top-K study ───────────────────────────────────────────────────────────────
# Each value sets both FUSION_TOPK and IMP_SOFT_TOPK simultaneously
TOPK_SET = [4, 8, 12, 16, 32]

# ── Fixed epoch counts ────────────────────────────────────────────────────────
PPO_EPOCHS     = 30   # prediction RL policy
FUSION_EPOCHS  = 30   # prediction FusionHead
IMP_PPO_EPOCHS = 50   # imputation RL policy
IMP_SOFT_EPOCHS = 30  # soft imputation head

# ── Modalities ────────────────────────────────────────────────────────────────
CAM0_DIM  = 512
CAM1_DIM  = 512
LANG_DIM  = 512
N_MODS    = 3
MOD_NAMES = ["agent_cam", "inhand_cam", "language"]
MOD_DIMS  = [CAM0_DIM, CAM1_DIM, LANG_DIM]
NUM_FRAMES = 8

VAL_FRAC   = 0.15
SEED       = 42
VARIANT    = "with"
STRATEGY   = "knn"

# ── BFS ───────────────────────────────────────────────────────────────────────
K_APPROX      = 20
K_SEED_KNN    = 5
K_GRAPH       = 5
MAX_BFS_DEPTH = 6
MAX_NODE2_RL  = 200

# ── RL hyperparams (fixed) ────────────────────────────────────────────────────
PPO_CLIP       = 0.2
LR             = 3e-4
ENT_COEF       = 0.0001
MINIBATCH_SIZE = 128

FUSION_HIDDEN  = 128
FUSION_HEADS   = 4
FUSION_LR      = 3e-4
FUSION_BATCH   = 32

IMP_MINIBATCH_SIZE = 64
IMP_LR             = 3e-4
IMP_ENT_COEF       = 0.0001
IMP_SOFT_HIDDEN    = 64
IMP_SOFT_HEADS     = 2
IMP_SOFT_LR        = 3e-4
IMP_SOFT_BATCH     = 32

USE_SOFT_FUSION = True

# ── Rollout evaluation ────────────────────────────────────────────────────────
N_ROLLOUT_SEEDS = 3
N_ROLLOUTS_TASK = 25
MAX_STEPS       = 260

# Both dropout configs evaluated separately (two columns in CSV)
DROPOUT_CONFIGS = [
    ("mask_0", False, True),   # agent cam missing
    ("mask_1", True,  False),  # in-hand cam missing
]

# =============================================================================
# FROZEN CLIP ENCODER
# =============================================================================

class FrozenCLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, frames_list):
        device = next(self.model.parameters()).device
        all_feats = []
        for frames in frames_list:
            if len(frames) == 0:
                all_feats.append(torch.zeros(512, device=device)); continue
            if len(frames) >= NUM_FRAMES:
                idxs = np.linspace(0, len(frames)-1, NUM_FRAMES, dtype=int)
            else:
                repeat = (NUM_FRAMES + len(frames) - 1) // len(frames)
                idxs = np.tile(np.arange(len(frames)), repeat)[:NUM_FRAMES]
            pil_frames = [Image.fromarray(frames[i].astype("uint8")) for i in idxs]
            inputs = self.processor(images=pil_frames, return_tensors="pt",
                                    padding=True).to(device)
            out = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            all_feats.append(out.mean(dim=0))
        return torch.stack(all_feats)

    @torch.no_grad()
    def encode_text(self, texts_list):
        device = next(self.model.parameters()).device
        inputs = self.processor(text=texts_list, return_tensors="pt",
                                padding=True, truncation=True, max_length=77).to(device)
        return self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"])


# =============================================================================
# DATA LOADING
# =============================================================================

def _find_hdf5(dataset_root, suite_name, task_name):
    suite_dir = os.path.join(dataset_root, suite_name)
    for p in [os.path.join(suite_dir, f"{task_name}_demo.hdf5"),
              os.path.join(suite_dir, f"{task_name}.hdf5")]:
        if os.path.exists(p): return p
    if not os.path.isdir(suite_dir): return None
    all_hdf5 = [f for f in os.listdir(suite_dir) if f.endswith(".hdf5")]
    tn_lower = task_name.lower()
    for fname in all_hdf5:
        stem = fname.replace(".hdf5","").replace("_demo","").lower()
        if tn_lower == stem or tn_lower in stem or stem in tn_lower:
            return os.path.join(suite_dir, fname)
    tn_words = set(tn_lower.split("_"))
    best_score, best_path = 0, None
    for fname in all_hdf5:
        stem = fname.replace(".hdf5","").replace("_demo","").lower()
        score = len(tn_words & set(stem.split("_")))
        if score > best_score:
            best_score = score; best_path = os.path.join(suite_dir, fname)
    return best_path if best_score >= max(1, len(tn_words)//2) else None


def _load_task_demos(hdf5_path, task_lang, rng):
    demos = []
    with h5py.File(hdf5_path, "r") as f:
        for dk in sorted(f["data"].keys()):
            demo_grp = f["data"][dk]
            actions  = demo_grp["actions"][:]
            obs_grp  = demo_grp["obs"]
            cam0 = None
            for key in ("agentview_rgb","agentview_image","agentview",
                        "frontview_rgb","frontview_image","frontview"):
                if key in obs_grp: cam0 = obs_grp[key][:]; break
            if cam0 is None:
                for key in obs_grp.keys():
                    if "agent" in key.lower() or "front" in key.lower():
                        cam0 = obs_grp[key][:]; break
            if cam0 is None: cam0 = np.zeros((1,128,128,3),dtype=np.uint8)
            cam1 = None
            for key in ("robot0_eye_in_hand_rgb","eye_in_hand_image",
                        "robot0_eye_in_hand","wrist_rgb","wrist_image","eye_in_hand_rgb"):
                if key in obs_grp: cam1 = obs_grp[key][:]; break
            if cam1 is None:
                for key in obs_grp.keys():
                    if "hand" in key.lower() or "wrist" in key.lower():
                        cam1 = obs_grp[key][:]; break
            if cam1 is None: cam1 = np.zeros((1,128,128,3),dtype=np.uint8)
            demos.append({"demo_key":dk,"task_lang":task_lang,
                          "cam0_frames":[cam0[t] for t in range(len(cam0))],
                          "cam1_frames":[cam1[t] for t in range(len(cam1))],
                          "actions":actions.astype(np.float32),"label":1})
    rng.shuffle(demos)
    return demos


def load_suite(dataset_root, suite_name, val_frac=VAL_FRAC, seed=SEED):
    benchmark_dict = get_benchmark_dict()
    task_suite     = benchmark_dict[suite_name]()
    rng = np.random.RandomState(seed)
    all_train, all_val, all_test = [], [], []
    print(f"Loading {suite_name} ({task_suite.n_tasks} tasks) …")
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        hdf5_path = _find_hdf5(dataset_root, suite_name, task.name)
        if hdf5_path is None:
            print(f"  [skip] {task.name}"); continue
        demos = _load_task_demos(hdf5_path, task.language, rng)
        n     = len(demos); n_val = max(1, int(n*val_frac))
        all_train.extend(demos[n_val:])
        all_val.extend(demos[:n_val])
        all_test.extend(demos[:n_val])
        print(f"  Task {task_id:02d}  train={n-n_val}  val={n_val}")
    print(f"  Total: train={len(all_train)} val={len(all_val)}\n")
    return all_train, all_val, all_test, task_suite


# =============================================================================
# EMBEDDINGS
# =============================================================================

@torch.no_grad()
def extract_embeddings(demos, clip_enc, batch_size=8):
    N = len(demos); c0l,c1l,ltl,lbl = [],[],[],[]
    nb = (N+batch_size-1)//batch_size
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
            np.array(lbl,dtype=np.float32))


def compute_norm_stats(c0,c1,lt):
    stats = {}
    for m,Z in enumerate([c0,c1,lt]):
        mu = Z.mean(0); sigma = Z.std(0,ddof=1)
        sigma = np.where(sigma<1e-8,1.0,sigma)
        stats[m] = (mu,sigma)
    return stats


def build_partial_embedding(raw_mods, present_mask, stats, variant):
    M_pres = max(1, int(present_mask.sum())); parts = []
    for m,(Z_row,dm) in enumerate(zip(raw_mods,MOD_DIMS)):
        Z = Z_row.copy().astype(np.float32)
        if not present_mask[m]:
            parts.append(np.zeros(dm,dtype=np.float32)); continue
        mu,sigma = stats[m]
        if variant == "with":
            Z = (Z-mu)/sigma; Z = Z/(dm*M_pres)**0.5
        elif variant == "just_scaling":
            Z = Z/(dm*M_pres)**0.5
        parts.append(Z)
    return np.concatenate(parts).astype(np.float32)


def _build_full_emb(raw_list, stats):
    N = raw_list[0].shape[0]; full_mask = np.ones(N_MODS,dtype=bool)
    return np.stack([build_partial_embedding(
        [raw_list[m][i] for m in range(N_MODS)],full_mask,stats,VARIANT)
        for i in range(N)]).astype(np.float32)


def build_all_embeddings(clip_enc, train_demos, val_demos, test_demos):
    print("Extracting embeddings …")
    print("  TRAIN …"); tr_c0,tr_c1,tr_lt,tr_labels = extract_embeddings(train_demos,clip_enc)
    print("  VAL   …"); va_c0,va_c1,va_lt,va_labels = extract_embeddings(val_demos,  clip_enc)
    print("  TEST  …"); te_c0,te_c1,te_lt,te_labels = extract_embeddings(test_demos, clip_enc)
    stats  = compute_norm_stats(tr_c0,tr_c1,tr_lt)
    tr_raw = [tr_c0,tr_c1,tr_lt]; va_raw = [va_c0,va_c1,va_lt]; te_raw = [te_c0,te_c1,te_lt]
    print("  Building full embeddings …")
    tr_emb = _build_full_emb(tr_raw,stats)
    va_emb = _build_full_emb(va_raw,stats)
    te_emb = _build_full_emb(te_raw,stats)
    print(f"  Dim: {tr_emb.shape[1]}\n")
    return (tr_emb,tr_labels,va_emb,va_labels,te_emb,te_labels,tr_raw,va_raw,te_raw,stats)


# =============================================================================
# BFS / RL CORE
# =============================================================================

def _build_knn_graph(emb, k=K_GRAPH):
    N=emb.shape[0]; kq=min(k+1,N)
    idx=NearestNeighbors(n_neighbors=kq,algorithm="auto",metric="euclidean",n_jobs=-1).fit(emb)
    dists,idxs=idx.kneighbors(emb)
    adj=[[] for _ in range(N)]
    for i in range(N):
        for p in range(kq):
            j=idxs[i,p]; d=dists[i,p]
            if j!=i and d>1e-12: adj[i].append((j,float(d)))
    return adj


def _build_knn_index(emb,k=K_APPROX):
    kq=min(k+1,emb.shape[0])
    return NearestNeighbors(n_neighbors=kq,algorithm="auto",
                            metric="euclidean",n_jobs=-1).fit(emb)


def get_seeds_knn(knn_idxs,knn_dists,k=K_SEED_KNN):
    return [(int(knn_idxs[i]),float(knn_dists[i])) for i in range(min(k,len(knn_idxs)))]


def _get_seeds(q_vec,tr_emb,knn_idxs,knn_dists):
    return get_seeds_knn(knn_idxs,knn_dists)


def _bfs(seeds,adj,target_sz=None,max_depth=MAX_BFS_DEPTH,max_nodes=MAX_NODE2_RL):
    visited={}; heap=[]
    for idx,dist in seeds:
        if idx not in visited:
            visited[idx]=(dist,0); heapq.heappush(heap,(dist,idx,0))
    sequence=[]
    while heap:
        g_dist,node,depth=heapq.heappop(heap)
        if g_dist>visited.get(node,(float("inf"),))[0]+1e-9: continue
        sequence.append((node,g_dist,depth))
        if target_sz is not None and len(sequence)>=target_sz: break
        for nb,ew in adj[node]:
            nd=g_dist+ew
            if nd<visited.get(nb,(float("inf"),))[0]:
                visited[nb]=(nd,depth+1); heapq.heappush(heap,(nd,nb,depth+1))
        if target_sz is None and depth>=max_depth and len(sequence)>=max_nodes: break
    return sequence


def build_candidate_set(seeds,target_label,tr_labels,adj,rng=None):
    sequence=_bfs(seeds,adj)
    if not sequence: return None,None
    oracle_pos=min(range(len(sequence)),
                   key=lambda p:abs(float(tr_labels[sequence[p][0]])-float(target_label)))
    if rng is None: rng=np.random.RandomState()
    perm=rng.permutation(len(sequence))
    sequence=[sequence[p] for p in perm]
    oracle_pos=int(np.where(perm==oracle_pos)[0][0])
    return sequence,oracle_pos


def build_all_sets(emb,labels,adj):
    N=emb.shape[0]; kq=min(K_APPROX+1,N)
    index=NearestNeighbors(n_neighbors=kq,algorithm="auto",metric="euclidean",n_jobs=-1).fit(emb)
    knn_dists_all,knn_idxs_all=index.kneighbors(emb)
    all_sets,all_oracles=[],[]; set_sizes=np.zeros(N,dtype=np.float32)
    rng=np.random.RandomState(SEED)
    for i in range(N):
        mask=knn_dists_all[i]>1e-12
        idxs=knn_idxs_all[i][mask]; dists=knn_dists_all[i][mask]
        seeds=_get_seeds(emb[i],emb,idxs,dists)
        if not seeds: all_sets.append(None); all_oracles.append(None); continue
        cset,oracle=build_candidate_set(seeds,labels[i],labels,adj,rng=rng)
        all_sets.append(cset); all_oracles.append(oracle)
        if cset is not None: set_sizes[i]=float(len(cset))
    v=set_sizes[set_sizes>0]
    print(f"  Sets: {sum(s is not None for s in all_sets)}/{N} | mean size: {v.mean():.1f}")
    return all_sets,all_oracles,set_sizes


def build_features(q_vec,candidate_set,tr_emb,tr_labels):
    n=len(candidate_set)
    cand_labels=np.array([float(tr_labels[idx]) for idx,_,_ in candidate_set])
    mean_lbl=cand_labels.mean(); var_lbl=cand_labels.var()
    max_depth=max(dep for _,_,dep in candidate_set)+1e-9
    max_dist =max(d   for _,d,_   in candidate_set)+1e-9
    rank_map={orig:r for r,(orig,_) in
              enumerate(sorted(enumerate(candidate_set),key=lambda x:x[1][1]))}
    state=np.concatenate([q_vec,[var_lbl],[float(n)]]).astype(np.float32)
    cfs=[]
    for pos,(idx,dist,depth) in enumerate(candidate_set):
        cfs.append(np.concatenate([tr_emb[idx],[dist/max_dist],[depth/max_depth],
                                   [rank_map[pos]/max(n-1,1)],
                                   [float(tr_labels[idx])-mean_lbl]]).astype(np.float32))
    return state,np.stack(cfs)


# =============================================================================
# MODELS  (topk passed as argument so models are built fresh per run)
# =============================================================================

class ScoringMLP(nn.Module):
    def __init__(self,state_dim,cand_dim,hidden=256):
        super().__init__()
        self.qe=nn.Sequential(nn.Linear(state_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.ce=nn.Sequential(nn.Linear(cand_dim, hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.sh=nn.Sequential(nn.Linear(3*hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,state,cand_feats):
        sq=state.dim()==1
        if sq: state=state.unsqueeze(0); cand_feats=cand_feats.unsqueeze(0)
        B,K,_=cand_feats.shape
        h_o=self.qe(state); h_oe=h_o.unsqueeze(1).expand(-1,K,-1)
        h_i=self.ce(cand_feats)
        s=self.sh(torch.cat([h_oe,h_i,h_oe*h_i],dim=-1)).squeeze(-1)
        return s.squeeze(0) if sq else s


class FusionHead(nn.Module):
    def __init__(self,emb_dim,hidden=FUSION_HIDDEN,n_heads=FUSION_HEADS,topk=32):
        super().__init__()
        self.topk=topk; self.n_heads=n_heads; self.d_head=hidden//n_heads
        assert hidden%n_heads==0
        self.q_proj=nn.Linear(emb_dim,hidden,bias=False)
        self.k_proj=nn.Linear(emb_dim,hidden,bias=False)
        self.scale=self.d_head**-0.5
        self.ctx_proj=nn.Linear(hidden,hidden)
        self.refine=nn.Sequential(nn.Linear(hidden+1,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,q_emb,cand_embs,cand_labels):
        K=cand_embs.shape[0]; H=self.n_heads; Dh=self.d_head
        q=self.q_proj(q_emb).view(H,Dh); k=self.k_proj(cand_embs).view(K,H,Dh)
        w=F.softmax(torch.einsum("hd,khd->hk",q,k)*self.scale,dim=-1).mean(0)
        att_lbl=(w*cand_labels).sum().unsqueeze(0)
        k_flat=k.reshape(K,H*Dh)
        ctx=F.relu(self.ctx_proj((w.unsqueeze(-1)*k_flat).sum(0)))
        return self.refine(torch.cat([ctx,att_lbl],-1)).squeeze(-1)


class SoftImputationHead(nn.Module):
    def __init__(self,d_full,d_m,hidden=IMP_SOFT_HIDDEN,n_heads=IMP_SOFT_HEADS):
        super().__init__()
        self.d_m=d_m; self.n_heads=n_heads; self.d_head=hidden//n_heads
        assert hidden%n_heads==0
        self.q_proj=nn.Linear(d_full,hidden,bias=False)
        self.k_proj=nn.Linear(d_m,   hidden,bias=False)
        self.scale=self.d_head**-0.5
        self.refine=nn.Sequential(nn.Linear(d_m+hidden,hidden),nn.ReLU(),nn.Linear(hidden,d_m))
    def forward(self,q_partial,donor_embs):
        K=donor_embs.shape[0]; H=self.n_heads; Dh=self.d_head
        q=self.q_proj(q_partial).view(H,Dh); k=self.k_proj(donor_embs).view(K,H,Dh)
        w=F.softmax(torch.einsum("hd,khd->hk",q,k)*self.scale,dim=-1).mean(0)
        attended=(w.unsqueeze(-1)*donor_embs).sum(0)
        k_flat=k.reshape(K,H*Dh)
        context=F.relu((w.unsqueeze(-1)*k_flat).sum(0))
        return self.refine(torch.cat([attended,context],-1))


# =============================================================================
# PPO UPDATE
# =============================================================================

def ppo_update(policy,optimizer,rollout,clip=PPO_CLIP,ent_coef=ENT_COEF):
    states,cands_l,actions,old_lps,advantages=rollout; losses=[]
    for i in range(len(states)):
        scores=policy(states[i],cands_l[i])
        log_pi=F.log_softmax(scores,dim=-1); pi=log_pi.exp()
        ratio=torch.exp(log_pi[actions[i]]-old_lps[i].detach())
        adv=advantages[i].detach(); entropy=-(pi*log_pi).sum()
        losses.append(-torch.min(ratio*adv,torch.clamp(ratio,1-clip,1+clip)*adv)-ent_coef*entropy)
    loss=torch.stack(losses).mean()
    optimizer.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(),1.0); optimizer.step()
    return loss.item()


# =============================================================================
# TRAINING FUNCTIONS  (topk passed as argument)
# =============================================================================

@torch.no_grad()
def _topk_from_sequence(q_vec,sequence,tr_emb,tr_labels,policy,topk):
    if not sequence: return None,None,None
    state,cfs=build_features(q_vec,sequence,tr_emb,tr_labels)
    scores=policy(torch.tensor(state,device=DEVICE),torch.tensor(cfs,device=DEVICE))
    top=scores.argsort(descending=True)[:topk].cpu().numpy()
    embs  =np.stack([tr_emb[sequence[j][0]]          for j in top])
    labels=np.array([float(tr_labels[sequence[j][0]]) for j in top])
    didxs =[sequence[j][0] for j in top]
    return embs,labels,didxs


def _infer_one_fusion(q_vec,knn_idxs,knn_dists,tr_emb,tr_labels,
                      adj,tr_set_sizes,policy,head,fusion_topk):
    mask=knn_dists>1e-12; idxs=knn_idxs[mask]; dists=knn_dists[mask]
    if len(idxs)==0: return 0,float(tr_labels[0])
    seeds=_get_seeds(q_vec,tr_emb,idxs,dists)
    if not seeds: return int(idxs[0]),float(tr_labels[idxs[0]])
    target_sz=max(1,int(max(float(tr_set_sizes[idx]) for idx,_ in seeds)))
    sequence=_bfs(seeds,adj,target_sz=target_sz)
    if not sequence: return int(idxs[0]),float(tr_labels[idxs[0]])
    if not USE_SOFT_FUSION or head is None:
        state,cfs=build_features(q_vec,sequence,tr_emb,tr_labels)
        sc=policy(torch.tensor(state,device=DEVICE),torch.tensor(cfs,device=DEVICE))
        best=int(sc.argmax().item())
        return sequence[best][0],float(tr_labels[sequence[best][0]])
    ce,cl,didxs=_topk_from_sequence(q_vec,sequence,tr_emb,tr_labels,policy,fusion_topk)
    if ce is None: return sequence[0][0],float(tr_labels[sequence[0][0]])
    q_t=torch.tensor(q_vec,dtype=torch.float32,device=DEVICE)
    ce_t=torch.tensor(ce,dtype=torch.float32,device=DEVICE)
    cl_t=torch.tensor(cl,dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): score=head(q_t,ce_t,cl_t).item()
    return didxs[0],score


def train_policy(tr_emb,tr_labels,tr_sets,tr_oracles,
                 va_emb,va_labels,adj,set_sizes):
    D=tr_emb.shape[1]
    policy=ScoringMLP(D+2,D+4).to(DEVICE)
    optimizer=torch.optim.Adam(policy.parameters(),lr=LR)
    best_acc,best_state=-1.0,None
    valid_tr=[i for i in range(len(tr_emb))
              if tr_sets[i] is not None and len(tr_sets[i])>=2 and tr_oracles[i] is not None]

    for epoch in range(PPO_EPOCHS):
        ep_rng=np.random.RandomState(SEED+epoch)
        epoch_sets,epoch_oracles={},{}
        for i in valid_tr:
            cset=tr_sets[i]; perm=ep_rng.permutation(len(cset))
            nc=[cset[p] for p in perm]; onode=cset[tr_oracles[i]][0]
            epoch_sets[i]=nc
            epoch_oracles[i]=int(np.where(np.array([n for n,_,_ in nc])==onode)[0][0])
        ep_rng.shuffle(valid_tr)
        states,cands_l,actions,old_lps,advantages=[],[],[],[],[]
        policy.eval()
        with torch.no_grad():
            for i in valid_tr:
                cset=epoch_sets[i]; oracle=epoch_oracles[i]
                state,cfs=build_features(tr_emb[i],cset,tr_emb,tr_labels)
                s_t=torch.tensor(state,device=DEVICE); cf_t=torch.tensor(cfs,device=DEVICE)
                scores=policy(s_t,cf_t); log_pi=F.log_softmax(scores,dim=-1)
                action=int(torch.multinomial(log_pi.exp(),1).item()); old_lp=log_pi[action]
                K=len(cset); y_s=float(tr_labels[i])
                if K<=1: reward=0.0
                else:
                    errs=[abs(float(tr_labels[cset[j][0]])-y_s) for j in range(K)]
                    si=np.argsort(errs); rm={int(si[r]):r for r in range(K)}
                    reward=float(rm[oracle]-rm[action])/(K-1)
                states.append(s_t); cands_l.append(cf_t)
                actions.append(action); old_lps.append(old_lp.detach())
                advantages.append(torch.tensor(reward,dtype=torch.float32,device=DEVICE))
        policy.train(); M=len(states); perm=ep_rng.permutation(M)
        for start in range(0,M,MINIBATCH_SIZE):
            mb=perm[start:start+MINIBATCH_SIZE]
            ppo_update(policy,optimizer,
                ([states[j] for j in mb],[cands_l[j] for j in mb],
                 [actions[j] for j in mb],[old_lps[j] for j in mb],
                 [advantages[j] for j in mb]))
        policy.eval(); idx=_build_knn_index(tr_emb); kd,ki=idx.kneighbors(va_emb); preds=[]
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
                sc=policy(torch.tensor(state,device=DEVICE),torch.tensor(cfs,device=DEVICE))
                preds.append(float(tr_labels[seq[int(sc.argmax().item())][0]]))
        val_acc=float(((np.array(preds)>=0.5).astype(int)==np.round(va_labels).astype(int)).mean())
        if val_acc>best_acc: best_acc=val_acc; best_state=copy.deepcopy(policy.state_dict())

    policy.load_state_dict(best_state)
    return policy


def train_fusion(policy,tr_emb,tr_labels,tr_sets,va_emb,va_labels,
                 adj,set_sizes,fusion_topk):
    """Train FusionHead with a specific fusion_topk value."""
    if not USE_SOFT_FUSION: return None
    D=tr_emb.shape[1]
    head=FusionHead(emb_dim=D,topk=fusion_topk).to(DEVICE)
    opt=torch.optim.Adam(head.parameters(),lr=FUSION_LR); policy.eval()
    valid=[i for i in range(len(tr_emb)) if tr_sets[i] is not None and len(tr_sets[i])>=1]
    best_acc,best_state=-1.0,None

    def _eval_acc():
        head.eval(); idx=_build_knn_index(tr_emb); kd,ki=idx.kneighbors(va_emb); preds=[]
        with torch.no_grad():
            for qi in range(len(va_labels)):
                _,s=_infer_one_fusion(va_emb[qi],ki[qi],kd[qi],
                                      tr_emb,tr_labels,adj,set_sizes,policy,head,fusion_topk)
                preds.append(s)
        preds=np.array(preds,dtype=np.float32)
        return float(((preds>=0.5).astype(int)==np.round(va_labels).astype(int)).mean())

    for epoch in range(FUSION_EPOCHS):
        head.train(); rng=np.random.RandomState(SEED+1000+epoch)
        perm=rng.permutation(len(valid))
        for start in range(0,len(valid),FUSION_BATCH):
            mb=[valid[perm[j]] for j in range(start,min(start+FUSION_BATCH,len(valid)))]
            bl=[]
            for i in mb:
                ce,cl,_=_topk_from_sequence(
                    tr_emb[i],tr_sets[i],tr_emb,tr_labels,policy,fusion_topk)
                if ce is None: continue
                q_t=torch.tensor(tr_emb[i],dtype=torch.float32,device=DEVICE)
                ce_t=torch.tensor(ce,dtype=torch.float32,device=DEVICE)
                cl_t=torch.tensor(cl,dtype=torch.float32,device=DEVICE)
                pred=head(q_t,ce_t,cl_t)
                tgt=torch.tensor(float(tr_labels[i]),dtype=torch.float32,device=DEVICE)
                bl.append(F.mse_loss(pred,tgt))
            if not bl: continue
            loss=torch.stack(bl).mean(); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
        val_acc=_eval_acc()
        if val_acc>best_acc: best_acc=val_acc; best_state=copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    return head


def _imp_bfs(seeds,valid_emb_m,adj_imp,gt_emb_m=None,rng=None):
    sequence=_bfs(seeds,adj_imp)
    if not sequence: return None,None
    oracle_pos=(min(range(len(sequence)),
                    key=lambda p:float(np.sum((valid_emb_m[sequence[p][0]]-gt_emb_m)**2)))
                if gt_emb_m is not None else 0)
    if rng is None: rng=np.random.RandomState()
    perm=rng.permutation(len(sequence))
    sequence=[sequence[p] for p in perm]
    oracle_pos=int(np.where(perm==oracle_pos)[0][0])
    return sequence,oracle_pos


def _imp_features(q_partial_emb,q_emb_m,candidate_set,valid_emb_m):
    n=len(candidate_set)
    cand_embs_m=np.stack([valid_emb_m[c[0]] for c in candidate_set])
    dists_m=np.linalg.norm(cand_embs_m-q_emb_m[np.newaxis,:],axis=1)
    max_dist=float(dists_m.max())+1e-9; max_dep=max(dep for _,_,dep in candidate_set)+1e-9
    sr=np.argsort(dists_m); rank_map={int(sr[r]):r for r in range(n)}
    state=np.concatenate([q_partial_emb,cand_embs_m.mean(0),
                           [float(dists_m.var())],[float(n)]]).astype(np.float32)
    cfs=[]
    for pos,(li,g_dist,depth) in enumerate(candidate_set):
        cfs.append(np.concatenate([valid_emb_m[li],[dists_m[pos]/max_dist],
                                   [depth/max_dep],[rank_map[pos]/max(n-1,1)]]).astype(np.float32))
    return state,np.stack(cfs)


def train_imp_policy_one(mod_idx,tr_raw,va_raw,stats):
    """Train imputation RL policy — topk does not affect this phase."""
    dm=MOD_DIMS[mod_idx]; N_tr=tr_raw[0].shape[0]
    tr_raw_m=tr_raw[mod_idx]; valid_emb_m=tr_raw_m
    kq=min(K_APPROX+1,N_tr)
    imp_index=NearestNeighbors(n_neighbors=kq,algorithm="auto",
                               metric="euclidean",n_jobs=-1).fit(valid_emb_m)
    adj_imp=_build_knn_graph(valid_emb_m,k=K_GRAPH)
    D_full=sum(MOD_DIMS)
    policy=ScoringMLP(D_full+dm+2,dm+3).to(DEVICE)
    optimizer=torch.optim.Adam(policy.parameters(),lr=IMP_LR)
    train_items=list(range(N_tr)); best_l2,best_state=float("inf"),None

    for epoch in range(IMP_PPO_EPOCHS):
        ep_rng=np.random.RandomState(SEED+200+mod_idx*100+epoch)
        ep_rng.shuffle(train_items); policy.eval()
        states,cands_l,actions,old_lps,advantages=[],[],[],[],[]
        for i in train_items:
            pm=np.ones(N_MODS,dtype=bool); pm[mod_idx]=False
            q_partial=build_partial_embedding([tr_raw[m][i] for m in range(N_MODS)],pm,stats,VARIANT)
            gt_emb_m=tr_raw_m[i]
            _,knn_loc=imp_index.kneighbors(gt_emb_m.reshape(1,-1)); knn_loc=knn_loc[0]
            knn_d_m=np.linalg.norm(valid_emb_m[knn_loc]-gt_emb_m,axis=1)
            seeds=_get_seeds(gt_emb_m,valid_emb_m,knn_loc,knn_d_m)
            if not seeds: continue
            cset,oracle=_imp_bfs(seeds,valid_emb_m,adj_imp,gt_emb_m,ep_rng)
            if cset is None or len(cset)<2: continue
            state,cfs=_imp_features(q_partial,gt_emb_m,cset,valid_emb_m)
            s_t=torch.tensor(state,dtype=torch.float32,device=DEVICE)
            cf_t=torch.tensor(cfs,dtype=torch.float32,device=DEVICE)
            with torch.no_grad():
                scores=policy(s_t,cf_t); log_pi=F.log_softmax(scores,dim=-1)
                action=int(torch.multinomial(log_pi.exp(),1).item()); old_lp=log_pi[action]
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
        for start in range(0,M,IMP_MINIBATCH_SIZE):
            mb=perm[start:start+IMP_MINIBATCH_SIZE]
            ppo_update(policy,optimizer,
                ([states[j] for j in mb],[cands_l[j] for j in mb],
                 [actions[j] for j in mb],[old_lps[j] for j in mb],
                 [advantages[j] for j in mb]),ent_coef=IMP_ENT_COEF)
        errs=[]; rng=np.random.RandomState(SEED+999); policy.eval()
        for i in range(va_raw[0].shape[0]):
            pm=np.ones(N_MODS,dtype=bool); pm[mod_idx]=False
            q_partial=build_partial_embedding([va_raw[m][i] for m in range(N_MODS)],pm,stats,VARIANT)
            gt_emb_m=va_raw[mod_idx][i]
            _,kl=imp_index.kneighbors(gt_emb_m.reshape(1,-1)); kl=kl[0]
            kd=np.linalg.norm(valid_emb_m[kl]-gt_emb_m,axis=1)
            seeds=_get_seeds(gt_emb_m,valid_emb_m,kl,kd)
            if not seeds: continue
            cset,_=_imp_bfs(seeds,valid_emb_m,adj_imp,gt_emb_m,rng)
            if not cset: continue
            state,cfs=_imp_features(q_partial,gt_emb_m,cset,valid_emb_m)
            with torch.no_grad():
                sc=policy(torch.tensor(state,device=DEVICE),torch.tensor(cfs,device=DEVICE))
            best=int(sc.argmax().item())
            errs.append(float(np.sum((valid_emb_m[cset[best][0]]-gt_emb_m)**2)))
        val_l2=float(np.mean(errs)) if errs else float("inf")
        if val_l2<best_l2: best_l2=val_l2; best_state=copy.deepcopy(policy.state_dict())

    policy.load_state_dict(best_state)
    return policy,imp_index,np.arange(N_tr),valid_emb_m,adj_imp


def train_soft_imp_head_one(mod_idx,imp_policy,imp_index,valid_emb_m,adj_imp,
                             tr_raw,va_raw,stats,imp_soft_topk):
    """Train SoftImputationHead with a specific imp_soft_topk value."""
    if not USE_SOFT_FUSION: return None
    D_full=sum(MOD_DIMS); dm=MOD_DIMS[mod_idx]
    head=SoftImputationHead(D_full,dm).to(DEVICE)
    opt=torch.optim.Adam(head.parameters(),lr=IMP_SOFT_LR); imp_policy.eval()
    train_items=list(range(tr_raw[0].shape[0])); best_mse,best_state=float("inf"),None

    for epoch in range(IMP_SOFT_EPOCHS):
        head.train(); rng=np.random.RandomState(SEED+3000+mod_idx*100+epoch)
        perm=rng.permutation(len(train_items))
        for start in range(0,len(train_items),IMP_SOFT_BATCH):
            mb=[train_items[perm[j]] for j in range(start,min(start+IMP_SOFT_BATCH,len(train_items)))]
            bl=[]
            for i in mb:
                pm=np.ones(N_MODS,dtype=bool); pm[mod_idx]=False
                q_partial=build_partial_embedding([tr_raw[m][i] for m in range(N_MODS)],pm,stats,VARIANT)
                gt_emb_m=tr_raw[mod_idx][i]
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
                # use imp_soft_topk here
                top=sc.argsort(descending=True)[:imp_soft_topk].cpu().numpy()
                donors=np.stack([valid_emb_m[cset[j][0]] for j in top])
                q_t=torch.tensor(q_partial,dtype=torch.float32,device=DEVICE)
                d_t=torch.tensor(donors,   dtype=torch.float32,device=DEVICE)
                pred=head(q_t,d_t)
                gt_t=torch.tensor(gt_emb_m,dtype=torch.float32,device=DEVICE)
                bl.append(F.mse_loss(pred,gt_t))
            if not bl: continue
            loss=torch.stack(bl).mean(); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
        # val mse
        head.eval(); errs=[]; rng2=np.random.RandomState(SEED+4444)
        for i in range(va_raw[0].shape[0]):
            pm=np.ones(N_MODS,dtype=bool); pm[mod_idx]=False
            q_partial=build_partial_embedding([va_raw[m][i] for m in range(N_MODS)],pm,stats,VARIANT)
            gt_emb_m=va_raw[mod_idx][i]
            _,kl=imp_index.kneighbors(gt_emb_m.reshape(1,-1)); kl=kl[0]
            kd=np.linalg.norm(valid_emb_m[kl]-gt_emb_m,axis=1)
            seeds=_get_seeds(gt_emb_m,valid_emb_m,kl,kd)
            if not seeds: continue
            cset,_=_imp_bfs(seeds,valid_emb_m,adj_imp,gt_emb_m,rng2)
            if not cset: continue
            state,cfs=_imp_features(q_partial,gt_emb_m,cset,valid_emb_m)
            with torch.no_grad():
                sc=imp_policy(torch.tensor(state,device=DEVICE),
                              torch.tensor(cfs,device=DEVICE))
            top=sc.argsort(descending=True)[:imp_soft_topk].cpu().numpy()
            donors=np.stack([valid_emb_m[cset[j][0]] for j in top])
            q_t=torch.tensor(q_partial,dtype=torch.float32,device=DEVICE)
            d_t=torch.tensor(donors,   dtype=torch.float32,device=DEVICE)
            with torch.no_grad(): pred=head(q_t,d_t).cpu().numpy()
            errs.append(float(np.mean((pred-gt_emb_m)**2)))
        val_mse=float(np.mean(errs)) if errs else float("inf")
        if val_mse<best_mse: best_mse=val_mse; best_state=copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    return head


# =============================================================================
# IMPUTATION UTILS
# =============================================================================

def impute_one_sample(raw_mods_i,present_mask,stats,imp_policies,soft_heads,imp_soft_topk):
    row=[raw_mods_i[m].copy() for m in range(N_MODS)]
    for mod_idx,item in enumerate(imp_policies):
        if present_mask[mod_idx] or item is None: continue
        policy,imp_index,valid_idx,valid_emb_m,adj_imp=item
        head=soft_heads[mod_idx]
        pm=present_mask.copy()
        q_partial=build_partial_embedding(row,pm,stats,VARIANT)
        zero_m=np.zeros(MOD_DIMS[mod_idx],dtype=np.float32)
        _,kl=imp_index.kneighbors(zero_m.reshape(1,-1)); kl=kl[0]
        kd=np.linalg.norm(valid_emb_m[kl]-zero_m,axis=1)
        seeds=_get_seeds(zero_m,valid_emb_m,kl,kd)
        if not seeds: row[mod_idx]=valid_emb_m.mean(axis=0); continue
        rng=np.random.RandomState(SEED+8888+mod_idx)
        cset,_=_imp_bfs(seeds,valid_emb_m,adj_imp,rng=rng)
        if not cset: row[mod_idx]=valid_emb_m.mean(axis=0); continue
        state,cfs=_imp_features(q_partial,zero_m,cset,valid_emb_m)
        with torch.no_grad():
            sc=policy(torch.tensor(state,dtype=torch.float32,device=DEVICE),
                      torch.tensor(cfs,dtype=torch.float32,device=DEVICE))
        top=sc.argsort(descending=True)[:imp_soft_topk].cpu().numpy()
        donors=np.stack([valid_emb_m[cset[j][0]] for j in top])
        if USE_SOFT_FUSION and head is not None:
            q_t=torch.tensor(q_partial,dtype=torch.float32,device=DEVICE)
            d_t=torch.tensor(donors,dtype=torch.float32,device=DEVICE)
            head.eval()
            with torch.no_grad(): row[mod_idx]=head(q_t,d_t).cpu().numpy()
        else:
            row[mod_idx]=valid_emb_m[cset[int(sc.argmax().item())][0]].copy()
    return row


def build_imputed_emb(raw_list,present_mask_all,stats,imp_policies,soft_heads,imp_soft_topk):
    N=raw_list[0].shape[0]; out=[]
    for i in range(N):
        row=impute_one_sample([raw_list[m][i] for m in range(N_MODS)],
                              present_mask_all,stats,imp_policies,soft_heads,imp_soft_topk)
        out.append(build_partial_embedding(row,np.ones(N_MODS,dtype=bool),stats,VARIANT))
    return np.stack(out).astype(np.float32)


# =============================================================================
# ROLLOUT EVALUATION
# =============================================================================

def _make_libero_env(task,resolution=128):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl_file=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file)
    return OffScreenRenderEnv(bddl_file_name=bddl_file,
                              camera_heights=resolution,camera_widths=resolution)


def _replay_demo(env,init_state,actions,max_steps=MAX_STEPS):
    env.reset(); env.set_init_state(init_state)
    for t,action in enumerate(actions):
        if t>=max_steps: break
        _,_,done,info=env.step(action)
        if done or info.get("success",False): return True
    return False


def evaluate_rollouts(policy,head,train_demos,tr_emb,tr_labels,adj,tr_set_sizes,
                      task_suite,te_emb_cfg,test_demos,config_label,fusion_topk):
    idx=_build_knn_index(tr_emb); kd_all,ki_all=idx.kneighbors(te_emb_cfg)
    task_demos={}
    for qi,demo in enumerate(test_demos):
        task_demos.setdefault(demo["task_lang"],[]).append((qi,demo))
    all_successes=[]
    for task_id in range(task_suite.n_tasks):
        task=task_suite.get_task(task_id); task_lang=task.language
        tlist=task_demos.get(task_lang,[])
        if not tlist: continue
        try:
            env=_make_libero_env(task)
            init_states=task_suite.get_task_init_states(task_id)
        except Exception as e:
            print(f"    [WARN] task {task_id}: {e}"); continue
        task_successes=[]
        for seed_idx in range(N_ROLLOUT_SEEDS):
            rng=np.random.RandomState(SEED+seed_idx*1000+task_id)
            qidxs=rng.choice(len(tlist),size=N_ROLLOUTS_TASK,replace=True); ok=0
            for rollout_i,qi_local in enumerate(qidxs):
                qi,_=tlist[qi_local]
                best_tr_idx,_=_infer_one_fusion(
                    te_emb_cfg[qi],ki_all[qi],kd_all[qi],
                    tr_emb,tr_labels,adj,tr_set_sizes,policy,head,fusion_topk)
                actions=train_demos[best_tr_idx]["actions"]
                init_state=init_states[rollout_i%len(init_states)]
                try:
                    if _replay_demo(env,init_state,actions): ok+=1
                except Exception: pass
            task_successes.append(ok/N_ROLLOUTS_TASK)
        all_successes.extend(task_successes); env.close()
    mean_all=float(np.mean(all_successes)) if all_successes else 0.0
    std_all =float(np.std(all_successes))  if all_successes else 0.0
    return mean_all, std_all


# =============================================================================
# ONE FULL RUN FOR A GIVEN TOP-K VALUE
# =============================================================================

def run_one_topk(K, train_demos, val_demos, test_demos, task_suite,
                 tr_emb, tr_labels, va_emb, va_labels,
                 te_emb, te_labels, tr_raw, va_raw, te_raw, stats,
                 shared_imp_policies):
    """
    Train prediction RL (fixed, shared), then for this K:
      - train FusionHead          with fusion_topk = K
      - train SoftImputationHeads with imp_soft_topk = K
      - evaluate mask_0 and mask_1 separately
      - return (sr_mask0, sr_mask1)

    Note: the prediction RL policy itself does NOT depend on topk,
    so it is trained once outside this function and passed in via
    shared_imp_policies structure. Only the fusion heads and soft
    imputation heads are retrained per K.
    """
    print(f"\n{'#'*60}")
    print(f"# TOP-K = {K}")
    print(f"{'#'*60}")

    adj = _build_knn_graph(tr_emb)
    all_sets, all_oracles, set_sizes = build_all_sets(tr_emb, tr_labels, adj)

    # Prediction RL policy — train once (same for all K values)
    print(f"\n[K={K}] Training prediction RL …")
    policy = train_policy(tr_emb, tr_labels, all_sets, all_oracles,
                          va_emb, va_labels, adj, set_sizes)

    # Prediction FusionHead — depends on K
    print(f"[K={K}] Training prediction FusionHead (topk={K}) …")
    head = train_fusion(policy, tr_emb, tr_labels, all_sets,
                        va_emb, va_labels, adj, set_sizes, fusion_topk=K)

    # Imputation RL policies — do NOT depend on K, reuse shared ones
    imp_policies = shared_imp_policies

    # Soft imputation heads — depend on K, retrain
    soft_heads = []
    for mod_idx in range(N_MODS):
        if mod_idx == 2:
            soft_heads.append(None); continue
        policy_m, imp_index_m, valid_idx_m, valid_emb_m, adj_imp_m = imp_policies[mod_idx]
        print(f"[K={K}] Soft imputation head mod={MOD_NAMES[mod_idx]} (topk={K}) …")
        sh = train_soft_imp_head_one(
            mod_idx, policy_m, imp_index_m, valid_emb_m, adj_imp_m,
            tr_raw, va_raw, stats, imp_soft_topk=K)
        soft_heads.append(sh)

    # Evaluate each dropout config separately → two SR values
    sr_per_config = {}
    for config_label, cam0_present, cam1_present in DROPOUT_CONFIGS:
        present_mask = np.array([cam0_present, cam1_present, True], dtype=bool)
        print(f"\n[K={K}] Building imputed embeddings [{config_label}] …")
        tr_emb_cfg = build_imputed_emb(
            tr_raw, present_mask, stats, imp_policies, soft_heads, imp_soft_topk=K)
        te_emb_cfg = build_imputed_emb(
            te_raw, present_mask, stats, imp_policies, soft_heads, imp_soft_topk=K)

        adj_cfg = _build_knn_graph(tr_emb_cfg)
        sets_cfg, oracles_cfg, sizes_cfg = build_all_sets(tr_emb_cfg, tr_labels, adj_cfg)

        print(f"[K={K}] Retraining prediction policy on imputed space …")
        policy_cfg = train_policy(tr_emb_cfg, tr_labels, sets_cfg, oracles_cfg,
                                  va_emb, va_labels, adj_cfg, sizes_cfg)
        head_cfg   = train_fusion(policy_cfg, tr_emb_cfg, tr_labels, sets_cfg,
                                  va_emb, va_labels, adj_cfg, sizes_cfg, fusion_topk=K)

        print(f"[K={K}] Evaluating rollouts [{config_label}] …")
        mean_sr, std_sr = evaluate_rollouts(
            policy_cfg, head_cfg, train_demos,
            tr_emb_cfg, tr_labels, adj_cfg, sizes_cfg,
            task_suite, te_emb_cfg, test_demos, config_label,
            fusion_topk=K)
        print(f"  [{config_label}] SR = {mean_sr:.3f} ± {std_sr:.3f}")
        sr_per_config[config_label] = mean_sr

    sr_mask0 = sr_per_config.get("mask_0", 0.0)
    sr_mask1 = sr_per_config.get("mask_1", 0.0)
    print(f"\n[K={K}] mask_0={sr_mask0:.3f}  mask_1={sr_mask1:.3f}")
    return sr_mask0, sr_mask1


# =============================================================================
# MAIN
# =============================================================================

def main():
    import copy
    globals()['copy'] = copy

    print(f"Device   : {DEVICE}")
    print(f"Suite    : {LIBERO_SUITE}")
    print(f"Variant  : {VARIANT}  |  Strategy: {STRATEGY}")
    print(f"Soft Fusion: {USE_SOFT_FUSION}")
    print(f"TopK set : {TOPK_SET}")
    print(f"Epochs   : PPO={PPO_EPOCHS} Fusion={FUSION_EPOCHS} "
          f"ImpPPO={IMP_PPO_EPOCHS} ImpSoft={IMP_SOFT_EPOCHS}\n")

    # ── load data and extract embeddings once ─────────────────────────────────
    train_demos, val_demos, test_demos, task_suite = load_suite(
        DATASET_ROOT, LIBERO_SUITE)

    print("Loading CLIP encoder …")
    clip_enc = FrozenCLIPEncoder().to(DEVICE)
    (tr_emb, tr_labels, va_emb, va_labels, te_emb, te_labels,
     tr_raw, va_raw, te_raw, stats) = build_all_embeddings(
        clip_enc, train_demos, val_demos, test_demos)
    del clip_enc; torch.cuda.empty_cache()
    print("CLIP encoder freed.\n")

    # ── train imputation RL policies once (independent of topk) ───────────────
    print("=" * 60)
    print("Training imputation RL policies (shared across all TopK values)")
    print("=" * 60)
    shared_imp_policies = []
    for mod_idx in range(N_MODS):
        if mod_idx == 2:
            shared_imp_policies.append(None); continue
        print(f"\nImputation RL — {MOD_NAMES[mod_idx]} …")
        result = train_imp_policy_one(mod_idx, tr_raw, va_raw, stats)
        shared_imp_policies.append(result)

    # ── CSV filename ──────────────────────────────────────────────────────────
    sf_tag = "sf1" if USE_SOFT_FUSION else "sf0"
    csv_path = (f"{LIBERO_SUITE}_{VARIANT}_{STRATEGY}_{sf_tag}"
                f"_ppo{PPO_EPOCHS}_topk.csv")

    # ── TopK study loop ───────────────────────────────────────────────────────
    rows = []
    for K in TOPK_SET:
        sr_mask0, sr_mask1 = run_one_topk(
            K, train_demos, val_demos, test_demos, task_suite,
            tr_emb, tr_labels, va_emb, va_labels,
            te_emb, te_labels, tr_raw, va_raw, te_raw, stats,
            shared_imp_policies)
        rows.append((K, sr_mask0, sr_mask1))

        # save after each K so partial results are not lost
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["top_k", "sr_mask0_NoCam0", "sr_mask1_NoCam1"])
            writer.writerows(rows)
        print(f"\n  CSV updated → {csv_path}\n")

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*52)
    print(f"  TopK study complete — {LIBERO_SUITE}")
    print(f"  {'top_k':<8s}  {'SR NoCam0':>12s}  {'SR NoCam1':>12s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}")
    for K, s0, s1 in rows:
        print(f"  {K:<8d}  {s0:.3f}        {s1:.3f}")
    print("="*52)
    print(f"  Saved → {csv_path}\n")


if __name__ == "__main__":
    main()