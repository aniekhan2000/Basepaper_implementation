"""
FLGuardian (MNIST): layer-wise clustering defense on real updates from MNIST.

What this script does
---------------------
- Loads MNIST train/test.
- Partitions train set across M clients (Dirichlet non-IID).
- Per round: each client trains locally on its shard for one epoch -> compute parameter deltas.
- (Optional) Malicious clients perform label-flip training to poison their updates.
- Runs FLGuardian on the client deltas.
- Aggregates selected updates, applies to the global model, and evaluates test accuracy.
- Reports detection metrics (accuracy, precision, recall/ADR, F1) and global test accuracy.
"""

from typing import List, Tuple, Dict, Set, Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import copy, random

# ---------------- Seed handling (added) ----------------
import sys

# default seed (will be overridden if passed as --seed=<value>)
seed = 42
for arg in sys.argv:
    if arg.startswith("--seed="):
        try:
            seed = int(arg.split("=")[1])
        except:
            pass

# Set seeds globally
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print(f"✅ Using random seed: {seed}")
# ------------------------------------------------------


# ===== Try to import sklearn for KMeans (optional) and torchvision for MNIST =====
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import torchvision
    import torchvision.transforms as T
    TV_AVAILABLE = True
except Exception as e:
    TV_AVAILABLE = False
    raise RuntimeError("torchvision is required for MNIST. Install with: pip install torchvision") from e


# ----------------------------- Utilities ---------------------------------

def flatten_layer_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().float().reshape(-1)

def pairwise_cosine_distances(vectors: List[torch.Tensor]) -> np.ndarray:
    N = len(vectors)
    if N == 0:
        return np.zeros((0, 0), dtype=np.float32)
    M = torch.stack([flatten_layer_tensor(v) for v in vectors])
    norms = torch.norm(M, dim=1, keepdim=True) + 1e-12
    Mn = M / norms
    sim = (Mn @ Mn.t()).double().cpu().numpy()
    return (1.0 - sim).astype(np.float32, copy=False)

def pairwise_euclidean_distances(vectors: List[torch.Tensor]) -> np.ndarray:
    N = len(vectors)
    if N == 0:
        return np.zeros((0, 0), dtype=np.float32)
    M = torch.stack([flatten_layer_tensor(v) for v in vectors]).cpu().numpy().astype(np.float64)
    sq = np.sum(M * M, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * (M @ M.T), 0.0)
    return np.sqrt(d2, dtype=np.float64).astype(np.float32, copy=False)

def two_means_1d(values: np.ndarray, iters: int = 25) -> np.ndarray:
    assert values.ndim == 1
    c1, c2 = values.min(), values.max()
    labels = np.zeros_like(values, dtype=np.int32)
    for _ in range(iters):
        d1, d2 = np.abs(values - c1), np.abs(values - c2)
        new_labels = (d2 < d1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        if (labels == 1).any(): c2 = values[labels == 1].mean()
        if (labels == 0).any(): c1 = values[labels == 0].mean()
    return labels

def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_mean = x.mean(axis=1, keepdims=True)
    row_std = x.std(axis=1, keepdims=True) + eps
    return (x - row_mean) / row_std

def metrics_from_sets(gt_mal: Set[int], pred_mal: Set[int], num_clients: int) -> Dict[str, float]:
    gt = np.zeros(num_clients, dtype=np.int32)
    pr = np.zeros(num_clients, dtype=np.int32)
    for i in gt_mal:
        if 0 <= i < num_clients:
            gt[i] = 1
    for i in pred_mal:
        if 0 <= i < num_clients:
            pr[i] = 1

    TP = int(np.sum((gt == 1) & (pr == 1)))
    FP = int(np.sum((gt == 0) & (pr == 1)))
    TN = int(np.sum((gt == 0) & (pr == 0)))
    FN = int(np.sum((gt == 1) & (pr == 0)))

    denom_acc = TP + TN + FP + FN
    acc = (TP + TN) / denom_acc if denom_acc > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return dict(
        accuracy=acc, precision=prec, recall=rec, attack_detection_rate=rec,
        f1=f1, TP=float(TP), FP=float(FP), TN=float(TN), FN=float(FN)
    )

# ----------------------------- Model -------------------------------------

class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --------------- Parameter helpers ------------------

def params_to_list(model: nn.Module) -> List[torch.Tensor]:
    return [p.detach().cpu().clone() for p in model.parameters()]

def apply_update_to_model(model: nn.Module, update: List[torch.Tensor], lr: float = 1.0) -> None:
    with torch.no_grad():
        for p, u in zip(model.parameters(), update):
            p.data.add_(u.to(p.data.device, p.data.dtype), alpha=lr)

def delta_from_local(global_params: List[torch.Tensor], local_model: nn.Module) -> List[torch.Tensor]:
    local_params = [p.detach().cpu().clone() for p in local_model.parameters()]
    return [lp - gp for lp, gp in zip(local_params, global_params)]

# ----------------------- FLGuardian core ---------------------------------

class FLGuardian:
    def __init__(self, model_prototype: List[torch.Tensor],
                 beta: float = 1.0, top_k: Optional[int] = None,
                 combine: Literal["intersection", "union"] = "intersection",
                 min_benign_layers: int = 2):
        self.layer_count = len(model_prototype)
        self.layer_shapes = [p.shape for p in model_prototype]
        raw = np.array([beta ** l for l in range(self.layer_count)], dtype=np.float64)
        self.layer_weights = (raw / (raw.sum() + 1e-12)).astype(np.float32)
        self.top_k = top_k
        self.combine = combine
        self.min_benign_layers = int(min_benign_layers)

    def _extract_layer_updates(self, client_updates, layer_idx):
        return [copy.deepcopy(upd[layer_idx]).detach() for upd in client_updates]

    def _kmeans_2cluster_labels(self, values_1d):
        values_1d = np.asarray(values_1d).reshape(-1, 1).astype(np.float64)
        if SKLEARN_AVAILABLE:
            km = KMeans(n_clusters=2, n_init=10, random_state=0)
            return km.fit_predict(values_1d)
        return two_means_1d(values_1d.ravel())

    def _larger_cluster_members(self, labels):
        ids0, ids1 = set(np.where(labels == 0)[0]), set(np.where(labels == 1)[0])
        return ids0 if len(ids0) >= len(ids1) else ids1

    def _get_candidate_benign_set_for_layer(self, layer_cosine, layer_euclid):
        N = layer_cosine.shape[0]
        if N == 0:
            return set()
        Xc = normalize_rows(layer_cosine.astype(np.float64))
        Xe = normalize_rows(layer_euclid.astype(np.float64))
        vec_c, vec_e = Xc.sum(axis=1), Xe.sum(axis=1)
        labels_c, labels_e = self._kmeans_2cluster_labels(vec_c), self._kmeans_2cluster_labels(vec_e)
        cand_c, cand_e = self._larger_cluster_members(labels_c), self._larger_cluster_members(labels_e)
        return cand_c.union(cand_e) if self.combine == "union" else cand_c.intersection(cand_e)

    def detect_layerwise_benign_sets(self, client_updates):
        benign_sets = []
        for l in range(self.layer_count):
            layer_vecs = self._extract_layer_updates(client_updates, l)
            cos_mat = pairwise_cosine_distances(layer_vecs)
            euc_mat = pairwise_euclidean_distances(layer_vecs)
            benign_sets.append(self._get_candidate_benign_set_for_layer(cos_mat, euc_mat))
        return benign_sets

    def compute_trust_scores(self, benign_sets, num_clients):
        scores = np.zeros(num_clients, dtype=np.float32)
        for l, s in enumerate(benign_sets):
            for idx in s:
                if 0 <= idx < num_clients:
                    scores[int(idx)] += self.layer_weights[l]
        return scores

    def select_topk(self, scores, num_clients):
        if self.top_k is None:
            selected = [i for i in range(num_clients) if scores[i] > 0]
            return selected if selected else list(range(num_clients))
        k = min(self.top_k, num_clients)
        return list(map(int, np.argsort(-scores)[:k]))

    def aggregate_mean(self, client_updates, selected):
        if len(selected) == 0:
            return [torch.zeros(s) for s in self.layer_shapes]
        agg = []
        for l in range(self.layer_count):
            stack = torch.stack([client_updates[i][l].detach().cpu() for i in selected], dim=0)
            agg.append(stack.mean(dim=0))
        return agg

    def predict_malicious(self, benign_sets, num_clients):
        counts = np.zeros(num_clients, dtype=np.int32)
        for s in benign_sets:
            for idx in s:
                if 0 <= idx < num_clients:
                    counts[idx] += 1
        return {i for i in range(num_clients) if counts[i] < self.min_benign_layers}

    def run_one_round(self, client_updates):
        M = len(client_updates)
        benign_sets = self.detect_layerwise_benign_sets(client_updates)
        scores = self.compute_trust_scores(benign_sets, M)
        selected = self.select_topk(scores, M)
        agg = self.aggregate_mean(client_updates, selected)
        diag = dict(benign_sets=benign_sets, scores=scores,
                    selected_clients=selected, layer_weights=self.layer_weights.tolist(),
                    combine_rule=self.combine)
        return agg, diag

# --------------------- MNIST data & client partitioning -------------------

def build_mnist_loaders(batch_size=64, num_clients=10, dirichlet_alpha=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    y = np.array(train_ds.targets)
    n_classes, idxs = 10, np.arange(len(train_ds))
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        class_idx = idxs[y == c]
        rng.shuffle(class_idx)
        props = rng.dirichlet(alpha=[dirichlet_alpha] * num_clients)
        splits = (np.cumsum(props) * len(class_idx)).astype(int)[:-1]
        shards = np.split(class_idx, splits)
        for i in range(num_clients):
            client_indices[i].extend(shards[i].tolist())
    client_loaders = {i: DataLoader(Subset(train_ds, client_indices[i]), batch_size=batch_size, shuffle=True, generator=g)
                      for i in range(num_clients)}
    return client_loaders, DataLoader(test_ds, batch_size=256, shuffle=False)

# -------------------------- Local train & eval ----------------------------

def train_one_epoch(model, loader, device, poison=False):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        if poison:
            target = (target + 1) % 10
        opt.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        opt.step()

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = F.cross_entropy(logits, target, reduction="sum")
            loss_sum += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / max(1, total), loss_sum / max(1, total)

# ----------------------- Multi-round Simulation ---------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True, floatmode="maxprec_equal")
    # torch.manual_seed(0)
    # np.random.seed(0)
    # random.seed(0)

    M, R, TOP_K, DIRICHLET_ALPHA = 15, 25, 5, 0.5
    BATCH_SIZE, LOCAL_POISON = 64, True
    MALICIOUS_IDS = {1, 7, 5}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, test_loader = build_mnist_loaders(BATCH_SIZE, M, DIRICHLET_ALPHA, seed=seed)
    global_model = MNISTMLP().to(device)

    guardian = FLGuardian(params_to_list(global_model), beta=2.0, top_k=TOP_K,
                          combine="intersection", min_benign_layers=2)

    hist = {k: [] for k in ["accuracy","precision","recall","f1","TP","FP","TN","FN","test_acc"]}
    init_acc, init_loss = evaluate(global_model, test_loader, device)
    print(f"Initial global model - Test Acc: {init_acc:.4f}, Loss: {init_loss:.4f}")

    for r in range(R):
        print(f"\n================= Round {r+1}/{R} =================")
        global_params_cpu = params_to_list(global_model)
        client_updates = []
        for cid in range(M):
            local = MNISTMLP().to(device)
            local.load_state_dict(global_model.state_dict(), strict=True)
            poison = (cid in MALICIOUS_IDS) and LOCAL_POISON
            train_one_epoch(local, client_loaders[cid], device, poison=poison)
            delta = delta_from_local(global_params_cpu, local)
            client_updates.append(delta)

        agg_update, diag = guardian.run_one_round(client_updates)
        pred_mal = guardian.predict_malicious(diag["benign_sets"], M)
        m = metrics_from_sets(MALICIOUS_IDS, pred_mal, M)
        for k in hist.keys():
            if k in m: hist[k].append(m[k])

        apply_update_to_model(global_model, agg_update, lr=1.0)
        test_acc, _ = evaluate(global_model, test_loader, device)
        hist["test_acc"].append(test_acc)

        print(f"Layer weights: {diag['layer_weights']}")
        print(f"Trust scores: {diag['scores']}")
        print(f"Selected clients: {[int(i) for i in diag['selected_clients']]}")
        print(f"Ground-truth malicious: {sorted(MALICIOUS_IDS)}")
        print(f"Predicted malicious: {sorted(int(i) for i in pred_mal)}")
        for l, s in enumerate(diag["benign_sets"]):
            print(f"Param {l} benign clients: {sorted(int(x) for x in s)}")
        print(f"Detection - Acc={m['accuracy']:.3f}, Prec={m['precision']:.3f}, "
              f"Rec/ADR={m['recall']:.3f}, F1={m['f1']:.3f} "
              f"(TP={int(m['TP'])}, FP={int(m['FP'])}, TN={int(m['TN'])}, FN={int(m['FN'])})")
        print(f"Global - Test Acc: {test_acc:.4f}")

    avg = lambda x: float(np.mean(x)) if len(x) else 0.0
    print("\n================= Overall (average over rounds) =================")
    print(f"Accuracy: {avg(hist['accuracy']):.3f}")
    print(f"Precision: {avg(hist['precision']):.3f}")
    print(f"Recall / Attack Detection Rate: {avg(hist['recall']):.3f}")
    print(f"F1 score: {avg(hist['f1']):.3f}")
    print(f"Avg TP: {avg(hist['TP']):.2f}, FP: {avg(hist['FP']):.2f}, TN: {avg(hist['TN']):.2f}, FN: {avg(hist['FN']):.2f}")
    print(f"Avg Global Test Acc: {avg(hist['test_acc']):.4f}")
