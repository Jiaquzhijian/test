# gradient_tv_kl_experiment.py
# -*- coding: utf-8 -*-
import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)  # last layer

    def forward(self, x, return_feat=False):
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7
        x = x.flatten(1)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        if return_feat:
            return logits, feat
        return logits


def train_model(model, dataset, device, epochs=5, batch_size=128, lr=1e-3, num_workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        print(f"[train] epoch={ep} loss={total_loss/total:.4f} acc={correct/total:.4f}")
    return model


def select_well_trained_subset(model, dataset, n, device, batch_size=256, num_workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    records = []  # (loss, local_idx)
    base = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses = F.cross_entropy(logits, y, reduction="none")
            pred = logits.argmax(dim=1)
            bsz = x.size(0)
            for i in range(bsz):
                if pred[i].item() == y[i].item():
                    records.append((losses[i].item(), base + i))
            base += bsz
    records.sort(key=lambda t: t[0])  # low loss -> "well-trained"
    if len(records) < n:
        raise RuntimeError(f"correctly classified samples ({len(records)}) < subset size ({n})")
    return [idx for _, idx in records[:n]]


def extract_features(model, dataset, device, batch_size=256, num_workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, f = model(x, return_feat=True)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def build_similarity_pool(model, dataset, anchor_indices, pool_size, device, batch_size=256, num_workers=0):
    feats = extract_features(model, dataset, device, batch_size=batch_size, num_workers=num_workers)
    anchor = feats[anchor_indices].mean(axis=0, keepdims=True)
    dists = np.linalg.norm(feats - anchor, axis=1)

    mask = np.ones(len(dataset), dtype=bool)
    mask[np.array(anchor_indices)] = False
    cand_idx = np.where(mask)[0]
    cand_d = dists[cand_idx]
    order = np.argsort(cand_d)  # near -> far
    sorted_idx = cand_idx[order].tolist()
    return sorted_idx[:pool_size]


def per_sample_last_layer_grads(model, dataset, indices, device):
    model.eval()
    grads = []
    for idx in indices:
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y)], dtype=torch.long, device=device)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        w = model.fc2.weight.grad.detach().flatten()
        b = model.fc2.bias.grad.detach().flatten()
        g = torch.cat([w, b], dim=0).cpu().numpy()
        grads.append(g)
    return np.stack(grads, axis=0)  # [N, D]


def fit_pca_1d(X):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    basis = vh[0]  # [D]
    return mu.squeeze(0), basis


def project_1d(X, mu, basis):
    return (X - mu[None, :]) @ basis


def hist_prob(z, bins, zmin, zmax, eps=1e-12):
    h, _ = np.histogram(z, bins=bins, range=(zmin, zmax), density=False)
    p = h.astype(np.float64) + eps
    p /= p.sum()
    return p


def tv_distance(p, q):
    return 0.5 * np.abs(p - q).sum()


def kl_div(p, q, eps=1e-12):
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


@dataclass
class CurveResult:
    tv_curve: np.ndarray
    G_A: np.ndarray
    G_S: np.ndarray


def compute_tv_curve(model, dataset, A_idx, S_idx_sorted, n, stride, bins, device):
    G_A = per_sample_last_layer_grads(model, dataset, A_idx, device)
    G_S = per_sample_last_layer_grads(model, dataset, S_idx_sorted, device)

    X = np.concatenate([G_A, G_S], axis=0)
    mu, basis = fit_pca_1d(X)
    z_all = project_1d(X, mu, basis)
    zmin, zmax = np.percentile(z_all, [1, 99])
    if zmax <= zmin:
        zmax = zmin + 1e-6

    zA = project_1d(G_A, mu, basis)
    pA = hist_prob(zA, bins=bins, zmin=zmin, zmax=zmax)

    vals = []
    for s in range(0, len(S_idx_sorted) - n + 1, stride):
        zB = project_1d(G_S[s:s + n], mu, basis)
        pB = hist_prob(zB, bins=bins, zmin=zmin, zmax=zmax)
        vals.append(tv_distance(pA, pB))
    return CurveResult(tv_curve=np.array(vals), G_A=G_A, G_S=G_S)


def kl_between_anchor_sets(G_A0, G_A1, bins):
    X = np.concatenate([G_A0, G_A1], axis=0)
    mu, basis = fit_pca_1d(X)
    z = project_1d(X, mu, basis)
    zmin, zmax = np.percentile(z, [1, 99])
    if zmax <= zmin:
        zmax = zmin + 1e-6

    p0 = hist_prob(project_1d(G_A0, mu, basis), bins=bins, zmin=zmin, zmax=zmax)
    p1 = hist_prob(project_1d(G_A1, mu, basis), bins=bins, zmin=zmin, zmax=zmax)
    return kl_div(p0, p1), kl_div(p1, p0)


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST(root=args.data_root, train=True, download=True, transform=tfm)

    N = len(full_train)
    if not (0 < args.d0_size < N):
        raise ValueError(f"d0_size must be in (0, {N})")
    perm = np.random.permutation(N)
    d0_ids = perm[:args.d0_size].tolist()
    new_ids = perm[args.d0_size:].tolist()

    D0 = Subset(full_train, d0_ids)
    D_new = Subset(full_train, new_ids)
    D_all = ConcatDataset([D0, D_new])

    print("== Train theta0 on D0 ==")
    model0 = SmallCNN()
    train_model(model0, D0, device, epochs=args.epochs0, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers)

    print("== Select A (well-trained subset) and S (similar pool) on D0 with theta0 ==")
    A_idx = select_well_trained_subset(model0, D0, args.subset_size, device, batch_size=args.batch_size, num_workers=args.num_workers)
    S_idx = build_similarity_pool(model0, D0, A_idx, args.pool_size, device, batch_size=args.batch_size, num_workers=args.num_workers)
    if len(S_idx) < args.subset_size:
        raise RuntimeError("pool_size too small: S must be >= subset_size")

    print("== TV curve with theta0 ==")
    res0 = compute_tv_curve(
        model0, D0, A_idx, S_idx,
        n=args.subset_size, stride=args.stride, bins=args.bins, device=device
    )

    print("== Train theta1 on D0 + D_new ==")
    model1 = SmallCNN()
    train_model(model1, D_all, device, epochs=args.epochs1, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers)

    print("== TV curve with theta1 (same A, same S) ==")
    res1 = compute_tv_curve(
        model1, D0, A_idx, S_idx,
        n=args.subset_size, stride=args.stride, bins=args.bins, device=device
    )

    print("== KL drift of anchor gradients under different params ==")
    kl01, kl10 = kl_between_anchor_sets(res0.G_A, res1.G_A, bins=args.bins)
    print(f"KL(P_A^theta0 || P_A^theta1) = {kl01:.6f}")
    print(f"KL(P_A^theta1 || P_A^theta0) = {kl10:.6f}")

    np.savetxt(os.path.join(args.output_dir, "tv_curve_theta0.csv"), res0.tv_curve, delimiter=",")
    np.savetxt(os.path.join(args.output_dir, "tv_curve_theta1.csv"), res1.tv_curve, delimiter=",")

    plt.figure(figsize=(8, 4))
    plt.plot(res0.tv_curve, label="TV curve (theta0)")
    plt.plot(res1.tv_curve, label="TV curve (theta1)")
    plt.xlabel("window index")
    plt.ylabel("TV distance")
    plt.title("Stability Curve (TV) Before/After Adding Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "tv_curve_compare.png"), dpi=160)
    plt.close()

    with open(os.path.join(args.output_dir, "kl_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"KL(P_A^theta0 || P_A^theta1) = {kl01:.6f}\n")
        f.write(f"KL(P_A^theta1 || P_A^theta0) = {kl10:.6f}\n")

    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--d0_size", type=int, default=50000)   # initial data size
    parser.add_argument("--subset_size", type=int, default=128) # |A| and |B_t|
    parser.add_argument("--pool_size", type=int, default=2048)  # |S|
    parser.add_argument("--stride", type=int, default=64)       # window stride
    parser.add_argument("--bins", type=int, default=40)

    parser.add_argument("--epochs0", type=int, default=5)
    parser.add_argument("--epochs1", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    main(args)
