"""
CSCI 611 Assignment 2 — CNN on CIFAR-10 (Task 1 + Task 2).
Run: python run_cnn.py
Configure: edit the CONFIG dict below.
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============== CONFIG (edit these) ==============
CONFIG = {
    "seed": 42,
    "data_dir": "./data",
    "out_dir": "outputs",
    "batch_size": 64,
    "epochs": 20,
    "val_frac": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 0,
    # Task 2A
    "num_feature_maps": 8,
    # Task 2B: layer index in model.features, filter channel indices, activation = "mean" or "max"
    "task2b_layer_index": 8,
    "task2b_filters": [3, 11, 29],
    "task2b_activation": "mean",
    "task2b_top_k": 5,
}
# =================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def step1_setup(cfg):
    """Seeds, device, output directory."""
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print("Device:", device)
    return device


def step2_dataloaders(cfg):
    """CIFAR-10 train/val/test with transforms and loaders."""
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_full = datasets.CIFAR10(root=cfg["data_dir"], train=True, download=True, transform=train_tfms)
    test_ds = datasets.CIFAR10(root=cfg["data_dir"], train=False, download=True, transform=test_tfms)
    classes = train_full.classes

    val_size = int(len(train_full) * cfg["val_frac"])
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    print("Train/val/test samples:", len(train_ds), len(val_ds), len(test_ds))
    return train_loader, val_loader, test_loader, test_ds, classes


def step3_model(device):
    """CNN: 4 conv layers, 2 pool, BatchNorm, dropout, FC head."""
    class SimpleCIFARCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.18),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.32),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model = SimpleCIFARCNN().to(device)
    print(model)
    return model


def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        total_acc += accuracy_from_logits(logits, y)
        n += 1
    return total_loss / n, total_acc / n


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_acc = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n += 1
    return total_loss / n, total_acc / n


def step4_train(cfg, device, model, train_loader, val_loader):
    """Training loop; save best by val accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:02d}/{cfg['epochs']} | train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | val loss {va_loss:.4f} acc {va_acc*100:.2f}%")

    model.load_state_dict(best_state)
    model.to(device)
    ckpt_path = os.path.join(cfg["out_dir"], "best_model.pt")
    torch.save({"model_state": model.state_dict(), "best_val_acc": best_val_acc, "history": history}, ckpt_path)
    print("Saved", ckpt_path, "| Best val acc:", best_val_acc * 100)
    return history


def step5_curves_and_test(cfg, history, model, test_loader, criterion, device):
    """Plot loss curves and compute test accuracy."""
    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(cfg["out_dir"], "loss_curves.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print("Saved", loss_path)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"TEST loss: {test_loss:.4f} | TEST acc: {test_acc*100:.2f}%")


def step6_task2a(cfg, device, model, test_ds, classes):
    """First conv feature maps: 3 images from 3 classes, 8 channels each."""
    inv_norm = transforms.Normalize(
        mean=[-m / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
        std=[1 / s for s in CIFAR10_STD],
    )

    def show_img(ax, img_t, title=None):
        img = inv_norm(img_t.cpu()).clamp(0, 1)
        ax.imshow(img.permute(1, 2, 0))
        if title:
            ax.set_title(title)
        ax.axis("off")

    @torch.no_grad()
    def get_first_conv_maps(model, x_one):
        model.eval()
        return model.features[2](model.features[1](model.features[0](x_one)))

    chosen = []
    seen = set()
    for i in range(len(test_ds)):
        x, y = test_ds[i]
        if y not in seen:
            chosen.append((x, y, i))
            seen.add(y)
        if len(chosen) == 3:
            break

    n_maps = cfg["num_feature_maps"]
    for idx, (x, y, i) in enumerate(chosen, start=1):
        fmap = get_first_conv_maps(model, x.unsqueeze(0).to(device)).squeeze(0).cpu()
        fig, axes = plt.subplots(1, n_maps + 1, figsize=(12, 3))
        show_img(axes[0], x, title=f"Input\n{classes[y]}")
        for k in range(n_maps):
            axes[k + 1].imshow(fmap[k], cmap="viridis")
            axes[k + 1].set_title(f"ch{k}")
            axes[k + 1].axis("off")
        fig.suptitle(f"Conv1 feature maps | test idx {i}", y=1.05)
        plt.tight_layout()
        path = os.path.join(cfg["out_dir"], f"task2A_featuremaps_img{idx}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved", path)


def step7_task2b(cfg, device, model, test_loader, test_ds, classes):
    """Maximally activating images: one layer, 3 filters, top-k per filter."""
    activations = {}
    layer = model.features[cfg["task2b_layer_index"]]
    hook_handle = layer.register_forward_hook(lambda m, inp, out: activations.update({"feat": out.detach()}))

    @torch.no_grad()
    def score_batch(feat, filter_idx, mode):
        fmap = F.relu(feat)[:, filter_idx, :, :]
        return fmap.mean(dim=(1, 2)) if mode == "mean" else fmap.amax(dim=(1, 2))

    @torch.no_grad()
    def find_topk(model, loader, filter_indices, k, mode):
        model.eval()
        best = {fi: [] for fi in filter_indices}
        for x, y in loader:
            x = x.to(device)
            _ = model(x)
            feat = activations["feat"]
            for fi in filter_indices:
                scores = score_batch(feat, fi, mode).cpu().numpy()
                for j in range(len(scores)):
                    best[fi].append((float(scores[j]), x[j].cpu(), int(y[j])))
        for fi in filter_indices:
            best[fi] = sorted(best[fi], key=lambda t: t[0], reverse=True)[:k]
        return best

    filters = cfg["task2b_filters"]
    mode = cfg["task2b_activation"]
    k = cfg["task2b_top_k"]
    topk = find_topk(model, test_loader, filters, k, mode)
    hook_handle.remove()

    inv_norm = transforms.Normalize(
        mean=[-m / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
        std=[1 / s for s in CIFAR10_STD],
    )

    def show_img(ax, img_t, title=None):
        img = inv_norm(img_t.cpu()).clamp(0, 1)
        ax.imshow(img.permute(1, 2, 0))
        if title:
            ax.set_title(title)
        ax.axis("off")

    for fi in filters:
        fig, axes = plt.subplots(1, k, figsize=(12, 3))
        for i, (score, img_t, y) in enumerate(topk[fi]):
            show_img(axes[i], img_t, title=f"{classes[y]}\n{mode}={score:.3f}")
        fig.suptitle(f"Top-{k} by {mode} | features[{cfg['task2b_layer_index']}] filter {fi}", y=1.05)
        plt.tight_layout()
        path = os.path.join(cfg["out_dir"], f"task2B_top5_filter{fi}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved", path)


def main():
    p = argparse.ArgumentParser(description="CSCI 611 Assignment 2 — CNN on CIFAR-10")
    p.add_argument("--data_dir", default="./data", help="CIFAR-10 root")
    p.add_argument("--out_dir", default="outputs", help="Output directory")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    cfg = CONFIG.copy()
    cfg["data_dir"] = args.data_dir
    cfg["out_dir"] = args.out_dir
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["seed"] = args.seed

    print("Step 1: Setup")
    device = step1_setup(cfg)

    print("\nStep 2: Data")
    train_loader, val_loader, test_loader, test_ds, classes = step2_dataloaders(cfg)

    print("\nStep 3: Model")
    model = step3_model(device)

    print("\nStep 4: Training")
    history = step4_train(cfg, device, model, train_loader, val_loader)

    print("\nStep 5: Loss curves + test eval")
    criterion = nn.CrossEntropyLoss()
    step5_curves_and_test(cfg, history, model, test_loader, criterion, device)

    print("\nStep 6: Task 2A — First conv feature maps")
    step6_task2a(cfg, device, model, test_ds, classes)

    print("\nStep 7: Task 2B — Maximally activating images")
    step7_task2b(cfg, device, model, test_loader, test_ds, classes)

    print("\nDone. Outputs in", cfg["out_dir"])


if __name__ == "__main__":
    main()
