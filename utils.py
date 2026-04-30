import json
import math
import os
from pathlib import Path

import numpy as np


def label_indices(y):
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    return y.astype(np.int64)


def accuracy(pred, label):
    return float(np.mean(np.asarray(pred) == label_indices(label)))


def confusion_matrix(y_true, y_pred, num_classes):
    y_true = label_indices(y_true)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def print_confusion_matrix(cm, class_names):
    width = max(9, max(len(name) for name in class_names) + 1)
    header = "true\\pred".ljust(width) + "".join(name[:8].rjust(9) for name in class_names)
    print(header)
    for name, row in zip(class_names, cm):
        print(name[:width - 1].ljust(width) + "".join(str(int(v)).rjust(9) for v in row))


def exponential_lr(initial_lr, epoch, decay=0.95, min_lr=1e-5):
    return max(min_lr, initial_lr * (decay ** epoch))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_training_curves(history, output_dir):
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = history.get("steps", list(range(1, len(history["train_loss"]) + 1)))

    plt.figure(figsize=(7, 4))
    plt.plot(steps, history["train_loss"], label="train loss")
    if history.get("val_loss"):
        plt.plot(steps[:len(history["val_loss"])], history["val_loss"], label="valid loss")
    plt.xlabel("update")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    if history.get("val_acc"):
        plt.figure(figsize=(7, 4))
        plt.plot(steps[:len(history["val_acc"])], history["val_acc"], label="valid accuracy")
        plt.xlabel("update")
        plt.ylabel("accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "val_accuracy_curve.png", dpi=160)
        plt.close()


def _set_plot_style(plt):
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")


def _hidden_label(value):
    if isinstance(value, str):
        return value
    return "x".join(str(int(v)) for v in value)


def _search_label(row, key):
    if key == "hidden":
        return _hidden_label(row[key])
    return f"{row[key]:g}"


def _plot_search_stage(ax, rows, key, title, color):
    xs = np.arange(len(rows))
    ys = np.asarray([row["val_acc"] for row in rows], dtype=np.float32)
    labels = [_search_label(row, key) for row in rows]
    best_idx = int(np.argmax(ys))

    ax.plot(xs, ys, color=color, marker="o", linewidth=2.4, markersize=6)
    ax.scatter([xs[best_idx]], [ys[best_idx]], s=90, color="#d62728", zorder=3, label="best")
    ax.annotate(
        f"{ys[best_idx]:.3f}",
        xy=(xs[best_idx], ys[best_idx]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="#d62728",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(key)
    ax.set_ylabel("validation accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ymin = max(0.0, float(ys.min()) - 0.03)
    ymax = min(1.0, float(ys.max()) + 0.03)
    if ymax - ymin < 0.08:
        center = float((ys.max() + ys.min()) / 2)
        ymin = max(0.0, center - 0.04)
        ymax = min(1.0, center + 0.04)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_search_curves(search_payload, output_dir):
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = search_payload.get("results", search_payload) if isinstance(search_payload, dict) else search_payload
    stages = [
        ("lr", "lr", "Learning Rate Search", "#1f77b4", "search_lr_curve.png"),
        ("hidden", "hidden", "Hidden Dimension Search", "#2ca02c", "search_hidden_curve.png"),
        ("weight_decay", "weight_decay", "Weight Decay Search", "#9467bd", "search_weight_decay_curve.png"),
    ]

    grouped = {
        stage: [row for row in results if row.get("stage") == stage]
        for stage, _, _, _, _ in stages
    }
    for stage, key, title, color, filename in stages:
        rows = grouped[stage]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        _plot_search_stage(ax, rows, key, title, color)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    if all(grouped[stage] for stage, _, _, _, _ in stages):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        for ax, (stage, key, title, color, _) in zip(axes, stages):
            _plot_search_stage(ax, grouped[stage], key, title, color)
        fig.tight_layout()
        fig.savefig(output_dir / "search_summary_curves.png", dpi=180)
        plt.close(fig)


def _normalise_image(arr):
    arr = arr.astype(np.float32, copy=False)
    low, high = np.percentile(arr, [1, 99])
    if high <= low:
        low, high = float(arr.min()), float(arr.max())
    if high <= low:
        return np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0, 1)


def save_first_layer_weights(model, image_shape, output_path, max_filters=64):
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    weights = model.layers[0].weights
    count = min(max_filters, weights.shape[1])
    cols = min(8, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i in range(count):
        filt = weights[:, i].reshape(image_shape)
        axes[i].imshow(_normalise_image(filt))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.2)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def save_error_examples(X, y_true, y_pred, class_names, image_shape, output_path, max_examples=16):
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    y_true_idx = label_indices(y_true)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    wrong = np.flatnonzero(y_true_idx != y_pred)
    if len(wrong) == 0:
        return 0

    chosen = wrong[:max_examples]
    cols = min(4, len(chosen))
    rows = int(math.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, chosen):
        ax.imshow(np.clip(X[idx].reshape(image_shape), 0, 1))
        ax.set_title(f"T: {class_names[y_true_idx[idx]]}\nP: {class_names[y_pred[idx]]}", fontsize=8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    return int(len(wrong))


def save_confusion_matrix(cm, class_names, output_path):
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
