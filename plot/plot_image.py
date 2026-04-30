import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import load_data


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Save a EuroSAT sample grid for the report.")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-path", default=str(ROOT / "output/data/examples.png"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train, _, _, _, _, class_names, image_shape = load_data.load_data(
        args.data_dir, image_size=args.image_size, seed=args.seed, normalize=False
    )
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X_train), size=min(36, len(X_train)), replace=False)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 6, figsize=(8, 8))
    for ax, sample_idx in zip(axes.ravel(), idx):
        ax.imshow(np.clip(X_train[sample_idx].reshape(image_shape), 0, 1))
        ax.set_title(class_names[int(np.argmax(y_train[sample_idx]))], fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
