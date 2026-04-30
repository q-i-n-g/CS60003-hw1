import argparse
import os
from pathlib import Path

import load_data
import model
import utils


def _progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def parse_hidden(value):
    if isinstance(value, tuple):
        return value
    parts = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("hidden dims must look like 256,128")
    return tuple(parts)


def build_parser():
    parser = argparse.ArgumentParser(description="Train/search a NumPy three-layer MLP on EuroSAT RGB.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./output/final_model")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--lr-decay", type=float, default=0.94)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=parse_hidden, default=(256, 128))
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--search", action="store_true", help="Run grid search before final training.")
    parser.add_argument("--search-epochs", type=int, default=8)
    parser.add_argument("--search-lrs", default="0.2,0.12,0.08,0.05,0.03,0.01")
    parser.add_argument("--search-hidden", default="128,64;256,128;512,128;512,256;1024,512")
    parser.add_argument("--search-weight-decay", default="0,0.00001,0.0001,0.0005,0.001")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser


def load_splits(args):
    return load_data.load_data(
        args.data_dir,
        image_size=args.image_size,
        seed=args.seed,
        cache=not args.no_cache,
        one_hot=True,
        normalize=not args.no_normalize,
    )


def train_one(args, X_train, y_train, X_valid, y_valid, class_names, image_shape, output_dir):
    clf = model.Model(
        hidden_layers=args.hidden,
        activation=args.activation,
        input_size=X_train.shape[1],
        output_size=len(class_names),
        lambda_=args.weight_decay,
        seed=args.seed,
    )
    history = clf.train(
        X_train,
        y_train,
        X_valid,
        y_valid,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_decay=args.lr_decay,
        min_lr=args.min_lr,
        eval_every=args.eval_every,
        save=True,
        output_dir=output_dir,
    )
    utils.save_training_curves(history, Path(output_dir) / "figure")
    utils.save_first_layer_weights(clf, image_shape, Path(output_dir) / "figure" / "first_layer_weights.png")
    utils.save_json(
        {
            "class_names": class_names,
            "image_shape": image_shape,
            "args": vars(args),
            "best_val_acc": max(history["val_acc"]) if history["val_acc"] else None,
        },
        Path(output_dir) / "metadata.json",
    )
    return clf, history


def _float_list(text):
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _hidden_list(text):
    return [parse_hidden(item) for item in text.split(";") if item.strip()]


def _run_search_trial(args, X_train, y_train, X_valid, y_valid, class_names, lr, hidden, weight_decay, stage):
    print(f"search[{stage}] lr={lr} hidden={hidden} weight_decay={weight_decay}")
    clf = model.Model(
        hidden_layers=hidden,
        activation=args.activation,
        input_size=X_train.shape[1],
        output_size=len(class_names),
        lambda_=weight_decay,
        seed=args.seed,
    )
    history = clf.train(
        X_train,
        y_train,
        X_valid,
        y_valid,
        lr=lr,
        batch_size=args.batch_size,
        epochs=args.search_epochs,
        lr_decay=args.lr_decay,
        min_lr=args.min_lr,
        eval_every=max(1, args.search_epochs),
        save=False,
        show_progress=False,
        verbose=False,
    )
    return {
        "stage": stage,
        "lr": lr,
        "hidden": hidden,
        "weight_decay": weight_decay,
        "val_acc": history["val_acc"][-1],
    }


def grid_search(args, X_train, y_train, X_valid, y_valid, class_names):
    lrs = _float_list(args.search_lrs)
    hiddens = _hidden_list(args.search_hidden)
    decays = _float_list(args.search_weight_decay)
    results = []

    current_lr = args.lr
    current_hidden = args.hidden
    current_weight_decay = args.weight_decay

    lr_results = []
    for lr in _progress(lrs, desc="search lr", unit="trial"):
        lr_results.append(
            _run_search_trial(
                args, X_train, y_train, X_valid, y_valid, class_names,
                lr=lr, hidden=current_hidden, weight_decay=current_weight_decay, stage="lr"
            )
        )
    results.extend(lr_results)
    current_lr = max(lr_results, key=lambda item: item["val_acc"])["lr"]
    print(f"best lr after stage 1: {current_lr}")

    hidden_results = []
    for hidden in _progress(hiddens, desc="search hidden", unit="trial"):
        hidden_results.append(
            _run_search_trial(
                args, X_train, y_train, X_valid, y_valid, class_names,
                lr=current_lr, hidden=hidden, weight_decay=current_weight_decay, stage="hidden"
            )
        )
    results.extend(hidden_results)
    current_hidden = max(hidden_results, key=lambda item: item["val_acc"])["hidden"]
    print(f"best hidden after stage 2: {current_hidden}")

    decay_results = []
    for weight_decay in _progress(decays, desc="search weight_decay", unit="trial"):
        decay_results.append(
            _run_search_trial(
                args, X_train, y_train, X_valid, y_valid, class_names,
                lr=current_lr, hidden=current_hidden, weight_decay=weight_decay, stage="weight_decay"
            )
        )
    results.extend(decay_results)
    current_weight_decay = max(decay_results, key=lambda item: item["val_acc"])["weight_decay"]
    print(f"best weight_decay after stage 3: {current_weight_decay}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    serialisable = [
        {**r, "hidden": list(r["hidden"])}
        for r in results
    ]
    best = {
        "lr": current_lr,
        "hidden": current_hidden,
        "weight_decay": current_weight_decay,
    }
    utils.save_json(
        {"strategy": "sequential", "best": {**best, "hidden": list(current_hidden)}, "results": serialisable},
        out_dir / "search_results.json",
    )

    figure_dir = out_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    utils.save_search_curves(
        {"strategy": "sequential", "best": {**best, "hidden": list(current_hidden)}, "results": serialisable},
        figure_dir,
    )

    args.lr = current_lr
    args.hidden = current_hidden
    args.weight_decay = current_weight_decay
    print(f"best search config: lr={args.lr}, hidden={args.hidden}, weight_decay={args.weight_decay}")
    return best


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names, image_shape = load_splits(args)
    print("classes:", class_names)
    print("split shapes:", X_train.shape, X_valid.shape, X_test.shape)

    if args.search:
        grid_search(args, X_train, y_train, X_valid, y_valid, class_names)

    train_one(args, X_train, y_train, X_valid, y_valid, class_names, image_shape, args.output_dir)


if __name__ == "__main__":
    main()
