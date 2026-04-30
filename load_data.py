import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def to_onehot(y, num_classes):
    y = np.asarray(y, dtype=np.int64)
    onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    onehot[np.arange(len(y)), y] = 1.0
    return onehot


def _resize_filter():
    return getattr(Image, "Resampling", Image).BILINEAR


def _cache_path(data_dir, image_size, cache_dir):
    resolved = str(Path(data_dir).resolve())
    digest = hashlib.md5(resolved.encode("utf-8")).hexdigest()[:8]
    return Path(cache_dir) / f"eurosat_{digest}_{image_size}x{image_size}.npz"


def load_image_folder_dataset(path, image_size=32, cache=True, cache_dir="./output/cache"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    cache_file = _cache_path(path, image_size, cache_dir)
    if cache and cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["X"].astype(np.float32, copy=False),
            data["y"].astype(np.int64, copy=False),
            data["class_names"].tolist(),
            data["paths"].tolist(),
            tuple(data["image_shape"].tolist()),
        )

    class_names = sorted(
        d.name for d in path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not class_names:
        raise ValueError(f"No class folders found in: {path}")

    images = []
    labels = []
    paths = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = path / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.name.startswith(".") or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if img.size != (image_size, image_size):
                    img = img.resize((image_size, image_size), _resize_filter())
                arr = np.asarray(img, dtype=np.float32) / 255.0
            images.append(arr.reshape(-1))
            labels.append(class_to_idx[class_name])
            paths.append(str(image_path))

    if not images:
        raise ValueError(f"No image files found under: {path}")

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    image_shape = (image_size, image_size, 3)

    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_file,
            X=X,
            y=y,
            class_names=np.asarray(class_names),
            paths=np.asarray(paths),
            image_shape=np.asarray(image_shape, dtype=np.int64),
        )

    return X, y, class_names, paths, image_shape


def stratified_split(y, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    rng = np.random.default_rng(seed)
    train_idx, valid_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        train_end = int(len(idx) * train_ratio)
        valid_end = train_end + int(len(idx) * valid_ratio)
        train_idx.append(idx[:train_end])
        valid_idx.append(idx[train_end:valid_end])
        test_idx.append(idx[valid_end:])

    train_idx = np.concatenate(train_idx)
    valid_idx = np.concatenate(valid_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)
    return train_idx, valid_idx, test_idx


def split_data(X, y, paths=None, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    train_idx, valid_idx, test_idx = stratified_split(
        y, train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, seed=seed
    )
    parts = (
        X[train_idx], y[train_idx],
        X[valid_idx], y[valid_idx],
        X[test_idx], y[test_idx],
    )
    if paths is None:
        return parts
    paths = np.asarray(paths)
    return parts + (paths[train_idx].tolist(), paths[valid_idx].tolist(), paths[test_idx].tolist())


def standardize_splits(X_train, X_valid, X_test, eps=1e-6):
    mean = X_train.mean(axis=0, keepdims=True, dtype=np.float32)
    std = X_train.std(axis=0, keepdims=True, dtype=np.float32)
    std = np.maximum(std, eps)
    return (
        ((X_train - mean) / std).astype(np.float32, copy=False),
        ((X_valid - mean) / std).astype(np.float32, copy=False),
        ((X_test - mean) / std).astype(np.float32, copy=False),
    )


def load_data(
    path,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    image_size=32,
    cache=True,
    cache_dir="./output/cache",
    one_hot=True,
    return_paths=False,
    normalize=True,
):
    X, y_idx, class_names, paths, image_shape = load_image_folder_dataset(
        path, image_size=image_size, cache=cache, cache_dir=cache_dir
    )
    split = split_data(
        X, y_idx, paths=paths, train_ratio=train_ratio,
        valid_ratio=valid_ratio, test_ratio=test_ratio, seed=seed
    )

    X_train, y_train, X_valid, y_valid, X_test, y_test = split[:6]
    if normalize:
        X_train, X_valid, X_test = standardize_splits(X_train, X_valid, X_test)

    if one_hot:
        num_classes = len(class_names)
        y_train = to_onehot(y_train, num_classes)
        y_valid = to_onehot(y_valid, num_classes)
        y_test = to_onehot(y_test, num_classes)

    result = [X_train, y_train, X_valid, y_valid, X_test, y_test, class_names, image_shape]
    if return_paths:
        result.extend(split[6:])
    return tuple(result)
