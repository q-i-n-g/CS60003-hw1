import os
from pathlib import Path

import numpy as np

import utils


def _progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


class Linear:
    def __init__(self, in_features, out_features, rng, activation="relu"):
        if activation == "relu":
            scale = np.sqrt(2.0 / in_features)
        else:
            scale = np.sqrt(1.0 / in_features)
        self.weights = (rng.standard_normal((in_features, out_features)) * scale).astype(np.float32)
        self.bias = np.zeros((1, out_features), dtype=np.float32)

    def forward(self, x):
        return x @ self.weights + self.bias


class Model:
    def __init__(
        self,
        hidden_layers=(256, 128),
        activation=("relu", "relu", "softmax"),
        input_size=32 * 32 * 3,
        output_size=10,
        lambda_=5e-4,
        seed=42,
    ):
        hidden_layers = tuple(int(v) for v in hidden_layers)
        if len(hidden_layers) != 2:
            raise ValueError("This homework model is a three-layer MLP: two hidden layers plus output layer.")
        if isinstance(activation, str):
            hidden_activation = (activation, activation)
        else:
            hidden_activation = tuple(activation[:2])
        if len(hidden_activation) != 2:
            raise ValueError("activation must be a string or contain at least two hidden-layer activations.")

        self.config = {
            "hidden_layers": hidden_layers,
            "activation": hidden_activation,
            "input_size": int(input_size),
            "output_size": int(output_size),
            "lambda_": float(lambda_),
            "seed": int(seed),
        }
        self.lambda_ = float(lambda_)
        self.rng = np.random.default_rng(seed)
        h1, h2 = hidden_layers
        self.layers = [
            Linear(input_size, h1, self.rng, hidden_activation[0]),
            Linear(h1, h2, self.rng, hidden_activation[1]),
            Linear(h2, output_size, self.rng, "linear"),
        ]
        self.activation = hidden_activation
        self.cache = {}

    def _activate(self, x, name):
        if name == "relu":
            return np.maximum(x, 0)
        if name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
        if name == "tanh":
            return np.tanh(x)
        raise ValueError(f"Unsupported activation: {name}")

    def _activation_grad(self, x, name):
        if name == "relu":
            return (x > 0).astype(np.float32)
        if name == "sigmoid":
            y = self._activate(x, name)
            return y * (1.0 - y)
        if name == "tanh":
            y = np.tanh(x)
            return 1.0 - y * y
        raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x, training=False):
        z1 = self.layers[0].forward(x)
        a1 = self._activate(z1, self.activation[0])
        z2 = self.layers[1].forward(a1)
        a2 = self._activate(z2, self.activation[1])
        logits = self.layers[2].forward(a2)
        if training:
            self.cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return logits

    def softmax(self, logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def loss(self, logits, y):
        y = y.astype(np.float32, copy=False)
        probs = self.softmax(logits)
        eps = 1e-8
        data_loss = -np.sum(y * np.log(np.clip(probs, eps, 1.0))) / len(y)
        l2 = 0.5 * self.lambda_ * sum(np.sum(layer.weights * layer.weights) for layer in self.layers)
        return float(data_loss + l2), probs

    def backward(self, probs, y):
        batch_size = len(y)
        dz3 = (probs - y) / batch_size
        a2 = self.cache["a2"]
        a1 = self.cache["a1"]
        x = self.cache["x"]

        grads = [None, None, None]
        grads[2] = {
            "weights": a2.T @ dz3 + self.lambda_ * self.layers[2].weights,
            "bias": np.sum(dz3, axis=0, keepdims=True),
        }

        da2 = dz3 @ self.layers[2].weights.T
        dz2 = da2 * self._activation_grad(self.cache["z2"], self.activation[1])
        grads[1] = {
            "weights": a1.T @ dz2 + self.lambda_ * self.layers[1].weights,
            "bias": np.sum(dz2, axis=0, keepdims=True),
        }

        da1 = dz2 @ self.layers[1].weights.T
        dz1 = da1 * self._activation_grad(self.cache["z1"], self.activation[0])
        grads[0] = {
            "weights": x.T @ dz1 + self.lambda_ * self.layers[0].weights,
            "bias": np.sum(dz1, axis=0, keepdims=True),
        }
        return grads

    def update(self, x, y, lr):
        logits = self.forward(x, training=True)
        loss, probs = self.loss(logits, y)
        grads = self.backward(probs, y)
        for layer, grad in zip(self.layers, grads):
            layer.weights -= lr * grad["weights"].astype(np.float32, copy=False)
            layer.bias -= lr * grad["bias"].astype(np.float32, copy=False)
        return loss

    def predict_proba(self, x, batch_size=2048):
        probs = []
        for start in range(0, len(x), batch_size):
            logits = self.forward(x[start:start + batch_size], training=False)
            probs.append(self.softmax(logits))
        return np.vstack(probs)

    def predict(self, x, batch_size=2048):
        return np.argmax(self.predict_proba(x, batch_size=batch_size), axis=1)

    def evaluate(self, x, y, batch_size=2048):
        losses = []
        preds = []
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]
            logits = self.forward(xb, training=False)
            loss, probs = self.loss(logits, yb)
            losses.append(loss)
            preds.append(np.argmax(probs, axis=1))
        preds = np.concatenate(preds)
        return float(np.mean(losses)), utils.accuracy(preds, y), preds

    def test(self, X, Y):
        _, acc, _ = self.evaluate(X, Y)
        return acc

    def para(self):
        return [
            {"weights": layer.weights.copy(), "bias": layer.bias.copy()}
            for layer in self.layers
        ]

    def set_para(self, para):
        for layer, saved in zip(self.layers, para):
            layer.weights = saved["weights"].astype(np.float32, copy=True)
            layer.bias = saved["bias"].astype(np.float32, copy=True)

    def save(self, path, extra=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"para": self.para(), "config": self.config}
        if extra:
            payload.update(extra)
        np.save(path, payload, allow_pickle=True)

    def train(
        self,
        X,
        Y,
        X_valid,
        Y_valid,
        lr=0.05,
        batch_size=256,
        epochs=20,
        lr_decay=0.95,
        min_lr=1e-5,
        eval_every=1,
        save=False,
        output_dir="./output/final_model",
        shuffle=True,
        show_progress=True,
        verbose=True,
        **legacy_kwargs,
    ):
        if "weight_decay" in legacy_kwargs:
            lr_decay = legacy_kwargs["weight_decay"]

        output_dir = Path(output_dir)
        history = {
            "steps": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }
        best_acc = -1.0
        best_para = self.para()
        step = 0
        indices = np.arange(len(X))

        epoch_iter = range(epochs)
        if show_progress:
            epoch_iter = _progress(epoch_iter, desc="train", unit="epoch", leave=False)

        for epoch in epoch_iter:
            current_lr = utils.exponential_lr(lr, epoch, decay=lr_decay, min_lr=min_lr)
            if shuffle:
                self.rng.shuffle(indices)
            for start in range(0, len(X), batch_size):
                step += 1
                batch_idx = indices[start:start + batch_size]
                loss = self.update(X[batch_idx], Y[batch_idx], current_lr)

            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                val_loss, val_acc, _ = self.evaluate(X_valid, Y_valid)
                history["steps"].append(step)
                history["train_loss"].append(float(loss))
                history["val_loss"].append(float(val_loss))
                history["val_acc"].append(float(val_acc))
                history["learning_rate"].append(float(current_lr))
                message = (
                    f"epoch {epoch + 1:03d}/{epochs} "
                    f"lr={current_lr:.6g} train_loss={loss:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )
                if show_progress and hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix(
                        lr=f"{current_lr:.3g}",
                        train_loss=f"{loss:.4f}",
                        val_acc=f"{val_acc:.4f}",
                    )
                elif verbose:
                    print(message)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_para = self.para()
                    if save:
                        self.save(output_dir / "model.npy", extra={"best_val_acc": best_acc})

        self.set_para(best_para)
        if save:
            os.makedirs(output_dir, exist_ok=True)
            self.save(output_dir / "model.npy", extra={"best_val_acc": best_acc})
            utils.save_json(history, output_dir / "history.json")
        return history


def load_model(model_path):
    payload = np.load(model_path, allow_pickle=True).item()
    config = payload["config"]
    model = Model(
        hidden_layers=tuple(config["hidden_layers"]),
        activation=tuple(config["activation"]),
        input_size=config["input_size"],
        output_size=config["output_size"],
        lambda_=config["lambda_"],
        seed=config.get("seed", 42),
    )
    model.set_para(payload["para"])
    return model
