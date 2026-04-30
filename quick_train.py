import argparse

import load_data
import model
import utils


def main():
    parser = argparse.ArgumentParser(description="Quick EuroSAT training run with a fast default configuration.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./output/final_model")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--lr-decay", type=float, default=0.94)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", default="256,128")
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    hidden = tuple(int(v) for v in args.hidden.split(","))
    X_train, y_train, X_valid, y_valid, _, _, class_names, image_shape = load_data.load_data(
        args.data_dir, image_size=args.image_size, seed=args.seed, normalize=not args.no_normalize
    )
    print("classes:", class_names)
    clf = model.Model(
        input_size=X_train.shape[1],
        output_size=len(class_names),
        hidden_layers=hidden,
        activation=args.activation,
        lambda_=args.weight_decay,
        seed=args.seed,
    )
    history = clf.train(
        X_train,
        y_train,
        X_valid,
        y_valid,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_decay=args.lr_decay,
        save=True,
        output_dir=args.output_dir,
    )
    utils.save_training_curves(history, f"{args.output_dir}/figure")
    utils.save_first_layer_weights(clf, image_shape, f"{args.output_dir}/figure/first_layer_weights.png")


if __name__ == "__main__":
    main()
