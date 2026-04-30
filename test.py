import argparse
from pathlib import Path

import load_data
import model
import utils


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved NumPy MLP on the EuroSAT test split.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-path", default="./output/final_model/model.npy")
    parser.add_argument("--output-dir", default="./output/final_model")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-errors", type=int, default=16)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names, image_shape = load_data.load_data(
        args.data_dir, image_size=args.image_size, seed=args.seed, normalize=not args.no_normalize
    )
    _, _, _, _, X_test_raw, _, _, _ = load_data.load_data(
        args.data_dir, image_size=args.image_size, seed=args.seed, normalize=False
    )
    clf = model.load_model(args.model_path)

    test_loss, test_acc, test_pred = clf.evaluate(X_test, y_test, batch_size=args.batch_size)
    train_acc = clf.test(X_train, y_train)
    valid_acc = clf.test(X_valid, y_valid)
    cm = utils.confusion_matrix(y_test, test_pred, len(class_names))

    print("classes:", class_names)
    print(f"test loss: {test_loss:.4f}")
    print(f"test accuracy: {test_acc:.4f}")
    print(f"train accuracy: {train_acc:.4f}")
    print(f"valid accuracy: {valid_acc:.4f}")
    utils.print_confusion_matrix(cm, class_names)

    figure_dir = Path(args.output_dir) / "figure"
    utils.save_confusion_matrix(cm, class_names, figure_dir / "confusion_matrix.png")
    wrong_count = utils.save_error_examples(
        X_test_raw,
        y_test,
        test_pred,
        class_names,
        image_shape,
        figure_dir / "error_examples.png",
        max_examples=args.max_errors,
    )
    print(f"wrong test examples: {wrong_count}")


if __name__ == "__main__":
    main()
