import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import load_data
import model
import utils


def main():
    parser = argparse.ArgumentParser(description="Regenerate report figures from a saved training run.")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--model-path", default=str(ROOT / "output/final_model/model.npy"))
    parser.add_argument("--history-path", default=str(ROOT / "output/final_model/history.json"))
    parser.add_argument("--search-results-path", default=str(ROOT / "output/final_model/search_results.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "output/final_model/figure"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, _, _, _, _, _, _, image_shape = load_data.load_data(
        args.data_dir, image_size=args.image_size, seed=args.seed, normalize=False
    )
    clf = model.load_model(args.model_path)
    history = utils.load_json(args.history_path)
    utils.save_training_curves(history, args.output_dir)
    utils.save_first_layer_weights(clf, image_shape, Path(args.output_dir) / "first_layer_weights.png")
    search_results_path = Path(args.search_results_path)
    if search_results_path.exists():
        utils.save_search_curves(utils.load_json(search_results_path), args.output_dir)


if __name__ == "__main__":
    main()
