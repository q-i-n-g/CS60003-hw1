# CV-pj1

NumPy implementation of a three-layer MLP classifier for EuroSAT RGB land-cover classification.

## Environment

```bash
pip install numpy pillow matplotlib tqdm
```

No PyTorch, TensorFlow, JAX, or automatic-differentiation framework is used.

## Data

Place the EuroSAT RGB class folders under `./data`, for example:

```text
data/
  AnnualCrop/
  Forest/
  ...
  SeaLake/
```

The loader performs a stratified 70/15/15 train/validation/test split. By default, images are resized to `32x32`, cached under `output/cache/`, and standardized with train-split mean/std for faster MLP optimization.

## Quick Training

```bash
python quick_train.py --data-dir ./data --output-dir ./output/final_model
```

## Hyperparameter Search + Final Training

```bash
python train.py --data-dir ./data --output-dir ./output/final_model --search
```

Default search uses a sequential strategy instead of a full Cartesian grid. It starts from the normal defaults, searches one hyperparameter at a time, and passes the best value to the next stage. This gives 16 short search runs by default:

- learning rate: `0.2`, `0.12`, `0.08`, `0.05`, `0.03`, `0.01`
- hidden dimensions: `128,64`, `256,128`, `512,128`, `512,256`, `1024,512`
- weight decay: `0`, `0.00001`, `0.0001`, `0.0005`, `0.001`

The order is learning rate first, hidden dimensions second, weight decay last.

Useful speed/quality knobs:

```bash
python train.py --data-dir ./data --output-dir ./output/final_model \
  --image-size 32 --batch-size 512 --epochs 30 \
  --hidden 256,128 --activation relu --lr 0.08 \
  --lr-decay 0.94 --weight-decay 0.0005
```

If validation accuracy is still rising near the end, use a larger model and slower decay:

```bash
python train.py --data-dir ./data --output-dir ./output/final_model \
  --image-size 32 --batch-size 256 --epochs 80 \
  --hidden 1024,512 --activation relu --lr 0.05 \
  --lr-decay 0.98 --weight-decay 0.0001
```

Use `--image-size 64` if the report must use original-resolution MLP inputs. This is much slower.

## Test

```bash
python test.py --data-dir ./data --model-path ./output/final_model/model.npy --output-dir ./output/final_model
```

The test script prints accuracy and the confusion matrix, and saves `confusion_matrix.png` and `error_examples.png`.

## Figures for Report

```bash
python plot/plot.py --data-dir ./data --model-path ./output/final_model/model.npy \
  --history-path ./output/final_model/history.json --output-dir ./output/final_model/figure

python plot/plot_image.py --data-dir ./data --output-path ./output/data/examples.png
```

Recommended figures for the homework report:

- Dataset examples: `output/data/examples.png`
- Training and validation loss curves: `output/final_model/figure/loss_curve.png`
- Validation accuracy curve: `output/final_model/figure/val_accuracy_curve.png`
- Hyperparameter search curves:
  - `output/final_model/figure/search_lr_curve.png`
  - `output/final_model/figure/search_hidden_curve.png`
  - `output/final_model/figure/search_weight_decay_curve.png`
  - `output/final_model/figure/search_summary_curves.png`
- First-layer weight visualization: `output/final_model/figure/first_layer_weights.png`
- Test confusion matrix: `output/final_model/figure/confusion_matrix.png`
- Error analysis examples: `output/final_model/figure/error_examples.png`

Training saves:

- `output/final_model/model.npy`: best validation model weights.
- `output/final_model/history.json`: train loss, validation loss, validation accuracy, learning rate.
- `output/final_model/search_results.json`: grid search results when `--search` is used.
- `output/final_model/figure/loss_curve.png`
- `output/final_model/figure/val_accuracy_curve.png`
- `output/final_model/figure/first_layer_weights.png`
- `output/final_model/figure/search_lr_curve.png`
- `output/final_model/figure/search_hidden_curve.png`
- `output/final_model/figure/search_weight_decay_curve.png`
