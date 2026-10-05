# Supervised dynamic feature selection

PyTorch/Lightning experiments for supervised image feature selection with DDS.
A classifier learns from selected image features. An optional reconstruction stage
trains a separate decoder to measure how much of the original image can be
recovered from the classifier input.

## Installation

Use Python 3.10 or 3.11. The pinned dependencies preserve the original PyTorch 2.2
stack. Run from the project root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.run --config configs/mnist.json
```

Alternatively: `bash start.sh --config configs/mnist.json`.
The example downloads MNIST into `data/`. Weights & Biases runs offline by default
in the JSON runner; set `wandb_mode` to `online` to use a configured account.
Set `wandb_mode` to `disabled` for local CSV metrics without the W&B service.
The JSON runner supports MNIST, FashionMNIST and CIFAR10 with a UNet selector.
Existing dataset-specific scripts remain available under `dfs_*/execute/`.
Those legacy scripts manage their own tracking settings and dataset paths.

## Optional reconstruction

In `configs/mnist.json`, set:

```json
"reconstruction": {
  "enabled": true,
  "autoencoder": "unet",
  "epochs": 20,
  "log_images": true
}
```

Supported decoders: `unet`, `simple`, `residual`. Channel counts are inferred for
UNet. A null or omitted `epochs` uses `privacy_epochs` from `Execution`.
Reconstruction is disabled by default, so existing experiments retain their
classification-only behavior. Reconstruction loaders are only created when enabled.
Image logging requires a validation split; numerical reconstruction evaluation
also works without one. Choose `validation_size` to reserve validation samples.

The Python API accepts `reconstruction=True`, a dictionary as above, or a
`ReconstructionConfig` instance. Existing `privacy_model_module` and
`privacy_model_module_kwargs` remain supported; explicit model kwargs override
the configured decoder and inferred channels.

The selector remains in evaluation mode with frozen parameters and BatchNorm
statistics while the decoder trains. Its `selected_input(x)` method supplies the
same weighted/masked input as classification, including the bias when configured.
Custom selectors must implement this method and return `(selected, penalty, mask)`.
The augmentation model selects in original image coordinates during reconstruction.

Results retain the existing list-of-experiments, list-of-dictionaries structure.
When enabled, reconstruction adds `privacyCheck_test_loss` alongside
`classification_test_accuracy` and `classification_test_loss`.
Reconstruction loss is an empirical measurement, not a formal privacy guarantee.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use synthetic CPU data and local tracking. They cover configuration errors,
classification input consistency, frozen-selector gradients/statistics, and a
complete one-epoch pipeline with reconstruction enabled and disabled.

## Source layout

- `src/dfs.py`: feature selector and classifier.
- `src/reconstruction.py`: reconstruction configuration and execution.
- `src/privacy_model.py`: reconstruction decoder with a frozen selector.
- `src/main_class.py`: repeated experiments and cross-validation.
- `src/run.py`, `configs/`: configurable entry point and example.
- `src/dfs_augmentation.py`: experimental augmentation variant.

## Publication notes

`.gitignore` excludes downloaded datasets, weights, experiment images, tracking
runs, caches, local environments and editor settings. Existing local outputs are
preserved. Inspect staged files before publishing; ignore rules do not untrack
files already committed in another checkout.

This source folder arrived without `.git` metadata or a license. A rights holder
must choose the license before presenting it as open-source software. Dataset
licenses and access conditions remain separate. Legacy ImageNet scripts require
locally available data. Full benchmark reproduction requires running the original
training schedules; synthetic smoke tests do not validate reported paper results.
