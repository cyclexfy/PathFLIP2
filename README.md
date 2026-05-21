# PathFLIP

PathFLIP is a pathology vision-language learning project for whole-slide image representation learning, image-text alignment, retrieval, classification, and VLM fine-tuning.

## Repository Layout

```text
src/
  dataset/        Dataset and datamodule definitions
  model/          PathFLIP, PathFLIP-VLM, LongNet/TorchScale modules, losses
  tools/          Training entry points and argument parsers
  eval/           Retrieval, classification, and VQA evaluation
scripts/          Reproducible shell entry points
datasets/         Lightweight split files and data instructions
```

## Setup

Create an environment with Python 3.10 or newer, then install dependencies:

```bash
pip install -r requirements.txt
```

Install the correct PyTorch build for your CUDA version if the default `pip` build is not appropriate for your machine.

## Data

Large JSON training files, WSI feature files, slides, checkpoints, and generated processing outputs are not tracked in Git. Put required dataset files under `datasets/`, or pass paths explicitly:

```bash
python -m src.tools.train_align \
  --train_data_path datasets/SlideInstruct_train_stage1_caption_fine_grained.json \
  --val_data_path datasets/SlideBench-Caption-TCGA-plus.json
```

See `datasets/README.md` for expected filenames.

## Training

Alignment training:

```bash
bash scripts/train_align.sh
```

VLM stage 1:

```bash
ALIGN_MODEL_CKPT_PATH=outputs/pathflip/checkpoint/pytorch_model.bin \
bash scripts/train_vlm_stage1.sh
```

VLM stage 2:

```bash
ALIGN_MODEL_CKPT_PATH=outputs/pathflip/checkpoint/pytorch_model.bin \
STAGE1_CKPT_PATH=outputs/pathflip_vlm_stage1/checkpoint/stage1_projector.bin \
bash scripts/train_vlm_stage2.sh
```

All scripts use environment variables for local paths so machine-specific paths do not need to be committed.

## Evaluation

Zero-shot classification:

```bash
ARGS_PATH=outputs/pathflip/lightning_logs/version_0/hparams.yaml \
CKPT_PATH=outputs/pathflip/checkpoint/pytorch_model.bin \
bash scripts/eval_zero_shot_cls.sh
```

Few-shot and retrieval scripts are available in `scripts/` and can be configured with `ARGS_PATH`, `CKPT_PATH`, `DATASETS`, and related environment variables.

## GitHub Hygiene

The repository is configured to ignore local caches, notebooks, checkpoints, WSI files, generated outputs, and large JSON datasets. Keep private paths, credentials, raw slides, and patient-sensitive data outside the repository.
