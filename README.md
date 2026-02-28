# DeRaDiff — Denoising Time Realignment of Diffusion Models (ICLR 2026)

https://arxiv.org/abs/2601.20198

Official implementation of **DeRaDiff**, a test-time realignment method for diffusion models. DeRaDiff performs regularized realignment at inference by interpolating between a base diffusion model and a reward-aligned anchor model during the denoising process, controlled by a hyperparameter λ. The framework supports both Stable Diffusion 1.5 and Stable Diffusion XL, and evaluates generated images using CLIP Score, HPS, and PickScore metrics.

## Quick Start

### 1. Environment Setup

```bash
conda create -n deradiff python=3.9 -y
conda activate deradiff
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Usage

Place your anchor model checkpoint in the `checkpoints/` directory.

**Stable Diffusion 1.5:**

```bash
python scripts/eval_sd15.py \
    --anchor_model_ckpt_path ./checkpoints/anchor_model_ckpt \
    --Lambda 1.0 \
    --Beta 1000 \
    --output_dir ./results/sd15_eval \
    --save_images
```

**Stable Diffusion XL:**

```bash
python scripts/eval_sdxl.py \
    --anchor_model_ckpt_path ./checkpoints/anchor_model_ckpt \
    --Lambda 1.0 \
    --Beta 1000 \
    --output_dir ./results/sdxl_eval \
    --save_images
```

| Argument | Description |
|---|---|
| `--anchor_model_ckpt_path` | Path to the aligned anchor model checkpoint |
| `--Lambda` | Realignment strength; approximate regularization is β/λ (default: `1.0`) |
| `--Beta` | Anchor model regularization strength (default: `1000`) |
| `--output_dir` | Directory for evaluation results and images |
| `--save_images` | Save generated images to disk |
| `--cuda_id` | CUDA device ID (default: `0`) |

## Project Structure

```
DeRaDiff/
├── README.md
├── requirements.txt
├── setup.py
├── checkpoints/             # Anchor model checkpoints
├── data/
│   └── test_prompts.yaml    # 631 evaluation prompts
├── scripts/
│   ├── eval_sd15.py         # Entry point: SD 1.5 evaluation
│   └── eval_sdxl.py         # Entry point: SDXL evaluation
└── src/
    ├── config/
    │   ├── sd15_config.py   # SD 1.5 hyperparameters
    │   └── sdxl_config.py   # SDXL hyperparameters
    ├── eval/
    │   └── evaluator.py     # Multi-metric evaluator (CLIP, HPS, PickScore)
    ├── pipeline/
    │   ├── custom_sd15_pipeline.py   # Custom SD 1.5 denoising pipeline
    │   └── custom_sdxl_pipeline.py   # Custom SDXL denoising pipeline
    └── utils/
        ├── aes_utils.py     # Aesthetic score utilities
        ├── clip_utils.py    # CLIP score computation
        ├── gen_utils.py     # Image generation helper
        ├── hps_utils.py     # HPS score computation
        ├── load_utils.py    # YAML / checkpoint loading
        └── pickscore_utils.py  # PickScore computation
```
