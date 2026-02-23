# z-image-train

Standalone CLI for LoRA fine-tuning, image generation, and model format conversion for [Z-Image](https://github.com/holg/z-image-burn) — built entirely in Rust on the [Burn](https://github.com/tracel-ai/burn) deep learning framework.

All models, training, and inference run natively through Burn — no Python, no PyTorch, no ONNX runtime. The entire pipeline from text encoding (Qwen3) to diffusion (Z-Image) to VAE decoding is implemented as Burn modules, using Burn's autodiff engine for gradient computation and its WGPU/Metal/Vulkan backends for GPU acceleration.

## Features

- **Pure Burn Stack** — End-to-end Rust/Burn pipeline: training, inference, and model I/O without external ML runtimes
- **LoRA Training** — Fine-tune Z-Image models using Low-Rank Adaptation with flow matching
- **Image Generation** — Generate images from text prompts using base model + LoRA weights
- **Model Conversion** — Convert weights between safetensors and Burn's native `.bpk` format
- **Cross-platform GPU** — Burn backends for Metal (macOS), Vulkan (Linux/Windows), and WGPU
- **Pre-computed Caching** — Image latents and text embeddings are cached upfront via Burn's forward passes to speed up training and reduce VRAM usage
- **Checkpoint Saving** — Optionally save intermediate LoRA checkpoints during training

## Requirements

- **Rust nightly** (edition 2024)
- GPU recommended (Metal on macOS, Vulkan on Linux/Windows)

Install nightly:

```sh
rustup install nightly
rustup default nightly
```

## Build

```sh
# Default (WGPU backend):
cargo +nightly build --release

# macOS with Metal GPU:
cargo +nightly build --release --features metal

# Linux/Windows with Vulkan GPU:
cargo +nightly build --release --features vulkan
```

## Usage

### `train` — Fine-tune with LoRA

Train a LoRA adapter on a dataset of images with text captions.

```sh
z-image-train train \
  --model-dir /path/to/models \
  --dataset /path/to/data \
  --output lora.safetensors \
  --rank 32 \
  --alpha 32 \
  --epochs 50 \
  --lr 2e-4 \
  --resolution 512 \
  --save-every 100
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | *(required)* | Directory containing model weights and tokenizer |
| `--dataset` | *(required)* | Directory with training images + `.txt` caption files |
| `--output` | `lora_weights.safetensors` | Output path for LoRA weights |
| `--rank` | `16` | LoRA rank (low-rank dimension) |
| `--alpha` | `16.0` | LoRA alpha scaling factor |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--resolution` | `256` | Training image resolution (images resized to this) |
| `--save-every` | `0` | Save checkpoint every N steps (0 = only at end) |

#### Dataset format

Place images and matching caption files in a directory:

```
dataset/
├── photo1.png
├── photo1.txt    # caption for photo1.png
├── photo2.jpg
├── photo2.txt
└── ...
```

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.webp`. Each image must have a corresponding `.txt` file with a non-empty caption.

#### Model directory

The model directory should contain:

| File | Description |
|------|-------------|
| `tokenizer.json` or `qwen3-tokenizer.json` | Qwen3 tokenizer |
| `qwen3_4b_text_encoder.{bpk,safetensors}` | Qwen3 4B text encoder |
| `z_image_turbo_f16.{bpk,safetensors}` | Z-Image transformer (f16 — recommended for WGPU) |
| `z_image_turbo_bf16.{bpk,safetensors}` | Z-Image transformer (bf16 — requires Metal/Vulkan) |
| `ae.{bpk,safetensors}` | Autoencoder (VAE) |

`.bpk` files are preferred if both formats are present.

#### Training output

- `{output}.safetensors` — LoRA weight tensors (A/B matrices only)
- `{output}.lora.json` — LoRA configuration sidecar
- `{output}.step{N}.safetensors` — Intermediate checkpoints (if `--save-every > 0`)

### `generate` — Generate images with LoRA

```sh
z-image-train generate \
  --model-dir /path/to/models \
  --lora lora.safetensors \
  --prompt "a fluffy cat wearing sunglasses" \
  --output cat.png \
  --width 768 \
  --height 768 \
  --steps 20
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | *(required)* | Directory containing model weights |
| `--lora` | *(required)* | Path to LoRA weights (`.safetensors`) |
| `--prompt` | *(required)* | Text prompt for generation |
| `--output` | `output.png` | Output image path |
| `--width` | `512` | Image width (must be multiple of 16) |
| `--height` | `512` | Image height (must be multiple of 16) |
| `--steps` | `8` | Number of inference steps |

Uses flow matching with Euler discrete scheduling (1000 timesteps, mu=3.0).

### `convert` — Convert model formats and dtypes

Convert weights between `.safetensors` and `.bpk` formats, and optionally cast to a different precision (e.g. bf16 → f32 for WGPU compatibility).

```sh
# safetensors → bpk
z-image-train convert \
  --input model.safetensors \
  --output model.bpk \
  --model-type transformer

# Convert bf16 weights to f32 (needed for WGPU/WGSL backends on AMD/Linux)
z-image-train convert \
  --input z_image_turbo_bf16.bpk \
  --output z_image_turbo_f32.bpk \
  --model-type transformer \
  --dtype f32

# bpk → safetensors
z-image-train convert \
  --input ae.bpk \
  --output ae.safetensors \
  --model-type autoencoder \
  --overwrite
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *(required)* | Input model file |
| `--output` | *(required)* | Output model file |
| `--model-type` | `transformer` | Model type: `transformer` (or `t`), `autoencoder` (or `ae`, `vae`), `text-encoder` (or `te`) |
| `--dtype` | *(preserve)* | Cast weights to: `f32`, `f16`, or `bf16` |
| `--overwrite` | `false` | Overwrite existing output file |

> **Note:** WGSL (the WGPU shader language) does not support bf16. If you're on a platform without Metal or working Vulkan (e.g. AMD on Linux), convert bf16 models to f16 or f32 using `--dtype f16` or `--dtype f32`.

### `inspect` — Check tensor dtypes in .bpk files

Inspect a `.bpk` model file to verify the actual dtype of its tensors. Useful to confirm whether a model is truly f16, bf16, or f32 before loading it on a backend that may not support all dtypes.

```sh
# Show first 20 tensors with their shapes and dtypes
z-image-train inspect --input model.bpk

# Show all tensors
z-image-train inspect --input model.bpk --limit 0

# Quick dtype check (just look at the summary at the bottom)
z-image-train inspect --input z_image_turbo_f16.bpk --limit 1
```

Example output:

```
Inspecting: z_image_turbo_f16.bpk
Total tensors: 453

  [   0] cap_embedder_0.gamma                                [2560]  F16
  [   1] cap_embedder_1.bias                                 [3840]  F16
  ...

--- Dtype Summary ---
  F16: 453 tensors
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *(required)* | Path to the `.bpk` file to inspect |
| `--limit` | `20` | Max number of tensors to display (0 = all) |

## Architecture

Everything runs through Burn's tensor engine and module system:

- **Burn WGPU backend** for GPU compute, with Metal and Vulkan feature flags
- **Burn autodiff** for automatic differentiation during LoRA training
- **Burn module system** — all models (Z-Image transformer, Qwen3 text encoder, VAE) are Burn `Module` implementations
- **Burn record format** — native `.bpk` serialization alongside safetensors support
- **Qwen3 4B** text encoder (Burn module) for prompt encoding
- **Flow matching** training with velocity prediction and MSE loss
- **AdamW** optimizer (Burn) with weight decay (1e-2)
- LoRA targets attention, feed-forward, and refiner layers
- Image preprocessing uses Lanczos3 filtering, normalized to [-1, 1]

## Project structure

```
z-image-train/
├── Cargo.toml
├── src/
│   ├── main.rs        # CLI entry point
│   ├── train.rs       # Training pipeline
│   ├── dataset.rs     # Dataset loading
│   ├── convert.rs     # Format conversion
│   └── lora_io.rs     # LoRA save/load
```

## License

See repository for license details.
