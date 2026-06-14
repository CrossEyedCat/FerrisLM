# FerrisLM 🦀

DeepSeek-V2-style decoder-only language model in Rust with Candle + CUDA.

> **FerrisLM** — MLA + MoE LLM training and inference on GPU.

- **MLA** (Multi-head Latent Attention) with decoupled RoPE
- **DeepSeekMoE** (shared + routed experts, top-k routing, auxiliary balance loss)
- **SwiGLU** FFN, **RMSNorm**, weight tying
- **[Candle](https://github.com/huggingface/candle)** autograd with **CUDA** backend
- Training and text generation examples

## Prerequisites (CUDA)

- NVIDIA GPU + driver (CUDA 13.2+ PTX requires driver **≥595.45** or newer)
- [CUDA Toolkit 13.2](https://developer.nvidia.com/cuda-downloads) (13.x; CCCL requires `/Zc:preprocessor` with MSVC 2026)
- **Visual Studio 2026** with the **MSVC x64/x86 build tools** component

Before building with CUDA on Windows, activate the MSVC + CUDA environment:

```powershell
. .\scripts\setup-msvc-cuda.ps1
cargo build --release --features cuda
```

The setup script configures:

- MSVC: `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\...\cl.exe`
- CUDA: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`

Optional: cuDNN for faster matmul (`--features cudnn`).

## Build

```bash
# GPU (default)
cargo build --release --features cuda

# CPU-only fallback (no CUDA required)
cargo build --release --features cpu --no-default-features
```

## Quick start

```bash
# Prepare tokenized data
cargo run --release --example prepare_data -- \
  --corpus data/corpus.txt \
  --output data/tokens.bin

# Train (micro preset)
cargo run --release --features cuda --example train -- \
  --config configs/micro.toml \
  --data data/tokens.bin \
  --steps 100 \
  --device cuda:0

# Train on WikiText-103 (download corpus first, then tokenize)
cargo run --release --example prepare_data -- \
  --corpus data/wikitext103.txt \
  --output data/wikitext103_tokens.bin

cargo run --release --features cuda --example train -- \
  --config configs/wikitext_micro.toml \
  --data data/wikitext103_tokens.bin \
  --steps 200 \
  --checkpoint-dir checkpoints/wikitext \
  --device cuda:0

# Generate (optional checkpoint)
cargo run --release --features cuda --example generate -- \
  --config configs/wikitext_micro.toml \
  --corpus data/wikitext103.txt \
  --prompt "The history of" \
  --checkpoint checkpoints/wikitext/checkpoint_000200.safetensors \
  --device cuda:0
```

## Config presets

| Preset | File | ~Params |
|--------|------|---------|
| micro | `configs/micro.toml` | ~5M |
| wikitext_micro | `configs/wikitext_micro.toml` | ~5M |
| wikitext_micro_long | `configs/wikitext_micro_long.toml` | ~5M (50k steps) |
| wikitext_tiny | `configs/wikitext_tiny.toml` | ~100M |
| tiny | `configs/tiny.toml` | ~100M |
| small | `configs/small.toml` | ~450M |
| medium | `configs/medium.toml` | ~900M |

Config fields `dtype` (`f32`, `bf16`) and `use_flash_attn` (reserved) are supported in TOML.

## Checkpoints

Checkpoints are saved as **safetensors** + JSON metadata:

- `checkpoint_000050.safetensors` — named weights (VarMap)
- `checkpoint_000050.meta.json` — step, lr, config path, dtype

**Breaking change:** old flat `.bin` checkpoints from the CPU/ndarray version are not compatible.

Large datasets and checkpoints are excluded from git (see `.gitignore`). Download WikiText-103 locally or use `prepare_data` on your own corpus.

## Tests

```bash
cargo test --features cpu --no-default-features
# or with CUDA:
cargo test --features cuda
```

## Architecture

Each transformer block (except layer 0 FFN) uses MLA + DeepSeekMoE. Layer 0 uses a dense SwiGLU FFN per the DeepSeek-V2 design.

See `configs/` for hyperparameters.

## License

MIT
