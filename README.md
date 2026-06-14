# FerrisLM 🦀

DeepSeek-V2-style decoder-only language model in Rust with Candle + CUDA:

> **FerrisLM** — MLA + MoE LLM training and inference on GPU.

- **MLA** (Multi-head Latent Attention) with decoupled RoPE
- **DeepSeekMoE** (shared + routed experts, top-k routing, auxiliary balance loss)
- **SwiGLU** FFN, **RMSNorm**, weight tying
- **[Candle](https://github.com/huggingface/candle)** autograd with **CUDA** backend
- Training and text generation examples

## Prerequisites (CUDA)

- NVIDIA GPU + driver
- [CUDA Toolkit 13.2](https://developer.nvidia.com/cuda-downloads) (13.x; CCCL требует `/Zc:preprocessor` с MSVC 2026)
- **Visual Studio 2026** с компонентом **MSVC x64/x86 build tools**

Перед сборкой с CUDA активируйте окружение MSVC + CUDA:

```powershell
. .\scripts\setup-msvc-cuda.ps1
cargo build --release --features cuda
```

Скрипт использует:
- MSVC: `C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.51.36231\bin\HostX64\x64\cl.exe`
- CUDA: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`

Optional: cuDNN for extra matmul speed (`--features cudnn`).

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
cargo run --release --example prepare_data -- --corpus data/corpus.txt --output data/tokens.bin

# Train (micro preset, CPU or CUDA)
cargo run --release --example train -- \
  --config configs/micro.toml \
  --data data/tokens.bin \
  --steps 100 \
  --device cuda:0

# Generate (optional checkpoint)
cargo run --release --example generate -- \
  --config configs/micro.toml \
  --corpus data/corpus.txt \
  --prompt "hello world" \
  --checkpoint checkpoints/checkpoint_000050.safetensors
```

## Config presets

| Preset | File | ~Params |
|--------|------|---------|
| micro | `configs/micro.toml` | ~5M |
| tiny | `configs/tiny.toml` | ~100M |
| small | `configs/small.toml` | ~450M |
| medium | `configs/medium.toml` | ~900M |

Config fields `dtype` (`f32`, `bf16`) and `use_flash_attn` (reserved) are supported in TOML.

## Checkpoints

Checkpoints are saved as **safetensors** + JSON metadata:

- `checkpoint_000050.safetensors` — named weights (VarMap)
- `checkpoint_000050.meta.json` — step, lr, config path, dtype

**Breaking change:** old flat `.bin` checkpoints from the CPU/ndarray version are not compatible.

## Tests

```bash
cargo test --features cpu --no-default-features
# or with CUDA:
cargo test --features cuda
```

## Architecture

Each transformer block (except layer 0 FFN) uses MLA + DeepSeekMoE. Layer 0 uses a dense SwiGLU FFN per the DeepSeek-V2 design.

See `configs/` for hyperparameters.
