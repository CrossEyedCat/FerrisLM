mod block;
mod mla;
mod moe;
mod rope;
mod rmsnorm;
mod swiglu;

pub use block::{FfnLayer, TransformerBlock};
pub use mla::{MlaKvCache, MultiHeadLatentAttention};
pub use moe::{DeepSeekMoE, MoEOutput};
pub use rope::RotaryEmbedding;
pub use rmsnorm::RmsNorm;
pub use swiglu::SwiGLU;

use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    init_std: f32,
    vb: VarBuilder,
) -> Result<Linear> {
    let init = candle_nn::Init::Randn {
        mean: 0.,
        stdev: init_std as f64,
    };
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init)?;
    if bias {
        let bound = 1. / (in_dim as f64).sqrt();
        let bs = vb.get_with_hints(
            out_dim,
            "bias",
            candle_nn::Init::Uniform {
                lo: -bound,
                up: bound,
            },
        )?;
        Ok(Linear::new(ws, Some(bs)))
    } else {
        Ok(Linear::new(ws, None))
    }
}

pub fn embedding(vocab: usize, hidden: usize, init_std: f32, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get_with_hints(
        (vocab, hidden),
        "weight",
        candle_nn::Init::Randn {
            mean: 0.,
            stdev: init_std as f64,
        },
    )?;
    Ok(Embedding::new(embeddings, hidden))
}

/// Causal mask broadcastable to `[B, H, S, S]`: 0 on/below diagonal, large negative above.
pub fn causal_mask(
    seq_len: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let ones = Tensor::ones((seq_len, seq_len), dtype, device)?;
    let tril = Tensor::tril2(seq_len, dtype, device)?;
    let triu = (&ones - &tril)?;
    Ok((triu * -1e9f64)?)
}

pub fn causal_mask_with_offset(
    query_len: usize,
    key_len: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> Result<Tensor> {
    if query_len == key_len {
        return causal_mask(query_len, dtype, device);
    }
    let start = key_len - query_len;
    let mut data = vec![-1e9f32; query_len * key_len];
    for q in 0..query_len {
        let abs_q = start + q;
        for k in 0..=abs_q {
            data[q * key_len + k] = 0.0;
        }
    }
    Tensor::from_vec(data, (1, 1, query_len, key_len), device)?.to_dtype(dtype)
}

pub fn merge_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let (b, s, _, _) = x.dims4()?;
    x.reshape((b, s, num_heads * head_dim))
}

pub fn split_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let (b, s, _) = x.dims3()?;
    x.reshape((b, s, num_heads, head_dim))
}

pub fn apply_causal_scores(scores: &Tensor, seq_len: usize) -> Result<Tensor> {
    let mask = causal_mask(seq_len, scores.dtype(), scores.device())?;
    let mask = mask.reshape((1, 1, seq_len, seq_len))?;
    scores.broadcast_add(&mask)
}
