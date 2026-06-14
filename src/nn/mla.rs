use crate::config::ModelConfig;
use crate::nn::{causal_mask_with_offset, linear, merge_heads, split_heads, RotaryEmbedding, RmsNorm};
use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

pub struct MultiHeadLatentAttention {
    w_dq: Option<Linear>,
    w_uq: Option<Linear>,
    w_q: Option<Linear>,
    w_qr: Linear,
    w_kr: Linear,
    w_dkv: Linear,
    q_norm: RmsNorm,
    kv_norm: RmsNorm,
    w_uk: Linear,
    w_uv: Linear,
    w_o: Linear,
    rope: RotaryEmbedding,
    config: ModelConfig,
}

#[derive(Clone, Default)]
pub struct MlaKvCache {
    pub c_kv: Vec<Option<Tensor>>,
    pub k_r: Vec<Option<Tensor>>,
    pub seq_len: usize,
}

impl MlaKvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            c_kv: vec![None; num_layers],
            k_r: vec![None; num_layers],
            seq_len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.c_kv.fill(None);
        self.k_r.fill(None);
        self.seq_len = 0;
    }
}

impl MultiHeadLatentAttention {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let d = config.hidden_dim;
        let q_dim = config.qkv_dim();
        let dc = config.kv_compression_dim;
        let dc_q = config.query_compression_dim;
        let rope_dim = config.rope_head_dim;
        let std = config.weight_init_std;
        let vb = vb.pp("mla");

        let (w_dq, w_uq, w_q) = if config.compress_queries {
            (
                Some(linear(d, dc_q, false, std, vb.pp("w_dq"))?),
                Some(linear(dc_q, q_dim, false, std, vb.pp("w_uq"))?),
                None,
            )
        } else {
            (None, None, Some(linear(d, q_dim, false, std, vb.pp("w_q"))?))
        };

        let q_source_dim = if config.compress_queries { dc_q } else { d };

        Ok(Self {
            w_dq,
            w_uq,
            w_q,
            w_qr: linear(q_source_dim, config.num_heads * rope_dim, false, std, vb.pp("w_qr"))?,
            w_kr: linear(d, rope_dim, false, std, vb.pp("w_kr"))?,
            w_dkv: linear(d, dc, false, std, vb.pp("w_dkv"))?,
            q_norm: RmsNorm::new(q_source_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            kv_norm: RmsNorm::new(dc, config.rms_norm_eps, vb.pp("kv_norm"))?,
            w_uk: linear(dc, q_dim, false, std, vb.pp("w_uk"))?,
            w_uv: linear(dc, q_dim, false, std, vb.pp("w_uv"))?,
            w_o: linear(q_dim, d, false, std, vb.pp("w_o"))?,
            rope: RotaryEmbedding::new(
                rope_dim,
                config.max_seq_len,
                10000.0,
                vb.device(),
                vb.dtype(),
            )?,
            config: config.clone(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cache: Option<&mut MlaKvCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let rope_dim = self.config.rope_head_dim;

        let offset = cache.as_ref().map(|c| c.seq_len).unwrap_or(0);

        let c_kv_new = self.kv_norm.forward(&self.w_dkv.forward(x)?)?;
        let k_r_new = self.rope.apply(&self.w_kr.forward(x)?, offset)?;

        let (c_kv, k_r, key_len) = if let Some(cache) = cache {
            let c_kv = match cache.c_kv[layer_idx].as_ref() {
                Some(prev) => Tensor::cat(&[prev, &c_kv_new], 1)?,
                None => c_kv_new.clone(),
            };
            let k_r = match cache.k_r[layer_idx].as_ref() {
                Some(prev) => Tensor::cat(&[prev, &k_r_new], 1)?,
                None => k_r_new.clone(),
            };
            cache.c_kv[layer_idx] = Some(c_kv.clone());
            cache.k_r[layer_idx] = Some(k_r.clone());
            cache.seq_len = offset + seq;
            let key_len = c_kv.dim(1)?;
            (c_kv, k_r, key_len)
        } else {
            (c_kv_new, k_r_new, seq)
        };

        let (q_c, q_source) = if self.config.compress_queries {
            let c_q = self.w_dq.as_ref().unwrap().forward(x)?;
            let c_q = self.q_norm.forward(&c_q)?;
            let q = self.w_uq.as_ref().unwrap().forward(&c_q)?;
            (q, c_q)
        } else {
            let q = self.w_q.as_ref().unwrap().forward(x)?;
            (q.clone(), x.clone())
        };

        let q_r = self
            .rope
            .apply_multi_head(&self.w_qr.forward(&q_source)?, num_heads, offset)?;

        let k_c = self.w_uk.forward(&c_kv)?;
        let v = self.w_uv.forward(&c_kv)?;

        let q_c = split_heads(&q_c, num_heads, head_dim)?;
        let q_r = q_r.reshape((batch, seq, num_heads, rope_dim))?;
        let k_c = split_heads(&k_c, num_heads, head_dim)?;
        let k_r = k_r.reshape((batch, key_len, 1, rope_dim))?;
        let k_r = k_r.broadcast_as((batch, key_len, num_heads, rope_dim))?;

        let q = Tensor::cat(&[&q_c, &q_r], D::Minus1)?;
        let k = Tensor::cat(&[&k_c, &k_r], D::Minus1)?;
        let v = split_heads(&v, num_heads, head_dim)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = self.config.attention_scale() as f64;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mut scores = q.matmul(&k_t)?.affine(scale, 0.)?;
        let mask = causal_mask_with_offset(seq, key_len, scores.dtype(), scores.device())?
            .broadcast_as(scores.shape())?;
        scores = scores.broadcast_add(&mask)?;

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.contiguous()?;
        let merged = merge_heads(&ctx, num_heads, head_dim)?;
        self.w_o.forward(&merged)
    }
}

impl Module for MultiHeadLatentAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x, None, 0)
    }
}
