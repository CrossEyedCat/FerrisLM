use candle_core::{DType, Device, Result, Tensor, D};

pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f64, device: &Device, dtype: DType) -> Result<Self> {
        let half = dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            inv_freq.push(1.0 / base.powf(i as f64 * 2.0 / dim as f64));
        }
        let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?.to_dtype(dtype)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(positions, (max_seq_len, 1), device)?.to_dtype(dtype)?;
        let freqs = pos.matmul(&inv_freq.reshape((1, half))?)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        Ok(Self { sin, cos })
    }

    /// Apply RoPE to `[batch, seq, dim]`.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let dim = x.dim(2)?;
        let half = dim / 2;

        let sin = self.sin.narrow(0, offset, seq_len)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = sin.unsqueeze(0)?;
        let cos = cos.unsqueeze(0)?;

        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let out1 = ((x1.broadcast_mul(&cos)?) - (x2.broadcast_mul(&sin)?))?;
        let out2 = ((x1.broadcast_mul(&sin)?) + (x2.broadcast_mul(&cos)?))?;
        Tensor::cat(&[&out1, &out2], D::Minus1)
    }

    /// Apply RoPE to `[batch, seq, num_heads * rope_dim]`.
    pub fn apply_multi_head(&self, x: &Tensor, num_heads: usize, offset: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let rope_dim = x.dim(2)? / num_heads;
        let x = x.reshape((b, s, num_heads, rope_dim))?;
        let mut heads = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let head = x.narrow(2, h, 1)?.squeeze(2)?;
            heads.push(self.apply(&head, offset)?.unsqueeze(2)?);
        }
        Tensor::cat(&heads, 2)?.reshape((b, s, num_heads * rope_dim))
    }
}
