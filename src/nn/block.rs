use crate::config::ModelConfig;
use crate::nn::{DeepSeekMoE, MultiHeadLatentAttention, RmsNorm, SwiGLU};
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub enum FfnLayer {
    Dense(SwiGLU),
    MoE(DeepSeekMoE),
}

pub struct TransformerBlock {
    attn_norm: RmsNorm,
    attn: MultiHeadLatentAttention,
    ffn_norm: RmsNorm,
    ffn: FfnLayer,
    layer_idx: usize,
}

impl TransformerBlock {
    pub fn new(config: &ModelConfig, layer_idx: usize, use_moe: bool, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp(format!("block.{layer_idx}"));
        let ffn = if use_moe {
            FfnLayer::MoE(DeepSeekMoE::new(config, vb.pp("moe"))?)
        } else {
            FfnLayer::Dense(SwiGLU::new(
                config.hidden_dim,
                config.expert_intermediate_dim,
                config.weight_init_std,
                vb.pp("ffn"),
            )?)
        };

        Ok(Self {
            attn_norm: RmsNorm::new(config.hidden_dim, config.rms_norm_eps, vb.pp("attn_norm"))?,
            attn: MultiHeadLatentAttention::new(config, vb.pp("attn"))?,
            ffn_norm: RmsNorm::new(config.hidden_dim, config.rms_norm_eps, vb.pp("ffn_norm"))?,
            ffn,
            layer_idx,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cache: Option<&mut crate::nn::MlaKvCache>,
    ) -> Result<(Tensor, Tensor)> {
        let normed = self.attn_norm.forward(x)?;
        let attn_out = self.attn.forward(&normed, cache, self.layer_idx)?;
        let x = (x + attn_out)?;

        let normed = self.ffn_norm.forward(&x)?;
        match &self.ffn {
            FfnLayer::Dense(ffn) => {
                let out = ffn.forward(&normed)?;
                let zero = Tensor::new(0f32, x.device())?.to_dtype(x.dtype())?;
                Ok(((x + out)?, zero))
            }
            FfnLayer::MoE(moe) => {
                let moe_out = moe.forward(&normed)?;
                Ok((moe_out.output, moe_out.aux_loss))
            }
        }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward(x, None)?.0)
    }
}
