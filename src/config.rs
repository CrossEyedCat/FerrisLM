use candle_core::DType;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

fn default_dtype() -> String {
    "f32".to_string()
}

fn default_use_flash_attn() -> bool {
    false
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub kv_compression_dim: usize,
    pub query_compression_dim: usize,
    pub rope_head_dim: usize,
    pub compress_queries: bool,
    pub num_shared_experts: usize,
    pub num_routed_experts: usize,
    pub num_experts_per_token: usize,
    pub expert_intermediate_dim: usize,
    pub moe_aux_loss_alpha: f32,
    pub weight_init_std: f32,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub tie_word_embeddings: bool,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    #[serde(default = "default_use_flash_attn")]
    pub use_flash_attn: bool,
    #[serde(default)]
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub warmup_steps: usize,
    pub grad_clip: f32,
    pub total_steps: usize,
    pub checkpoint_every: usize,
}

impl ModelConfig {
    pub fn from_toml_path(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let content = fs::read_to_string(path)?;
        let config: ModelConfig = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn candle_dtype(&self) -> DType {
        match self.dtype.to_lowercase().as_str() {
            "bf16" | "bfloat16" => DType::BF16,
            "f16" | "float16" | "half" => DType::F16,
            _ => DType::F32,
        }
    }

    pub fn attention_scale(&self) -> f32 {
        1.0 / ((self.head_dim + self.rope_head_dim) as f32).sqrt()
    }

    pub fn qkv_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn q_dim_with_rope(&self) -> usize {
        self.num_heads * (self.head_dim + self.rope_head_dim)
    }

    pub fn estimate_total_params(&self) -> usize {
        let d = self.hidden_dim;
        let v = self.vocab_size;
        let l = self.num_layers;

        let embed = v * d;
        let lm_head = if self.tie_word_embeddings { 0 } else { v * d };

        let mla_per_layer = self.estimate_mla_params();
        let moe_per_layer = self.estimate_moe_params();
        let dense_ffn = self.estimate_swiglu_params(d, self.expert_intermediate_dim);

        let blocks = if l > 0 {
            dense_ffn + (l - 1) as usize * (mla_per_layer + moe_per_layer)
        } else {
            0
        };
        let first_block = mla_per_layer + dense_ffn;

        embed + lm_head + first_block + blocks.saturating_sub(dense_ffn)
    }

    fn estimate_mla_params(&self) -> usize {
        let d = self.hidden_dim;
        let q_dim = self.qkv_dim();
        let dc = self.kv_compression_dim;
        let dc_q = self.query_compression_dim;
        let rope = self.rope_head_dim;
        let nh = self.num_heads;

        let q_params = if self.compress_queries {
            d * dc_q + dc_q + dc_q * q_dim
        } else {
            d * q_dim
        };

        let kv_down = d * dc + dc;
        let kv_up = dc * q_dim * 2;
        let rope_params = dc_q * nh * rope + d * rope;
        let out = q_dim * d;

        q_params + kv_down + kv_up + rope_params + out + dc * 2
    }

    fn estimate_moe_params(&self) -> usize {
        let d = self.hidden_dim;
        let expert = self.estimate_swiglu_params(d, self.expert_intermediate_dim);
        let shared = self.num_shared_experts * expert;
        let routed = self.num_routed_experts * expert;
        let router = d * self.num_routed_experts;
        shared + routed + router
    }

    fn estimate_swiglu_params(&self, d: usize, intermediate: usize) -> usize {
        d * intermediate * 3 + intermediate + d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_tiny_config() {
        let cfg = ModelConfig::from_toml_path("configs/tiny.toml").unwrap();
        assert_eq!(cfg.num_layers, 12);
        assert_eq!(cfg.hidden_dim, 768);
    }
}
