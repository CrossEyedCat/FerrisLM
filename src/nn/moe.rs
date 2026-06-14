use crate::config::ModelConfig;
use crate::nn::SwiGLU;
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub struct DeepSeekMoE {
    shared_experts: Vec<SwiGLU>,
    routed_experts: Vec<SwiGLU>,
    router_centroids: Tensor,
    config: ModelConfig,
}

pub struct MoEOutput {
    pub output: Tensor,
    pub aux_loss: Tensor,
    pub expert_counts: Vec<usize>,
}

impl DeepSeekMoE {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let std = config.weight_init_std;
        let d = config.hidden_dim;
        let inter = config.expert_intermediate_dim;
        let vb = vb.pp("moe");

        let shared_experts = (0..config.num_shared_experts)
            .map(|i| SwiGLU::new(d, inter, std, vb.pp(format!("shared.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let routed_experts = (0..config.num_routed_experts)
            .map(|i| SwiGLU::new(d, inter, std, vb.pp(format!("routed.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let router_centroids = vb.get_with_hints(
            (d, config.num_routed_experts),
            "router_centroids",
            candle_nn::Init::Randn {
                mean: 0.,
                stdev: std as f64,
            },
        )?;

        Ok(Self {
            shared_experts,
            routed_experts,
            router_centroids,
            config: config.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<MoEOutput> {
        let (batch, seq, hidden) = x.dims3()?;
        let num_tokens = batch * seq;
        let num_experts = self.config.num_routed_experts;
        let top_k = self.config.num_experts_per_token;
        let device = x.device();
        let dtype = x.dtype();

        let xs = x.reshape((num_tokens, hidden))?;
        let mut out = Tensor::zeros((num_tokens, hidden), dtype, device)?;

        for expert in &self.shared_experts {
            out = (out + expert.forward(&xs)?)?;
        }

        let scores = xs.matmul(&self.router_centroids)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let probs_data = probs.to_vec2::<f32>()?;

        let mut top_probs_data = vec![vec![0f32; top_k]; num_tokens];
        let mut top_idx_data = vec![vec![0u32; top_k]; num_tokens];
        for t in 0..num_tokens {
            let mut indexed: Vec<(usize, f32)> = probs_data[t]
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for k in 0..top_k {
                top_idx_data[t][k] = indexed[k].0 as u32;
                top_probs_data[t][k] = indexed[k].1;
            }
        }

        let mut expert_counts = vec![0usize; num_experts];
        let mut gate_probs_sum = vec![0f32; num_experts];

        for e in 0..num_experts {
            let mut token_indices = Vec::new();
            let mut weights = Vec::new();

            for t in 0..num_tokens {
                for k in 0..top_k {
                    if top_idx_data[t][k] as usize == e {
                        token_indices.push(t as u32);
                        weights.push(top_probs_data[t][k]);
                        expert_counts[e] += 1;
                        gate_probs_sum[e] += top_probs_data[t][k];
                    }
                }
            }

            if token_indices.is_empty() {
                continue;
            }

            let idx = Tensor::from_vec(token_indices.clone(), (token_indices.len(),), device)?;
            let selected = xs.index_select(&idx, 0)?;
            let expert_out = self.routed_experts[e].forward(&selected)?;

            let n = weights.len();
            let weights_t = Tensor::from_vec(weights, (n, 1), device)?.to_dtype(dtype)?;
            let weighted = expert_out.broadcast_mul(&weights_t)?;

            out = out.index_add(&idx, &weighted, 0)?;
        }

        out = (out + xs)?;
        let out = out.reshape((batch, seq, hidden))?;

        let aux_loss =
            self.compute_aux_loss(&expert_counts, &gate_probs_sum, num_tokens, device, dtype)?;

        Ok(MoEOutput {
            output: out,
            aux_loss,
            expert_counts,
        })
    }

    fn compute_aux_loss(
        &self,
        expert_counts: &[usize],
        gate_probs_sum: &[f32],
        num_tokens: usize,
        device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> Result<Tensor> {
        let nr = self.config.num_routed_experts as f32;
        let kr = self.config.num_experts_per_token as f32;
        let t = num_tokens as f32;
        let alpha = self.config.moe_aux_loss_alpha;

        let mut loss = 0f32;
        for i in 0..self.config.num_routed_experts {
            let f_i = expert_counts[i] as f32 * nr / (kr * t);
            let p_i = gate_probs_sum[i] / t;
            loss += f_i * p_i;
        }

        Tensor::new(alpha * loss, device)?.to_dtype(dtype)
    }
}

impl Module for DeepSeekMoE {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward(x)?.output)
    }
}
