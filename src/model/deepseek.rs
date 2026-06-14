use crate::config::ModelConfig;
use crate::nn::{embedding, linear, MlaKvCache, RmsNorm, TransformerBlock};
use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::{loss, Embedding, Linear, VarBuilder, VarMap};

pub struct ForwardOutput {
    pub logits: Tensor,
    pub loss: Option<Tensor>,
    pub aux_loss: Tensor,
}

pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
        }
    }
}

pub struct DeepSeekModel {
    pub varmap: VarMap,
    pub config: ModelConfig,
    embed: Embedding,
    blocks: Vec<TransformerBlock>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
}

impl DeepSeekModel {
    pub fn new(config: ModelConfig, device: &candle_core::Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, config.candle_dtype(), device);
        Self::from_varbuilder(config, vb, varmap)
    }

    pub fn from_varbuilder(
        config: ModelConfig,
        vb: VarBuilder,
        varmap: VarMap,
    ) -> Result<Self> {
        let std = config.weight_init_std;
        let embed = embedding(config.vocab_size, config.hidden_dim, std, vb.pp("embed"))?;
        let blocks: Vec<_> = (0..config.num_layers)
            .map(|i| TransformerBlock::new(&config, i, i > 0, vb.clone()))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(config.hidden_dim, config.rms_norm_eps, vb.pp("norm"))?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(linear(
                config.hidden_dim,
                config.vocab_size,
                false,
                std,
                vb.pp("lm_head"),
            )?)
        };

        Ok(Self {
            varmap,
            config,
            embed,
            blocks,
            norm,
            lm_head,
        })
    }

    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    pub fn forward(&self, input_ids: &Tensor, targets: Option<&Tensor>) -> Result<ForwardOutput> {
        let mut x = self.embed.forward(input_ids)?;
        let mut aux_loss = Tensor::new(0f32, input_ids.device())?.to_dtype(x.dtype())?;

        for block in &self.blocks {
            let (out, aux) = block.forward(&x, None)?;
            x = out;
            aux_loss = (aux_loss + aux)?;
        }

        x = self.norm.forward(&x)?;
        let logits = self.lm_head_logits(&x)?;

        let loss = targets.map(|t| self.compute_loss(&logits, t)).transpose()?;

        Ok(ForwardOutput {
            logits,
            loss,
            aux_loss,
        })
    }

    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor,
        cache: &mut MlaKvCache,
    ) -> Result<Tensor> {
        let mut x = self.embed.forward(input_ids)?;
        for block in &self.blocks {
            let (out, _) = block.forward(&x, Some(cache))?;
            x = out;
        }
        x = self.norm.forward(&x)?;
        self.lm_head_logits(&x)
    }

    fn lm_head_logits(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(head) = &self.lm_head {
            head.forward(x)
        } else {
            let (b, s, h) = x.dims3()?;
            let w = self.embed.embeddings().t()?.contiguous()?;
            let logits = x.reshape((b * s, h))?.matmul(&w)?;
            logits.reshape((b, s, self.config.vocab_size))
        }
    }

    fn compute_loss(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let (batch, seq, vocab) = logits.dims3()?;
        let logits = logits.reshape((batch * seq, vocab))?;
        let targets = targets.flatten_all()?;
        loss::cross_entropy(&logits, &targets)
    }

    pub fn generate(
        &self,
        prompt: &[u32],
        gen_cfg: &GenerateConfig,
        device: &candle_core::Device,
    ) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();
        let mut cache = MlaKvCache::new(self.config.num_layers);

        for _ in 0..gen_cfg.max_new_tokens {
            let input = if cache.seq_len == 0 {
                Tensor::from_vec(tokens.clone(), (1, tokens.len()), device)?
            } else {
                let last = *tokens.last().unwrap();
                Tensor::from_vec(vec![last], (1, 1), device)?
            };

            let logits = self.forward_with_cache(&input, &mut cache)?;
            let logits = logits.i((0, input.dim(1)? - 1))?;
            let logits = if gen_cfg.temperature > 0.0 {
                (&logits / gen_cfg.temperature as f64)?
            } else {
                logits
            };

            let next = sample_top_p(&logits, gen_cfg.top_k, gen_cfg.top_p)?;
            tokens.push(next);
            if next == 0 {
                break;
            }
        }

        Ok(tokens)
    }

    pub fn num_parameters(&self) -> usize {
        self.config.estimate_total_params()
    }
}

impl Module for DeepSeekModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(self.forward(input_ids, None)?.logits)
    }
}

fn sample_top_p(logits: &Tensor, top_k: usize, top_p: f32) -> Result<u32> {
    let logits = logits.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, (l - max).exp()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(top_k);

    let sum: f32 = indexed.iter().map(|(_, p)| p).sum();
    let mut cum = 0.0f32;
    let threshold = top_p * sum;
    for &(idx, p) in &indexed {
        cum += p;
        if cum >= threshold {
            return Ok(idx as u32);
        }
    }
    Ok(indexed[0].0 as u32)
}
