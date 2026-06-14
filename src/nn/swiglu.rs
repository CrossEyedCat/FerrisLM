use crate::nn::linear;
use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

pub struct SwiGLU {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl SwiGLU {
    pub fn new(hidden: usize, intermediate: usize, std: f32, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: linear(hidden, intermediate, false, std, vb.pp("w1"))?,
            w2: linear(intermediate, hidden, false, std, vb.pp("w2"))?,
            w3: linear(hidden, intermediate, false, std, vb.pp("w3"))?,
        })
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?.silu()?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}
