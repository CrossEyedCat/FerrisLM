use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f32, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints((size,), "weight", candle_nn::Init::Const(1.))?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}
