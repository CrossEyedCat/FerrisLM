use crate::model::ForwardOutput;
use candle_core::{Result, Tensor};

pub fn compute_total_loss(output: &ForwardOutput) -> Result<Tensor> {
    match &output.loss {
        Some(ce) => ce + &output.aux_loss,
        None => Ok(output.aux_loss.clone()),
    }
}
