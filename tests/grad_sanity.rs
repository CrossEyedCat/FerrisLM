use ferris_lm::device::default_device;
use ferris_lm::nn::{linear, RmsNorm, SwiGLU};
use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;

#[test]
fn grad_flows_through_linear() {
    let device = default_device().unwrap();
    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
    let layer = linear(4, 3, true, 0.02, vb).unwrap();
    let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device).unwrap();
    let y = layer.forward(&x).unwrap();
    let loss = y.sqr().unwrap().sum_all().unwrap();
    loss.backward().unwrap();
}

#[test]
fn grad_flows_through_rmsnorm() {
    let device = default_device().unwrap();
    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
    let norm = RmsNorm::new(4, 1e-6, vb).unwrap();
    let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], &device).unwrap();
    let y = norm.forward(&x).unwrap();
    let loss = y.sum_all().unwrap();
    loss.backward().unwrap();
}

#[test]
fn grad_flows_through_swiglu() {
    let device = default_device().unwrap();
    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
    let ffn = SwiGLU::new(8, 16, 0.02, vb).unwrap();
    let x = Tensor::new(&[[0.1f32; 8], [0.1f32; 8]], &device).unwrap();
    let y = ffn.forward(&x).unwrap();
    let loss = y.sum_all().unwrap();
    loss.backward().unwrap();
}
