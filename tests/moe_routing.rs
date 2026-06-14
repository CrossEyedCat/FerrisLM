use ferris_lm::config::ModelConfig;
use ferris_lm::device::default_device;
use ferris_lm::nn::DeepSeekMoE;
use candle_nn::VarBuilder;

fn micro_config() -> ModelConfig {
    ModelConfig::from_toml_path("configs/micro.toml").unwrap()
}

#[test]
fn moe_routing_uses_all_experts_over_batch() {
    let cfg = micro_config();
    let device = default_device().unwrap();
    let vb = VarBuilder::zeros(cfg.candle_dtype(), &device);
    let moe = DeepSeekMoE::new(&cfg, vb).unwrap();
    let x = candle_core::Tensor::zeros((2, 4, cfg.hidden_dim), cfg.candle_dtype(), &device).unwrap();
    let out = moe.forward(&x).unwrap();
    assert_eq!(out.output.dims(), &[2, 4, cfg.hidden_dim]);
    let total_routed: usize = out.expert_counts.iter().sum();
    assert_eq!(total_routed, 2 * 4 * cfg.num_experts_per_token);
}

#[test]
fn moe_aux_loss_non_negative() {
    let cfg = micro_config();
    let device = default_device().unwrap();
    let vb = VarBuilder::zeros(cfg.candle_dtype(), &device);
    let moe = DeepSeekMoE::new(&cfg, vb).unwrap();
    let x = candle_core::Tensor::zeros((1, 2, cfg.hidden_dim), cfg.candle_dtype(), &device).unwrap();
    let out = moe.forward(&x).unwrap();
    assert!(out.aux_loss.to_scalar::<f32>().unwrap() >= 0.0);
}
