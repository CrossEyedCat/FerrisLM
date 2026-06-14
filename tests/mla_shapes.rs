use ferris_lm::config::ModelConfig;
use ferris_lm::device::default_device;
use ferris_lm::model::DeepSeekModel;
use ferris_lm::nn::MultiHeadLatentAttention;
use candle_nn::VarBuilder;

fn micro_config() -> ModelConfig {
    ModelConfig::from_toml_path("configs/micro.toml").unwrap()
}

#[test]
fn mla_output_shape() {
    let cfg = micro_config();
    let device = default_device().unwrap();
    let vb = VarBuilder::zeros(cfg.candle_dtype(), &device);
    let mla = MultiHeadLatentAttention::new(&cfg, vb).unwrap();
    let batch = 2;
    let seq = 8;
    let x = candle_core::Tensor::zeros((batch, seq, cfg.hidden_dim), cfg.candle_dtype(), &device).unwrap();
    let y = mla.forward(&x, None, 0).unwrap();
    assert_eq!(y.dims(), &[batch, seq, cfg.hidden_dim]);
}

#[test]
fn model_forward_shape() {
    let cfg = micro_config();
    let device = default_device().unwrap();
    let model = DeepSeekModel::new(cfg.clone(), &device).unwrap();
    let input_ids =
        candle_core::Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], (2, 8), &device)
            .unwrap();
    let targets =
        candle_core::Tensor::from_vec(vec![2u32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], (2, 8), &device)
            .unwrap();
    let out = model.forward(&input_ids, Some(&targets)).unwrap();
    assert_eq!(out.logits.dims(), &[2, 8, cfg.vocab_size]);
}
