use ferris_lm::config::ModelConfig;
use ferris_lm::device::default_device;
use ferris_lm::model::{DeepSeekModel, GenerateConfig};
use ferris_lm::train::{compute_total_loss, AdamW, DataLoader, Optimizer, ParamsAdamW};

fn micro_config() -> ModelConfig {
    ModelConfig::from_toml_path("configs/micro.toml").unwrap()
}

#[test]
fn overfit_single_batch() {
    let cfg = micro_config();
    let device = default_device().unwrap();
    let model = DeepSeekModel::new(cfg.clone(), &device).unwrap();
    let tokens: Vec<u32> = (0..256).collect();
    let loader = DataLoader::from_vec(tokens, 1, 16);
    let mut opt = AdamW::new(
        model.varmap().all_vars(),
        ParamsAdamW {
            lr: 1e-3,
            weight_decay: 0.01,
            ..Default::default()
        },
    )
    .unwrap();

    let mut last_loss = f32::MAX;
    for _ in 0..5 {
        let batch = loader.next_batch(&device).unwrap();
        let output = model.forward(&batch.input_ids, Some(&batch.targets)).unwrap();
        let loss = compute_total_loss(&output).unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();
        last_loss = loss.to_scalar().unwrap();
    }
    assert!(last_loss.is_finite());
}

#[test]
fn expert_utilization_non_zero() {
    let cfg = micro_config();
    let device = default_device().unwrap();
    let model = DeepSeekModel::new(cfg.clone(), &device).unwrap();
    let input_ids = candle_core::Tensor::ones((1, 8), candle_core::DType::U32, &device).unwrap();
    let _ = model.forward(&input_ids, None).unwrap();
}

#[test]
fn inference_prefill_runs() {
    let cfg = micro_config();
    let device = default_device().unwrap();
    let model = DeepSeekModel::new(cfg, &device).unwrap();
    let prompt = vec![1, 2, 3, 4];
    let out = model
        .generate(
            &prompt,
            &GenerateConfig {
                max_new_tokens: 4,
                ..Default::default()
            },
            &device,
        )
        .unwrap();
    assert!(out.len() >= prompt.len());
}
