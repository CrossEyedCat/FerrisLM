use clap::Parser;
use ferris_lm::config::ModelConfig;
use ferris_lm::device::resolve_device;
use ferris_lm::model::DeepSeekModel;
use ferris_lm::train::{
    clip_grad_norm, compute_total_loss, save_checkpoint, AdamW, DataLoader, LearningRateScheduler,
    Optimizer, ParamsAdamW,
};
use std::path::PathBuf;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "configs/tiny.toml")]
    config: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long, default_value_t = 10000)]
    steps: usize,
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,
    #[arg(long)]
    device: Option<String>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let model_config =
        ModelConfig::from_toml_path(&args.config).map_err(|e| anyhow::anyhow!("{e}"))?;
    let train_cfg = model_config.training.clone();
    let device = resolve_device(args.device.as_deref())?;
    tracing::info!(?device, "using device");

    let loader = DataLoader::from_binary(
        &args.data,
        train_cfg.batch_size,
        train_cfg.seq_len,
    )?;
    tracing::info!("loaded {} tokens", loader.len_tokens());
    tracing::info!("estimated params: {}", model_config.estimate_total_params());

    let model = DeepSeekModel::new(model_config.clone(), &device)?;
    let mut opt = AdamW::new(
        model.varmap().all_vars(),
        ParamsAdamW {
            lr: train_cfg.learning_rate as f64,
            weight_decay: train_cfg.weight_decay as f64,
            beta1: train_cfg.beta1 as f64,
            beta2: train_cfg.beta2 as f64,
            ..Default::default()
        },
    )?;
    let scheduler = LearningRateScheduler::new(
        train_cfg.learning_rate,
        train_cfg.warmup_steps,
        train_cfg.total_steps.min(args.steps),
    );

    for step in 1..=args.steps {
        let step_start = Instant::now();
        let batch = loader.next_batch(&device)?;
        let output = model.forward(&batch.input_ids, Some(&batch.targets))?;
        let loss = compute_total_loss(&output)?;

        opt.set_learning_rate(scheduler.lr_at_step(step));
        let mut grads = loss.backward()?;
        if train_cfg.grad_clip > 0.0 {
            clip_grad_norm(
                &model.varmap().all_vars(),
                &mut grads,
                train_cfg.grad_clip as f64,
            )?;
        }
        opt.step(&grads)?;

        if step % 10 == 0 {
            let loss_val: f32 = loss.to_scalar()?;
            let aux_val: f32 = output.aux_loss.to_scalar()?;
            tracing::info!(
                step,
                loss = loss_val,
                aux = aux_val,
                lr = opt.learning_rate(),
                ms = step_start.elapsed().as_millis(),
                "train"
            );
        }

        if step % train_cfg.checkpoint_every == 0 {
            let path = save_checkpoint(
                &args.checkpoint_dir,
                step,
                model.varmap(),
                opt.learning_rate() as f32,
                &args.config.to_string_lossy(),
                &model_config.dtype,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            tracing::info!(?path, "saved checkpoint");
        }
    }

    Ok(())
}
