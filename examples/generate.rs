use clap::Parser;
use ferris_lm::config::ModelConfig;
use ferris_lm::device::resolve_device;
use ferris_lm::model::{DeepSeekModel, GenerateConfig};
use ferris_lm::token::{decode_tokens, encode_text, load_or_train_tokenizer};
use ferris_lm::train::load_checkpoint;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "configs/tiny.toml")]
    config: PathBuf,
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long)]
    device: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let model_config =
        ModelConfig::from_toml_path(&args.config).map_err(|e| anyhow::anyhow!("{e}"))?;
    let device = resolve_device(args.device.as_deref())?;
    let tokenizer = load_or_train_tokenizer(&args.corpus, model_config.vocab_size)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut varmap = candle_nn::VarMap::new();
    if let Some(path) = &args.checkpoint {
        let meta = load_checkpoint(path, &mut varmap).map_err(|e| anyhow::anyhow!("{e}"))?;
        tracing::info!(step = meta.step, lr = meta.lr, "loaded checkpoint");
    }
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, model_config.candle_dtype(), &device);
    let model = DeepSeekModel::from_varbuilder(model_config.clone(), vb, varmap)?;

    let prompt_ids = encode_text(&tokenizer, &args.prompt);
    let gen_cfg = GenerateConfig {
        max_new_tokens: args.max_tokens,
        temperature: args.temperature,
        ..Default::default()
    };

    let output = model.generate(&prompt_ids, &gen_cfg, &device)?;
    let decoded = decode_tokens(&tokenizer, &output);
    println!("{decoded}");
    Ok(())
}
