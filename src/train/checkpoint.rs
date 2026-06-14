use candle_nn::VarMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    pub step: usize,
    pub lr: f32,
    pub config_path: String,
    pub dtype: String,
}

pub fn save_checkpoint(
    dir: &Path,
    step: usize,
    varmap: &VarMap,
    lr: f32,
    config_path: &str,
    dtype: &str,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    fs::create_dir_all(dir)?;

    let weights_path = dir.join(format!("checkpoint_{step:06}.safetensors"));
    varmap.save(&weights_path)?;

    let meta = CheckpointMeta {
        step,
        lr,
        config_path: config_path.to_string(),
        dtype: dtype.to_string(),
    };
    let meta_path = dir.join(format!("checkpoint_{step:06}.meta.json"));
    fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

    Ok(weights_path)
}

pub fn load_checkpoint(
    weights_path: &Path,
    varmap: &mut VarMap,
) -> Result<CheckpointMeta, Box<dyn std::error::Error + Send + Sync>> {
    varmap.load(weights_path)?;

    let stem = weights_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("invalid checkpoint path")?;
    let meta_path = weights_path
        .with_file_name(format!("{stem}.meta.json"));
    let content = fs::read_to_string(meta_path)?;
    Ok(serde_json::from_str(&content)?)
}

pub fn load_checkpoint_meta(dir: &Path, step: usize) -> Result<CheckpointMeta, Box<dyn std::error::Error + Send + Sync>> {
    let meta_path = dir.join(format!("checkpoint_{step:06}.meta.json"));
    let content = fs::read_to_string(meta_path)?;
    Ok(serde_json::from_str(&content)?)
}
