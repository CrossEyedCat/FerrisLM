use candle_core::{Device, Result};

/// Parse device string: `cpu`, `cuda`, `cuda:0`, `cuda:1`.
pub fn parse_device(s: &str) -> Result<Device> {
    let s = s.trim().to_lowercase();
    match s.as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => Device::new_cuda(0),
        other if other.starts_with("cuda:") => {
            let idx: usize = other
                .strip_prefix("cuda:")
                .unwrap_or("0")
                .parse()
                .map_err(|_| candle_core::Error::Msg(format!("invalid device: {other}")))?;
            Device::new_cuda(idx)
        }
        other => Err(candle_core::Error::Msg(format!("unknown device: {other}"))),
    }
}

/// Prefer CUDA when compiled with `cuda` feature, else CPU.
pub fn default_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = Device::new_cuda(0) {
            return Ok(dev);
        }
    }
    Ok(Device::Cpu)
}

pub fn resolve_device(requested: Option<&str>) -> Result<Device> {
    match requested {
        Some(s) => parse_device(s),
        None => default_device(),
    }
}
