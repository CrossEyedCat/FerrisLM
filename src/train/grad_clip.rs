use candle_core::{backprop::GradStore, Result, Var};

/// Global L2 gradient clipping over a GradStore.
pub fn clip_grad_norm(vars: &[Var], grads: &mut GradStore, max_norm: f64) -> Result<()> {
    if max_norm <= 0.0 {
        return Ok(());
    }

    let mut total = 0f64;
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            total += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }

    let norm = total.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for var in vars {
            if let Some(g) = grads.get(var.as_tensor()) {
                grads.insert(var.as_tensor(), (g * scale)?);
            }
        }
    }
    Ok(())
}
