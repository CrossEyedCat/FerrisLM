pub struct LearningRateScheduler {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
}

impl LearningRateScheduler {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr: base_lr as f64,
            warmup_steps,
            total_steps: total_steps.max(1),
        }
    }

    pub fn lr_at_step(&self, step: usize) -> f64 {
        if step <= self.warmup_steps {
            return self.base_lr * step as f64 / self.warmup_steps.max(1) as f64;
        }
        let progress = (step - self.warmup_steps) as f64
            / (self.total_steps - self.warmup_steps).max(1) as f64;
        let min_lr = self.base_lr * 0.1;
        min_lr + 0.5 * (self.base_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}
