use candle_core::{Device, Result, Tensor};
use memmap2::Mmap;
use rand::Rng;
use std::fs::File;
use std::path::Path;

enum TokenStorage {
    Mmap(Mmap),
    Vec(Vec<u32>),
}

pub struct DataLoader {
    storage: TokenStorage,
    num_tokens: usize,
    batch_size: usize,
    seq_len: usize,
}

pub struct DataBatch {
    pub input_ids: Tensor,
    pub targets: Tensor,
}

impl DataLoader {
    pub fn from_binary(path: &Path, batch_size: usize, seq_len: usize) -> Result<Self> {
        let file = File::open(path).map_err(candle_core::Error::wrap)?;
        let mmap = unsafe { Mmap::map(&file).map_err(candle_core::Error::wrap)? };
        let num_tokens = mmap.len() / 4;
        Ok(Self {
            storage: TokenStorage::Mmap(mmap),
            num_tokens,
            batch_size,
            seq_len,
        })
    }

    pub fn from_vec(tokens: Vec<u32>, batch_size: usize, seq_len: usize) -> Self {
        let num_tokens = tokens.len();
        Self {
            storage: TokenStorage::Vec(tokens),
            num_tokens,
            batch_size,
            seq_len,
        }
    }

    fn read_token(&self, idx: usize) -> u32 {
        match &self.storage {
            TokenStorage::Mmap(mmap) => {
                let start = idx * 4;
                let bytes: [u8; 4] = mmap[start..start + 4].try_into().unwrap();
                u32::from_le_bytes(bytes)
            }
            TokenStorage::Vec(tokens) => tokens[idx],
        }
    }

    pub fn len_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn next_batch(&self, device: &Device) -> Result<DataBatch> {
        let mut rng = rand::thread_rng();
        let needed = self.seq_len + 1;
        let max_start = self.num_tokens.saturating_sub(needed);

        let mut flat = Vec::with_capacity(self.batch_size * needed);
        for _ in 0..self.batch_size {
            let start = if max_start == 0 {
                0
            } else {
                rng.gen_range(0..max_start)
            };
            for i in 0..needed {
                flat.push(self.read_token(start + i));
            }
        }

        let batch = Tensor::from_vec(flat, (self.batch_size, needed), device)?;
        let input_ids = batch.narrow(1, 0, self.seq_len)?;
        let targets = batch.narrow(1, 1, self.seq_len)?;
        Ok(DataBatch { input_ids, targets })
    }
}
