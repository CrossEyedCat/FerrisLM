use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub struct TokenizerHandle {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub vocab_size: usize,
}

impl TokenizerHandle {
    fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            vocab_size: 0,
        }
    }

    fn add_token(&mut self, token: String) -> u32 {
        if let Some(&id) = self.token_to_id.get(&token) {
            return id;
        }
        let id = self.vocab_size as u32;
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
        self.vocab_size += 1;
        id
    }
}

pub fn load_or_train_tokenizer(corpus_path: &Path, vocab_size: usize) -> Result<TokenizerHandle, Box<dyn std::error::Error + Send + Sync>> {
    let tok_path = corpus_path.with_extension("vocab.txt");
    if tok_path.exists() {
        return load_vocab(&tok_path);
    }

    let mut handle = TokenizerHandle::new();
    handle.add_token("<pad>".to_string());
    handle.add_token("<unk>".to_string());

    let file = File::open(corpus_path)?;
    let reader = BufReader::new(file);
    let mut freq: HashMap<String, usize> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        for word in line.split_whitespace() {
            *freq.entry(word.to_string()).or_insert(0) += 1;
        }
    }

    let mut words: Vec<_> = freq.into_iter().collect();
    words.sort_by(|a, b| b.1.cmp(&a.1));

    for (word, _) in words.into_iter().take(vocab_size.saturating_sub(2)) {
        handle.add_token(word);
    }

    save_vocab(&handle, &tok_path)?;
    Ok(handle)
}

fn save_vocab(handle: &TokenizerHandle, path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = File::create(path)?;
    for id in 0..handle.vocab_size {
        let token = handle.id_to_token.get(&(id as u32)).unwrap();
        writeln!(file, "{id}\t{token}")?;
    }
    Ok(())
}

fn load_vocab(path: &Path) -> Result<TokenizerHandle, Box<dyn std::error::Error + Send + Sync>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut handle = TokenizerHandle::new();
    for line in reader.lines() {
        let line = line?;
        let mut parts = line.splitn(2, '\t');
        let id: u32 = parts.next().unwrap().parse()?;
        let token = parts.next().unwrap().to_string();
        handle.token_to_id.insert(token.clone(), id);
        handle.id_to_token.insert(id, token);
        handle.vocab_size = handle.vocab_size.max(id as usize + 1);
    }
    Ok(handle)
}

pub fn encode_text(handle: &TokenizerHandle, text: &str) -> Vec<u32> {
    text.split_whitespace()
        .map(|w| {
            *handle
                .token_to_id
                .get(w)
                .unwrap_or(&1)
        })
        .collect()
}

pub fn decode_tokens(handle: &TokenizerHandle, ids: &[u32]) -> String {
    ids.iter()
        .filter_map(|id| handle.id_to_token.get(id))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn save_tokens_binary(tokens: &[u32], path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut file = File::create(path)?;
    for &t in tokens {
        file.write_all(&t.to_le_bytes())?;
    }
    Ok(())
}

pub fn load_corpus_tokens(corpus_path: &Path) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let handle = load_or_train_tokenizer(corpus_path, 32000)?;
    let file = File::open(corpus_path)?;
    let reader = BufReader::new(file);
    let mut tokens = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        tokens.extend(encode_text(&handle, &line));
        tokens.push(0);
    }
    Ok(tokens)
}
