use clap::Parser;
use ferris_lm::token::{load_corpus_tokens, save_tokens_binary};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long, default_value = "data/tokens.bin")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let tokens = load_corpus_tokens(&args.corpus)?;
    save_tokens_binary(&tokens, &args.output)?;
    println!("saved {} tokens to {}", tokens.len(), args.output.display());
    Ok(())
}
