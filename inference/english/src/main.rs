use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::olmo::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    // 1. Setup hardware acceleration based on your Cargo.toml features
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        device = Device::new_cuda(0)?;
    }
    #[cfg(feature = "metal")]
    if candle_core::utils::metal_is_available() {
        device = Device::new_metal(0)?;
    }
    println!("Using device: {:?}", device);

    // 2. Fetch model files from the Hugging Face Hub
    println!("Locating files on Hugging Face (will download if not cached)...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new("allenai/OLMo-1B-hf".to_string(), RepoType::Model));
    
    let config_filename = repo.get("config.json")?;
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let weights_filename = repo.get("model.safetensors")?;

    // 3. Load Tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_filename)
        .map_err(|e| E::msg(e.to_string()))?;

    // 4. Load Configuration
    println!("Loading configuration...");
    let config: Config = serde_json::from_reader(std::fs::File::open(config_filename)?)?;

    // 5. Load Model Weights
    // Memory mapping (mmap) allows us to load the weights efficiently.
    // Standard inference typically executes in f32 or f16.
    println!("Loading weights into memory (this may take a moment)...");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? };
    let mut model = Model::new(&config, vb)?;

    // 6. Setup Prompt
    let prompt = "The impact of artificial";
    println!("\nPrompt: \"{}\"", prompt);
    let encoding = tokenizer.encode(prompt, false).map_err(|e| E::msg(e.to_string()))?;
    let mut tokens = encoding.get_ids().to_vec();

    // 7. Generation Loop (Greedy Decoding)
    let max_tokens_to_generate = 200;
    let mut pos = 0;
    let mut input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    let mut first_token_time = std::time::Duration::default();
    let mut subsequent_tokens_time = std::time::Duration::default();
    let mut last_token_time = std::time::Duration::default();
    
    for i in 0..max_tokens_to_generate {
        let start = Instant::now();
        // println!("\rGenerating token {}/{}...", i + 1, max_tokens_to_generate);
        // std::io::stdout().flush()?;
        
        // Forward pass. No KV cacheing for now.
        let logits = model.forward(&input, pos)?;
        pos += 1;
               
        // Extract the logits for the very last token in the sequence
        let next_token_logits = logits.squeeze(0)?.get(logits.dim(1)? - 1)?;
        
        // Greedily select the token ID with the highest probability
        let next_token_id = next_token_logits.argmax(0)?.to_scalar::<u32>()?;
        tokens.push(next_token_id);
        
        // Decode and print the new token as soon as it is generated
        if let Ok(text) = tokenizer.decode(&[next_token_id], false) {
            print!("{}", text);
            std::io::stdout().flush()?;
        } else {
            print!("-{}-", next_token_id);
            std::io::stdout().flush()?;
        }
       
        // Prepare input for the next step: just the generated token
        input = Tensor::new(&[next_token_id], &device)?.unsqueeze(0)?;

        let duration = start.elapsed();
        last_token_time = duration;
        if i == 0 {
            first_token_time = duration;
        } else {
            subsequent_tokens_time += duration;
        }
    }
    
    println!();
    println!("Time to first token: {:.2?}", first_token_time);
    if max_tokens_to_generate > 1 {
        let mean_time = subsequent_tokens_time / (max_tokens_to_generate - 1) as u32;
        println!("Mean time for subsequent tokens: {:.2?} ({:.2} t/s)", mean_time, 1.0 / mean_time.as_secs_f64());
    }
    println!("Time to last token: {:.2?}", last_token_time);
    Ok(())
}