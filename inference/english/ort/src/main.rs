use anyhow::{Error as E, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use ndarray::{s, Array2};
use ort::{
    execution_providers::DirectMLExecutionProvider,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
};
use std::{collections::HashMap, io::Write};
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    // 1. Initialize ONNX Runtime with DirectML execution provider
    println!("Initializing ONNX Runtime with DirectML...");
    ort::init()
        .with_name("olmo_inference")
        .with_execution_providers([DirectMLExecutionProvider::default().build()])
        .commit();

    // 2. Fetch model files from the Hugging Face Hub
    println!("Locating files on Hugging Face (will download if not cached)...");
    let api = Api::new()?;
    
    // Note: You must point this to a repository that contains an exported .onnx version of OLMo
    let repo = api.repo(Repo::new("your-username/OLMo-1B-onnx".to_string(), RepoType::Model));
    
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let model_filename = repo.get("model.onnx")?; // ONNX Runtime requires the pre-compiled graph

    // 3. Load Tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_filename)
        .map_err(|e| E::msg(e.to_string()))?;

    // 4 & 5. Load Configuration and Model Weights
    println!("Loading ONNX model into DirectML session (this may take a moment)...");
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_filename)?;

    // 6. Setup Prompt
    let prompt = "The impact of artificial";
    println!("\nPrompt: \"{}\"", prompt);
    let encoding = tokenizer.encode(prompt, false).map_err(|e| E::msg(e.to_string()))?;
    let mut tokens = encoding.get_ids().to_vec();

    // 7. Generation Loop (Greedy Decoding)
    let max_tokens_to_generate = 200;
    
    let mut first_token_time = std::time::Duration::default();
    let mut subsequent_tokens_time = std::time::Duration::default();
    let mut last_token_time = std::time::Duration::default();
    
    for i in 0..max_tokens_to_generate {
        let start = Instant::now();
        
        // Convert tokens to an ndarray of shape [batch_size, sequence_length]
        // ONNX LLMs typically expect input_ids as INT64
        let input_ids_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let input_array = Array2::from_shape_vec((1, tokens.len()), input_ids_i64)?;
        
        // Attention mask of 1s (matching the length of the input)
        let attention_mask = Array2::from_elem((1, tokens.len()), 1i64);

        // Forward pass. 
        // Note: Raw ONNX requires explicit KV cache state management for efficient single-token 
        // stepping. For simplicity and parity with the naive loop, we pass the full sequence here.
        let inputs: HashMap<&str, ort::value::Value> = inputs! {
            "input_ids" => ort::value::Value::from_array(input_array.into_dyn())?,
            "attention_mask" => ort::value::Value::from_array(attention_mask.into_dyn())?
        }?;

        let outputs = session.run(inputs)?;
                
        // Extract the logits tensor (typically f32)
        let logits = outputs["logits"].try_extract_tensor::<f32>()?;
        let logits_view = logits.view();
        
        // Shape is usually [batch_size, sequence_length, vocab_size]
        let seq_len = logits_view.shape()[1];
        
        // Extract the logits for the very last token in the sequence
        let next_token_logits = logits_view.slice(s![0, seq_len - 1, ..]);
        
        // Greedily select the token ID with the highest probability (Argmax)
        let mut max_val = f32::NEG_INFINITY;
        let mut next_token_id = 0u32;
        
        for (idx, &val) in next_token_logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                next_token_id = idx as u32;
            }
        }
        
        tokens.push(next_token_id);
        
        // Decode and print the new token
        if let Ok(text) = tokenizer.decode(&[next_token_id], false) {
            print!("{}", text);
            std::io::stdout().flush()?;
        } else {
            print!("-{}-", next_token_id);
            std::io::stdout().flush()?;
        }

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