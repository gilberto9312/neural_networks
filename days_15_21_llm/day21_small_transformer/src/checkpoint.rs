// Checkpoint saving/loading (simplificado)

use crate::tokenizer::SimpleTokenizer;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
}

/// Guarda configuración del modelo
pub fn save_config(config: &ModelConfig, path: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(config)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Carga configuración del modelo
pub fn load_config(path: &str) -> std::io::Result<ModelConfig> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let config: ModelConfig = serde_json::from_str(&contents)?;
    Ok(config)
}

/// Guarda tokenizer
pub fn save_tokenizer(tokenizer: &SimpleTokenizer, path: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(tokenizer)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Carga tokenizer
pub fn load_tokenizer(path: &str) -> std::io::Result<SimpleTokenizer> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let tokenizer: SimpleTokenizer = serde_json::from_str(&contents)?;
    Ok(tokenizer)
}
