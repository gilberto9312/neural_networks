// Tokenizer simple basado en palabras (simplificado del BPE del día 17)

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Tokenizer simple basado en palabras
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, usize>,
    pub pad_token_id: usize,
    pub unk_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
}

impl SimpleTokenizer {
    pub fn from_corpus(corpus: &[String], min_freq: usize) -> Self {
        let mut token_counts: HashMap<String, usize> = HashMap::new();

        for text in corpus {
            for word in text.to_lowercase().split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                if !clean.is_empty() {
                    *token_counts.entry(clean.to_string()).or_insert(0) += 1;
                }
            }
        }

        let mut vocab = vec![
            "<PAD>".to_string(),
            "<UNK>".to_string(),
            "<BOS>".to_string(),
            "<EOS>".to_string(),
        ];

        let mut regular_tokens: Vec<String> = token_counts
            .iter()
            .filter(|(_, count)| **count >= min_freq)
            .map(|(token, _)| token.clone())
            .collect();

        regular_tokens.sort();
        vocab.extend(regular_tokens);

        let token_to_id: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), i))
            .collect();

        Self {
            vocab: vocab.clone(),
            token_to_id,
            pad_token_id: 0,
            unk_token_id: 1,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<usize> {
        let mut ids = Vec::new();

        if add_special {
            ids.push(self.bos_token_id);
        }

        for word in text.to_lowercase().split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean.is_empty() {
                let id = self.token_to_id
                    .get(clean)
                    .copied()
                    .unwrap_or(self.unk_token_id);
                ids.push(id);
            }
        }

        if add_special {
            ids.push(self.eos_token_id);
        }

        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| {
                if id >= self.vocab.len() {
                    return None;
                }
                let token = &self.vocab[id];
                if token == "<PAD>" || token == "<BOS>" || token == "<EOS>" {
                    None
                } else if token == "<UNK>" {
                    Some("<UNK>".to_string())
                } else {
                    Some(token.clone())
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
