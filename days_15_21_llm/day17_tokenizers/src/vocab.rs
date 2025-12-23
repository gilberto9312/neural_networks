// Vocabulario y mapeo token↔ID
// Gestión de tokens especiales y mapeo bidireccional

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tokens especiales utilizados en tokenización
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub pad: String,
    pub unk: String,
    pub bos: String,
    pub eos: String,
    pub eow: String, // End of word para BPE
}

impl Default for SpecialTokens {
    fn default() -> Self {
        SpecialTokens {
            pad: "<PAD>".to_string(),
            unk: "<UNK>".to_string(),
            bos: "<BOS>".to_string(),
            eos: "<EOS>".to_string(),
            eow: "</w>".to_string(),
        }
    }
}

/// Vocabulario con mapeo bidireccional token ↔ ID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    pub special_tokens: SpecialTokens,
}

impl Vocabulary {
    /// Crea un vocabulario vacío con tokens especiales
    pub fn new() -> Self {
        let special_tokens = SpecialTokens::default();
        let mut vocab = Vocabulary {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            special_tokens: special_tokens.clone(),
        };

        // Agregar tokens especiales al vocabulario
        vocab.add_token(special_tokens.pad.clone());
        vocab.add_token(special_tokens.unk.clone());
        vocab.add_token(special_tokens.bos.clone());
        vocab.add_token(special_tokens.eos.clone());
        vocab.add_token(special_tokens.eow.clone());

        vocab
    }

    /// Crea un vocabulario desde una lista de tokens
    pub fn from_tokens(tokens: Vec<String>) -> Self {
        let mut vocab = Vocabulary::new();
        for token in tokens {
            if !vocab.contains(&token) {
                vocab.add_token(token);
            }
        }
        vocab
    }

    /// Agrega un token al vocabulario
    pub fn add_token(&mut self, token: String) -> usize {
        if let Some(&id) = self.token_to_id.get(&token) {
            id
        } else {
            let id = self.id_to_token.len();
            self.token_to_id.insert(token.clone(), id);
            self.id_to_token.push(token);
            id
        }
    }

    /// Obtiene el ID de un token (retorna UNK si no existe)
    pub fn get_id(&self, token: &str) -> usize {
        self.token_to_id
            .get(token)
            .copied()
            .unwrap_or_else(|| self.token_to_id[&self.special_tokens.unk])
    }

    /// Obtiene el token correspondiente a un ID
    pub fn get_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    /// Verifica si un token existe en el vocabulario
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Retorna el tamaño del vocabulario
    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Retorna el ID del token PAD
    pub fn pad_id(&self) -> usize {
        self.token_to_id[&self.special_tokens.pad]
    }

    /// Retorna el ID del token UNK
    pub fn unk_id(&self) -> usize {
        self.token_to_id[&self.special_tokens.unk]
    }

    /// Retorna el ID del token BOS
    pub fn bos_id(&self) -> usize {
        self.token_to_id[&self.special_tokens.bos]
    }

    /// Retorna el ID del token EOS
    pub fn eos_id(&self) -> usize {
        self.token_to_id[&self.special_tokens.eos]
    }

    /// Retorna el ID del token EOW
    pub fn eow_id(&self) -> usize {
        self.token_to_id[&self.special_tokens.eow]
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Aplica padding o truncamiento a una secuencia de IDs
pub fn pad_or_truncate(ids: Vec<usize>, max_length: usize, pad_id: usize) -> Vec<usize> {
    let mut result = ids;

    if result.len() > max_length {
        // Truncar
        result.truncate(max_length);
    } else if result.len() < max_length {
        // Padding
        result.resize(max_length, pad_id);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.size(), 5); // 5 tokens especiales
        assert!(vocab.contains("<PAD>"));
        assert!(vocab.contains("<UNK>"));
    }

    #[test]
    fn test_add_token() {
        let mut vocab = Vocabulary::new();
        let id = vocab.add_token("hello".to_string());
        assert_eq!(vocab.get_id("hello"), id);
        assert_eq!(vocab.get_token(id), Some("hello"));
    }

    #[test]
    fn test_unknown_token() {
        let vocab = Vocabulary::new();
        let unk_id = vocab.unk_id();
        assert_eq!(vocab.get_id("nonexistent"), unk_id);
    }

    #[test]
    fn test_pad_or_truncate() {
        let ids = vec![1, 2, 3];
        let padded = pad_or_truncate(ids.clone(), 5, 0);
        assert_eq!(padded, vec![1, 2, 3, 0, 0]);

        let truncated = pad_or_truncate(ids, 2, 0);
        assert_eq!(truncated, vec![1, 2]);
    }
}
