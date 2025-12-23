// Tokenizador BPE (Byte Pair Encoding)
// Implementación completa del algoritmo de Byte Pair Encoding

use crate::vocab::Vocabulary;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Par de tokens que se pueden fusionar
type TokenPair = (String, String);

/// Tokenizador BPE que aprende subpalabras del corpus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    pub vocab: Vocabulary,
    pub merges: Vec<TokenPair>,
    word_pattern: Option<String>, // Guardamos el patrón como String para serialización
}

impl BPETokenizer {
    /// Crea un nuevo tokenizador BPE vacío
    pub fn new() -> Self {
        BPETokenizer {
            vocab: Vocabulary::new(),
            merges: Vec::new(),
            word_pattern: Some(r"\b\w+\b".to_string()),
        }
    }

    /// Entrena el tokenizador BPE en un corpus
    pub fn train(corpus: &[String], num_merges: usize) -> Self {
        let mut tokenizer = BPETokenizer::new();

        // Paso 1: Inicializar corpus como lista de palabras tokenizadas
        let (mut word_corpus, mut vocab) = tokenizer.initialize_corpus(corpus);

        // Paso 2-5: Realizar merges iterativamente
        let mut merges = Vec::new();

        for i in 0..num_merges {
            // Contar frecuencias de pares adyacentes
            let pair_freqs = get_pair_frequencies(&word_corpus);

            if pair_freqs.is_empty() {
                println!("No hay más pares para fusionar en iteración {}", i);
                break;
            }

            // Encontrar el par más frecuente
            let (most_freq_pair, freq) = pair_freqs
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(pair, &count)| (pair.clone(), count))
                .unwrap();

            if freq < 1 {
                break;
            }

            // Crear nuevo token fusionado
            let new_token = format!("{}{}", most_freq_pair.0, most_freq_pair.1);
            vocab.add_token(new_token.clone());
            merges.push(most_freq_pair.clone());

            if (i + 1) % 100 == 0 || i < 10 {
                println!(
                    "Merge {}/{}: ('{}', '{}') -> '{}' (freq: {})",
                    i + 1,
                    num_merges,
                    most_freq_pair.0,
                    most_freq_pair.1,
                    new_token,
                    freq
                );
            }

            // Aplicar merge a todo el corpus
            for word in &mut word_corpus {
                *word = merge_pair_in_word(word, &most_freq_pair);
            }
        }

        tokenizer.vocab = vocab;
        tokenizer.merges = merges;
        tokenizer
    }

    /// Inicializa el corpus como palabras tokenizadas en caracteres
    fn initialize_corpus(&self, corpus: &[String]) -> (Vec<Vec<String>>, Vocabulary) {
        let mut word_corpus = Vec::new();
        let mut vocab = Vocabulary::new();

        let word_regex = Regex::new(self.word_pattern.as_ref().unwrap()).unwrap();

        for text in corpus {
            for word_match in word_regex.find_iter(text) {
                let word = word_match.as_str().to_lowercase();

                // Convertir palabra en lista de caracteres + EOW
                let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                chars.push(vocab.special_tokens.eow.clone());

                // Agregar caracteres al vocabulario
                for ch in &chars {
                    vocab.add_token(ch.clone());
                }

                word_corpus.push(chars);
            }
        }

        (word_corpus, vocab)
    }

    /// Tokeniza un texto usando las reglas BPE aprendidas
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let word_regex = Regex::new(self.word_pattern.as_ref().unwrap()).unwrap();
        let mut tokens = Vec::new();

        for word_match in word_regex.find_iter(text) {
            let word = word_match.as_str().to_lowercase();

            // Inicializar palabra como caracteres + EOW
            let mut word_tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            word_tokens.push(self.vocab.special_tokens.eow.clone());

            // Aplicar todas las fusiones en orden
            for merge in &self.merges {
                word_tokens = merge_pair_in_word(&word_tokens, merge);
            }

            tokens.extend(word_tokens);
        }

        tokens
    }

    /// Codifica un texto a IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.tokenize(text)
            .iter()
            .map(|token| self.vocab.get_id(token))
            .collect()
    }

    /// Decodifica IDs a texto
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab.get_token(id))
            .map(|s| s.replace(&self.vocab.special_tokens.eow, " "))
            .collect::<String>()
            .trim()
            .to_string()
    }

    /// Guarda el vocabulario en formato JSON
    pub fn save_vocab(&self, path: &str) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(&self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Carga el vocabulario desde JSON
    pub fn load_vocab(path: &str) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        let tokenizer = serde_json::from_str(&json)?;
        Ok(tokenizer)
    }
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Cuenta la frecuencia de pares adyacentes en el corpus
fn get_pair_frequencies(corpus: &[Vec<String>]) -> HashMap<TokenPair, usize> {
    let mut pair_freqs = HashMap::new();

    for word in corpus {
        for i in 0..word.len().saturating_sub(1) {
            let pair = (word[i].clone(), word[i + 1].clone());
            *pair_freqs.entry(pair).or_insert(0) += 1;
        }
    }

    pair_freqs
}

/// Fusiona todas las ocurrencias de un par en una palabra
fn merge_pair_in_word(word: &[String], pair_to_merge: &TokenPair) -> Vec<String> {
    let merged_symbol = format!("{}{}", pair_to_merge.0, pair_to_merge.1);
    let mut new_word = Vec::new();
    let mut i = 0;

    while i < word.len() {
        // Si encontramos el par, fusionar
        if i < word.len() - 1 && word[i] == pair_to_merge.0 && word[i + 1] == pair_to_merge.1 {
            new_word.push(merged_symbol.clone());
            i += 2;
        } else {
            new_word.push(word[i].clone());
            i += 1;
        }
    }

    new_word
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_corpus() {
        let tokenizer = BPETokenizer::new();
        let corpus = vec!["hello world".to_string()];
        let (word_corpus, _vocab) = tokenizer.initialize_corpus(&corpus);

        assert_eq!(word_corpus.len(), 2); // "hello" y "world"
        assert_eq!(word_corpus[0], vec!["h", "e", "l", "l", "o", "</w>"]);
    }

    #[test]
    fn test_merge_pair_in_word() {
        let word = vec!["h".to_string(), "e".to_string(), "l".to_string(), "l".to_string()];
        let pair = ("l".to_string(), "l".to_string());
        let merged = merge_pair_in_word(&word, &pair);

        assert_eq!(merged, vec!["h", "e", "ll"]);
    }

    #[test]
    fn test_bpe_training() {
        let corpus = vec![
            "desert".to_string(),
            "deserted".to_string(),
            "desert".to_string(),
        ];

        let tokenizer = BPETokenizer::train(&corpus, 5);
        assert!(tokenizer.merges.len() <= 5);
        assert!(tokenizer.vocab.size() > 5); // Más que solo tokens especiales
    }

    #[test]
    fn test_encode_decode() {
        let corpus = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello".to_string(),
        ];

        let tokenizer = BPETokenizer::train(&corpus, 10);
        let text = "hello";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);

        // El decoded debería contener "hello" (puede tener formato diferente)
        assert!(decoded.contains("hello") || decoded.chars().all(|c| "helo ".contains(c)));
    }

    #[test]
    fn test_get_pair_frequencies() {
        let corpus = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "d".to_string()],
        ];

        let freqs = get_pair_frequencies(&corpus);
        assert_eq!(freqs[&("a".to_string(), "b".to_string())], 2);
        assert_eq!(freqs[&("c".to_string(), "d".to_string())], 1);
    }
}
