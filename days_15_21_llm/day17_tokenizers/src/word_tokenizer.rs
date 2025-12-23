// Tokenizador por palabras
// Separa el texto en palabras usando espacios en blanco

use crate::vocab::Vocabulary;
use regex::Regex;

/// Tokenizador que divide el texto en palabras
pub struct WordTokenizer {
    pub vocab: Vocabulary,
    word_pattern: Regex,
}

impl WordTokenizer {
    /// Crea un nuevo tokenizador de palabras entrenado con un corpus
    pub fn new(corpus: &[String]) -> Self {
        let mut vocab = Vocabulary::new();

        // Patrón regex para extraer palabras (letras, números, apóstrofes)
        let word_pattern = Regex::new(r"\b\w+\b").unwrap();

        // Extraer todas las palabras únicas del corpus
        for text in corpus {
            for word in word_pattern.find_iter(text) {
                let word_lower = word.as_str().to_lowercase();
                vocab.add_token(word_lower);
            }
        }

        WordTokenizer {
            vocab,
            word_pattern,
        }
    }

    /// Crea un tokenizador simple que divide por espacios
    pub fn simple(corpus: &[String]) -> Self {
        let mut vocab = Vocabulary::new();

        // Extraer palabras dividiendo por espacios
        for text in corpus {
            for word in text.split_whitespace() {
                let word_lower = word.to_lowercase();
                vocab.add_token(word_lower);
            }
        }

        let word_pattern = Regex::new(r"\S+").unwrap();

        WordTokenizer {
            vocab,
            word_pattern,
        }
    }

    /// Tokeniza un texto en palabras (lowercase)
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.word_pattern
            .find_iter(text)
            .map(|m| m.as_str().to_lowercase())
            .collect()
    }

    /// Codifica un texto a una secuencia de IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.tokenize(text)
            .iter()
            .map(|token| self.vocab.get_id(token))
            .collect()
    }

    /// Decodifica una secuencia de IDs a texto
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab.get_token(id))
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Codifica con tokens especiales (BOS y EOS)
    pub fn encode_with_special(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![self.vocab.bos_id()];
        ids.extend(self.encode(text));
        ids.push(self.vocab.eos_id());
        ids
    }

    /// Retorna las palabras más frecuentes del vocabulario
    pub fn most_common(&self, n: usize) -> Vec<String> {
        // Retorna las primeras n palabras (después de tokens especiales)
        self.vocab
            .id_to_token
            .iter()
            .skip(5) // Saltar tokens especiales
            .take(n)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_tokenizer() {
        let corpus = vec![
            "Hello world".to_string(),
            "Hello Rust".to_string(),
        ];
        let tokenizer = WordTokenizer::new(&corpus);

        let tokens = tokenizer.tokenize("Hello world");
        assert_eq!(tokens, vec!["hello", "world"]);

        let text = "hello world";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_word_tokenizer_punctuation() {
        let corpus = vec!["Hello, world!".to_string()];
        let tokenizer = WordTokenizer::new(&corpus);

        // El patrón \b\w+\b elimina puntuación
        let tokens = tokenizer.tokenize("Hello, world!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_simple_tokenizer() {
        let corpus = vec!["one two three".to_string()];
        let tokenizer = WordTokenizer::simple(&corpus);

        let tokens = tokenizer.tokenize("one two");
        assert!(tokens.len() == 2);
    }

    #[test]
    fn test_encode_with_special() {
        let corpus = vec!["test".to_string()];
        let tokenizer = WordTokenizer::new(&corpus);

        let ids = tokenizer.encode_with_special("test");
        assert_eq!(ids[0], tokenizer.vocab.bos_id());
        assert_eq!(ids[ids.len() - 1], tokenizer.vocab.eos_id());
    }
}
