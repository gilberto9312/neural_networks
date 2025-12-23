// Tokenizador por caracteres
// Separa el texto en caracteres individuales

use crate::vocab::Vocabulary;
use unicode_segmentation::UnicodeSegmentation;

/// Tokenizador que divide el texto en caracteres individuales
pub struct CharTokenizer {
    pub vocab: Vocabulary,
}

impl CharTokenizer {
    /// Crea un nuevo tokenizador de caracteres entrenado con un corpus
    pub fn new(corpus: &[String]) -> Self {
        let mut vocab = Vocabulary::new();

        // Extraer todos los caracteres únicos del corpus
        for text in corpus {
            for grapheme in text.graphemes(true) {
                vocab.add_token(grapheme.to_string());
            }
        }

        CharTokenizer { vocab }
    }

    /// Tokeniza un texto en caracteres
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.graphemes(true).map(|s| s.to_string()).collect()
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
            .join("")
    }

    /// Codifica con tokens especiales (BOS y EOS)
    pub fn encode_with_special(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![self.vocab.bos_id()];
        ids.extend(self.encode(text));
        ids.push(self.vocab.eos_id());
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer() {
        let corpus = vec!["hello".to_string(), "world".to_string()];
        let tokenizer = CharTokenizer::new(&corpus);

        let tokens = tokenizer.tokenize("hello");
        assert_eq!(tokens, vec!["h", "e", "l", "l", "o"]);

        let text = "hello";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_char_tokenizer_unicode() {
        let corpus = vec!["café".to_string(), "niño".to_string()];
        let tokenizer = CharTokenizer::new(&corpus);

        let tokens = tokenizer.tokenize("café");
        assert_eq!(tokens, vec!["c", "a", "f", "é"]);
    }

    #[test]
    fn test_encode_with_special() {
        let corpus = vec!["test".to_string()];
        let tokenizer = CharTokenizer::new(&corpus);

        let ids = tokenizer.encode_with_special("hi");
        assert!(ids.len() > 2); // BOS + chars + EOS
        assert_eq!(ids[0], tokenizer.vocab.bos_id());
        assert_eq!(ids[ids.len() - 1], tokenizer.vocab.eos_id());
    }
}
