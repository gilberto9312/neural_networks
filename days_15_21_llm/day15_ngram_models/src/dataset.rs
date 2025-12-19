// Carga del dataset Africa Galore

use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

/// Estructura que representa un ítem del dataset Africa Galore
#[derive(Debug, Deserialize)]
pub struct AfricaItem {
    pub category: String,
    pub name: String,
    pub description: String,
}

/// Carga el dataset Africa Galore desde un archivo JSON
///
/// # Argumentos
/// * `path` - Ruta al archivo africa_galore.json
///
/// # Retorna
/// Vector de textos (descripciones)
pub fn load_africa_galore(path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let items: Vec<AfricaItem> = serde_json::from_reader(reader)?;

    // Extraer solo las descripciones
    let texts = items.into_iter()
        .map(|item| item.description)
        .collect();

    Ok(texts)
}

/// Tokeniza un texto en palabras, convirtiendo a minúsculas y eliminando puntuación
///
/// # Argumentos
/// * `text` - Texto a tokenizar
///
/// # Retorna
/// Vector de tokens (palabras)
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// Tokeniza múltiples textos y los combina en un solo vector de tokens
///
/// # Argumentos
/// * `texts` - Vector de textos a tokenizar
///
/// # Retorna
/// Vector único con todos los tokens combinados
pub fn tokenize_corpus(texts: &[String]) -> Vec<String> {
    texts.iter()
        .flat_map(|text| tokenize(text))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "Hello, World! This is a test.";
        let tokens = tokenize(text);
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_tokenize_corpus() {
        let texts = vec![
            "First sentence.".to_string(),
            "Second one!".to_string(),
        ];
        let tokens = tokenize_corpus(&texts);
        assert_eq!(tokens, vec!["first", "sentence", "second", "one"]);
    }
}
