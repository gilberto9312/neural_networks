// Muestreo de probabilidades para generación de texto

use crate::ngram::{BigramModel, TrigramModel, UnigramModel};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::HashMap;

/// Genera texto usando un modelo Unigram
///
/// # Argumentos
/// * `model` - Modelo Unigram entrenado
/// * `num_words` - Número de palabras a generar
///
/// # Retorna
/// Vector de palabras generadas
pub fn generate_unigram(model: &UnigramModel, num_words: usize) -> Vec<String> {
    let mut rng = thread_rng();
    let frequencies = model.word_frequencies();

    // Extraer palabras y sus pesos
    let words: Vec<String> = frequencies.keys().cloned().collect();
    let weights: Vec<usize> = words.iter().map(|w| frequencies[w]).collect();

    // Crear distribución ponderada
    let dist = WeightedIndex::new(&weights).expect("No se pudo crear distribución ponderada");

    // Generar palabras
    (0..num_words)
        .map(|_| words[dist.sample(&mut rng)].clone())
        .collect()
}

/// Genera texto usando un modelo Bigram
///
/// # Argumentos
/// * `model` - Modelo Bigram entrenado
/// * `start_word` - Palabra inicial
/// * `num_words` - Número de palabras adicionales a generar
///
/// # Retorna
/// Vector de palabras generadas (incluyendo start_word)
pub fn generate_bigram(model: &BigramModel, start_word: &str, num_words: usize) -> Vec<String> {
    let mut rng = thread_rng();
    let mut result = vec![start_word.to_string()];

    for _ in 0..num_words {
        let current_word = result.last().unwrap();
        let next_words = model.possible_next_words(current_word);

        if next_words.is_empty() {
            // Si no hay continuación posible, seleccionar palabra aleatoria del vocabulario
            let vocab = model.vocabulary();
            if vocab.is_empty() {
                break;
            }
            let random_word = vocab.choose(&mut rng).unwrap().clone();
            result.push(random_word);
        } else {
            // Muestreo ponderado de las posibles palabras siguientes
            let next_word = sample_from_distribution(&next_words, &mut rng);
            result.push(next_word);
        }
    }

    result
}

/// Genera texto usando un modelo Trigram
///
/// # Argumentos
/// * `model` - Modelo Trigram entrenado
/// * `start_words` - Par de palabras iniciales (w1, w2)
/// * `num_words` - Número de palabras adicionales a generar
///
/// # Retorna
/// Vector de palabras generadas (incluyendo start_words)
pub fn generate_trigram(
    model: &TrigramModel,
    start_words: (&str, &str),
    num_words: usize,
) -> Vec<String> {
    let mut rng = thread_rng();
    let mut result = vec![start_words.0.to_string(), start_words.1.to_string()];

    for _ in 0..num_words {
        let len = result.len();
        let w1 = &result[len - 2];
        let w2 = &result[len - 1];

        let next_words = model.possible_next_words(w1, w2);

        if next_words.is_empty() {
            // Si no hay continuación posible, terminar
            break;
        } else {
            let next_word = sample_from_distribution(&next_words, &mut rng);
            result.push(next_word);
        }
    }

    result
}

/// Función auxiliar para muestreo de una distribución de probabilidad
///
/// # Argumentos
/// * `distribution` - HashMap de palabra -> probabilidad
/// * `rng` - Generador de números aleatorios
///
/// # Retorna
/// Palabra muestreada
fn sample_from_distribution(distribution: &HashMap<String, f64>, rng: &mut ThreadRng) -> String {
    let words: Vec<&String> = distribution.keys().collect();
    let probs: Vec<f64> = words.iter().map(|w| distribution[*w]).collect();

    // Normalizar probabilidades (por si acaso no suman 1.0)
    let total: f64 = probs.iter().sum();
    let normalized_probs: Vec<f64> = probs.iter().map(|p| p / total).collect();

    // Convertir a pesos enteros para WeightedIndex (multiplicar por 10000 para mantener precisión)
    let weights: Vec<usize> = normalized_probs
        .iter()
        .map(|p| (p * 10000.0) as usize)
        .collect();

    // Asegurarnos de que al menos un peso sea > 0
    let max_weight = *weights.iter().max().unwrap_or(&0);
    if max_weight == 0 {
        // Si todos los pesos son 0, hacer distribución uniforme
        return words.choose(rng).unwrap().to_string();
    }

    let dist = WeightedIndex::new(&weights).expect("Error creando distribución");
    words[dist.sample(rng)].clone()
}

/// Genera una palabra de inicio aleatoria del vocabulario
pub fn random_start_word(vocabulary: &[String], rng: &mut ThreadRng) -> String {
    vocabulary.choose(rng).unwrap_or(&"the".to_string()).clone()
}

/// Genera un par de palabras de inicio aleatorias del vocabulario
pub fn random_start_bigram(vocabulary: &[String], rng: &mut ThreadRng) -> (String, String) {
    let w1 = vocabulary.choose(rng).unwrap_or(&"the".to_string()).clone();
    let w2 = vocabulary.choose(rng).unwrap_or(&"music".to_string()).clone();
    (w1, w2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ngram::{BigramModel, UnigramModel};

    #[test]
    fn test_generate_unigram() {
        let tokens = vec![
            "the".to_string(),
            "cat".to_string(),
            "sat".to_string(),
            "on".to_string(),
            "the".to_string(),
            "mat".to_string(),
        ];
        let model = UnigramModel::new(&tokens);
        let generated = generate_unigram(&model, 5);

        assert_eq!(generated.len(), 5);
        // Todas las palabras generadas deben estar en el vocabulario
        for word in generated {
            assert!(model.word_frequencies().contains_key(&word));
        }
    }

    #[test]
    fn test_generate_bigram() {
        let tokens = vec![
            "the".to_string(),
            "cat".to_string(),
            "sat".to_string(),
            "on".to_string(),
            "the".to_string(),
            "mat".to_string(),
        ];
        let model = BigramModel::new(&tokens);
        let generated = generate_bigram(&model, "the", 3);

        assert!(generated.len() >= 1); // Al menos la palabra inicial
        assert_eq!(generated[0], "the");
    }
}
