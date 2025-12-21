// Estructuras N-gram (Unigram, Bigram, Trigram)

use std::collections::HashMap;

/// Modelo Unigram - Distribución de probabilidad de palabras individuales
#[derive(Debug)]
pub struct UnigramModel {
    /// Contador de frecuencias de cada palabra
    counts: HashMap<String, usize>,
    /// Total de palabras en el corpus
    total_count: usize,
}

impl UnigramModel {
    /// Crea un nuevo modelo unigram a partir de un corpus tokenizado
    pub fn new(tokens: &[String]) -> Self {
        let mut counts = HashMap::new();
        for token in tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        let total_count = tokens.len();

        UnigramModel {
            counts,
            total_count,
        }
    }

    /// Calcula la probabilidad de una palabra
    ///
    /// P(palabra) = count(palabra) / total_palabras
    pub fn probability(&self, word: &str) -> f64 {
        let count = self.counts.get(word).copied().unwrap_or(0);
        if self.total_count == 0 {
            0.0
        } else {
            count as f64 / self.total_count as f64
        }
    }

    /// Retorna el vocabulario (todas las palabras únicas)
    pub fn vocabulary(&self) -> Vec<String> {
        self.counts.keys().cloned().collect()
    }

    /// Retorna las palabras y sus frecuencias
    pub fn word_frequencies(&self) -> &HashMap<String, usize> {
        &self.counts
    }
}

/// Modelo Bigram - Distribución de probabilidad de pares de palabras
#[derive(Debug)]
pub struct BigramModel {
    /// Contador de bigramas: (palabra1, palabra2) -> frecuencia
    bigram_counts: HashMap<(String, String), usize>,
    /// Contador de unigramas para normalización
    unigram_counts: HashMap<String, usize>,
}

impl BigramModel {
    /// Crea un nuevo modelo bigram a partir de un corpus tokenizado
    pub fn new(tokens: &[String]) -> Self {
        let mut bigram_counts = HashMap::new();
        let mut unigram_counts = HashMap::new();

        // Contar bigramas
        for window in tokens.windows(2) {
            let w1 = window[0].clone();
            let w2 = window[1].clone();
            *bigram_counts.entry((w1.clone(), w2)).or_insert(0) += 1;
            *unigram_counts.entry(w1).or_insert(0) += 1;
        }

        // Agregar la última palabra al conteo de unigramas
        if let Some(last) = tokens.last() {
            *unigram_counts.entry(last.clone()).or_insert(0) += 1;
        }

        BigramModel {
            bigram_counts,
            unigram_counts,
        }
    }

    /// Calcula la probabilidad condicional P(w2 | w1)
    ///
    /// P(w2|w1) = count(w1, w2) / count(w1)
    pub fn probability(&self, w1: &str, w2: &str) -> f64 {
        let bigram_count = self.bigram_counts.get(&(w1.to_string(), w2.to_string()))
            .copied()
            .unwrap_or(0);
        let unigram_count = self.unigram_counts.get(w1).copied().unwrap_or(0);

        if unigram_count == 0 {
            0.0
        } else {
            bigram_count as f64 / unigram_count as f64
        }
    }

    /// Retorna todas las posibles palabras siguientes dado un contexto
    pub fn possible_next_words(&self, word: &str) -> HashMap<String, f64> {
        let mut next_words = HashMap::new();

        for ((w1, w2), _) in &self.bigram_counts {
            if w1 == word {
                let prob = self.probability(word, w2);
                next_words.insert(w2.clone(), prob);
            }
        }

        next_words
    }

    /// Retorna el vocabulario
    pub fn vocabulary(&self) -> Vec<String> {
        self.unigram_counts.keys().cloned().collect()
    }
}

/// Modelo Trigram - Distribución de probabilidad de tríos de palabras
#[derive(Debug)]
pub struct TrigramModel {
    /// Contador de trigramas: (palabra1, palabra2, palabra3) -> frecuencia
    trigram_counts: HashMap<(String, String, String), usize>,
    /// Contador de bigramas para normalización
    bigram_counts: HashMap<(String, String), usize>,
}

impl TrigramModel {
    /// Crea un nuevo modelo trigram a partir de un corpus tokenizado
    pub fn new(tokens: &[String]) -> Self {
        let mut trigram_counts = HashMap::new();
        let mut bigram_counts = HashMap::new();

        // Contar trigramas
        for window in tokens.windows(3) {
            let w1 = window[0].clone();
            let w2 = window[1].clone();
            let w3 = window[2].clone();
            *trigram_counts.entry((w1.clone(), w2.clone(), w3)).or_insert(0) += 1;
            *bigram_counts.entry((w1, w2)).or_insert(0) += 1;
        }

        // Agregar el último bigrama
        if tokens.len() >= 2 {
            let w1 = tokens[tokens.len() - 2].clone();
            let w2 = tokens[tokens.len() - 1].clone();
            *bigram_counts.entry((w1, w2)).or_insert(0) += 1;
        }

        TrigramModel {
            trigram_counts,
            bigram_counts,
        }
    }

    /// Calcula la probabilidad condicional P(w3 | w1, w2)
    ///
    /// P(w3|w1,w2) = count(w1, w2, w3) / count(w1, w2)
    pub fn probability(&self, w1: &str, w2: &str, w3: &str) -> f64 {
        let trigram_count = self.trigram_counts
            .get(&(w1.to_string(), w2.to_string(), w3.to_string()))
            .copied()
            .unwrap_or(0);
        let bigram_count = self.bigram_counts
            .get(&(w1.to_string(), w2.to_string()))
            .copied()
            .unwrap_or(0);

        if bigram_count == 0 {
            0.0
        } else {
            trigram_count as f64 / bigram_count as f64
        }
    }

    /// Retorna todas las posibles palabras siguientes dado un contexto de dos palabras
    pub fn possible_next_words(&self, w1: &str, w2: &str) -> HashMap<String, f64> {
        let mut next_words = HashMap::new();

        for ((t1, t2, t3), _) in &self.trigram_counts {
            if t1 == w1 && t2 == w2 {
                let prob = self.probability(w1, w2, t3);
                next_words.insert(t3.clone(), prob);
            }
        }

        next_words
    }
}

/// Calcula la perplexity de un modelo sobre un conjunto de tokens de prueba
///
/// Perplexity = exp(-1/N * Σ log P(palabra_i | contexto))
///
/// Menor perplexity = mejor modelo
pub fn calculate_perplexity_unigram(model: &UnigramModel, test_tokens: &[String]) -> f64 {
    let n = test_tokens.len() as f64;
    let mut log_prob_sum = 0.0;

    for token in test_tokens {
        let prob = model.probability(token);
        if prob > 0.0 {
            log_prob_sum += prob.ln();
        } else {
            // Suavizado simple: asignar probabilidad muy pequeña a palabras no vistas
            log_prob_sum += 1e-10_f64.ln();
        }
    }

    (-log_prob_sum / n).exp()
}

/// Calcula la perplexity de un modelo bigram
pub fn calculate_perplexity_bigram(model: &BigramModel, test_tokens: &[String]) -> f64 {
    if test_tokens.len() < 2 {
        return f64::INFINITY;
    }

    let n = (test_tokens.len() - 1) as f64;
    let mut log_prob_sum = 0.0;

    for window in test_tokens.windows(2) {
        let prob = model.probability(&window[0], &window[1]);
        if prob > 0.0 {
            log_prob_sum += prob.ln();
        } else {
            log_prob_sum += 1e-10_f64.ln();
        }
    }

    (-log_prob_sum / n).exp()
}

/// Calcula la perplexity de un modelo trigram
pub fn calculate_perplexity_trigram(model: &TrigramModel, test_tokens: &[String]) -> f64 {
    if test_tokens.len() < 3 {
        return f64::INFINITY;
    }

    let n = (test_tokens.len() - 2) as f64;
    let mut log_prob_sum = 0.0;

    for window in test_tokens.windows(3) {
        let prob = model.probability(&window[0], &window[1], &window[2]);
        if prob > 0.0 {
            log_prob_sum += prob.ln();
        } else {
            log_prob_sum += 1e-10_f64.ln();
        }
    }

    (-log_prob_sum / n).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unigram() {
        let tokens = vec!["the".to_string(), "cat".to_string(), "the".to_string()];
        let model = UnigramModel::new(&tokens);

        assert_eq!(model.probability("the"), 2.0 / 3.0);
        assert_eq!(model.probability("cat"), 1.0 / 3.0);
        assert_eq!(model.probability("dog"), 0.0);
    }

    #[test]
    fn test_bigram() {
        let tokens = vec!["the".to_string(), "cat".to_string(), "the".to_string(), "dog".to_string()];
        let model = BigramModel::new(&tokens);

        // P(cat|the) = 1/2 porque "the" aparece antes de "cat" 1 vez y "the" aparece 2 veces
        assert_eq!(model.probability("the", "cat"), 0.5);
        assert_eq!(model.probability("the", "dog"), 0.5);
        assert_eq!(model.probability("cat", "the"), 1.0);
    }
}
