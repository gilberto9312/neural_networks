// batch.rs
// Procesamiento por lotes (batching) para entrenamiento

use ndarray::{Array2, Axis};
use std::collections::HashMap;

/// Representa un lote de datos para entrenamiento
#[derive(Clone)]
pub struct Batch {
    pub inputs: Vec<Vec<usize>>,  // Token IDs de cada ejemplo
    pub targets: Vec<usize>,       // Clase objetivo para cada ejemplo
}

impl Batch {
    /// Crea un nuevo lote vacío
    pub fn new() -> Self {
        Batch {
            inputs: Vec::new(),
            targets: Vec::new(),
        }
    }

    /// Añade un ejemplo al lote
    pub fn add(&mut self, input: Vec<usize>, target: usize) {
        self.inputs.push(input);
        self.targets.push(target);
    }

    /// Retorna el tamaño del lote
    pub fn size(&self) -> usize {
        self.inputs.len()
    }
}

/// DataLoader para iterar sobre batches
pub struct DataLoader {
    batches: Vec<Batch>,
    current_idx: usize,
    batch_size: usize,
}

impl DataLoader {
    /// Crea un nuevo DataLoader
    ///
    /// # Argumentos
    /// * `data` - Vector de tuplas (token_ids, clase)
    /// * `batch_size` - Tamaño de cada lote
    pub fn new(data: Vec<(Vec<usize>, usize)>, batch_size: usize) -> Self {
        let mut batches = Vec::new();
        let mut current_batch = Batch::new();

        for (tokens, target) in data {
            current_batch.add(tokens, target);

            if current_batch.size() >= batch_size {
                batches.push(current_batch.clone());
                current_batch = Batch::new();
            }
        }

        // Añadir el último batch si no está vacío
        if current_batch.size() > 0 {
            batches.push(current_batch);
        }

        DataLoader {
            batches,
            current_idx: 0,
            batch_size,
        }
    }

    /// Retorna el siguiente batch, o None si no hay más
    pub fn next_batch(&mut self) -> Option<&Batch> {
        if self.current_idx < self.batches.len() {
            let batch = &self.batches[self.current_idx];
            self.current_idx += 1;
            Some(batch)
        } else {
            None
        }
    }

    /// Reinicia el iterador al principio
    pub fn reset(&mut self) {
        self.current_idx = 0;
    }

    /// Retorna el número total de batches
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Mezcla los batches (para entrenamientos aleatorios)
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        self.batches.shuffle(&mut rng);
    }
}

/// Tokenizador simple basado en palabras
pub struct SimpleTokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
    unk_id: usize,
}

impl SimpleTokenizer {
    /// Crea un tokenizador a partir de un corpus
    pub fn from_corpus(texts: &[String]) -> Self {
        let mut word_to_id = HashMap::new();
        let mut id_to_word = Vec::new();

        // Token especial para palabras desconocidas
        word_to_id.insert("<UNK>".to_string(), 0);
        id_to_word.push("<UNK>".to_string());

        // Construir vocabulario
        for text in texts {
            for word in text.split_whitespace() {
                let word_lower = word.to_lowercase();
                if !word_to_id.contains_key(&word_lower) {
                    let id = id_to_word.len();
                    word_to_id.insert(word_lower.clone(), id);
                    id_to_word.push(word_lower);
                }
            }
        }

        SimpleTokenizer {
            word_to_id,
            id_to_word,
            unk_id: 0,
        }
    }

    /// Convierte texto en IDs de tokens
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                let word_lower = word.to_lowercase();
                *self.word_to_id.get(&word_lower).unwrap_or(&self.unk_id)
            })
            .collect()
    }

    /// Convierte IDs de tokens en texto
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| {
                if id < self.id_to_word.len() {
                    Some(self.id_to_word[id].clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Retorna el tamaño del vocabulario
    pub fn vocab_size(&self) -> usize {
        self.id_to_word.len()
    }
}

/// Promedia embeddings de una secuencia de tokens
///
/// # Argumentos
/// * `embeddings` - Matriz de embeddings (vocab_size × embedding_dim)
/// * `token_ids` - IDs de tokens a promediar
///
/// # Retorna
/// Vector promedio de embeddings
pub fn average_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Vec<f32> {
    let embedding_dim = embeddings.ncols();
    let mut avg = vec![0.0; embedding_dim];
    let count = token_ids.len() as f32;

    if count == 0.0 {
        return avg;
    }

    for &token_id in token_ids {
        if token_id < embeddings.nrows() {
            for (j, &val) in embeddings.row(token_id).iter().enumerate() {
                avg[j] += val;
            }
        }
    }

    // Promediar
    for val in avg.iter_mut() {
        *val /= count;
    }

    avg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_creation() {
        let data = vec![
            (vec![1, 2, 3], 0),
            (vec![4, 5], 1),
            (vec![6, 7, 8, 9], 0),
        ];

        let mut loader = DataLoader::new(data, 2);

        assert_eq!(loader.num_batches(), 2);

        let batch1 = loader.next_batch().unwrap();
        assert_eq!(batch1.size(), 2);

        let batch2 = loader.next_batch().unwrap();
        assert_eq!(batch2.size(), 1);

        assert!(loader.next_batch().is_none());
    }

    #[test]
    fn test_tokenizer() {
        let corpus = vec![
            "hello world".to_string(),
            "hello rust".to_string(),
        ];

        let tokenizer = SimpleTokenizer::from_corpus(&corpus);

        assert_eq!(tokenizer.vocab_size(), 4); // <UNK>, hello, world, rust

        let ids = tokenizer.encode("hello world");
        assert_eq!(ids.len(), 2);

        let text = tokenizer.decode(&ids);
        assert_eq!(text, "hello world");
    }
}
