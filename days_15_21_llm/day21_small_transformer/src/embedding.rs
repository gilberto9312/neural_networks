// Embedding Layer para Transformer

use ndarray::{Array1, Array2};
use rand::Rng;

/// Capa de Embeddings
pub struct EmbeddingLayer {
    pub vocab_size: usize,
    pub d_model: usize,
    pub embeddings: Array2<f32>,  // (vocab_size, d_model)
    pub embeddings_grad: Array2<f32>,
}

impl EmbeddingLayer {
    /// Crea una nueva capa de embeddings
    ///
    /// # Argumentos
    /// * `vocab_size` - Tamaño del vocabulario
    /// * `d_model` - Dimensión de los embeddings
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        let mut rng = rand::thread_rng();
        let limit = (1.0 / d_model as f32).sqrt();

        let embeddings = Array2::from_shape_fn((vocab_size, d_model), |_| {
            rng.gen_range(-limit..limit)
        });

        Self {
            vocab_size,
            d_model,
            embeddings: embeddings.clone(),
            embeddings_grad: Array2::zeros(embeddings.dim()),
        }
    }

    /// Forward pass: convierte IDs de tokens a embeddings
    ///
    /// # Argumentos
    /// * `token_ids` - Slice de IDs de tokens
    ///
    /// # Retorna
    /// Matriz de embeddings (seq_len, d_model)
    pub fn forward(&self, token_ids: &[usize]) -> Array2<f32> {
        let seq_len = token_ids.len();
        let mut output = Array2::zeros((seq_len, self.d_model));

        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id < self.vocab_size {
                for j in 0..self.d_model {
                    output[[i, j]] = self.embeddings[[token_id, j]];
                }
            }
        }

        output
    }

    /// Obtiene el embedding de un token específico
    pub fn get_embedding(&self, token_id: usize) -> Option<Array1<f32>> {
        if token_id < self.vocab_size {
            Some(self.embeddings.row(token_id).to_owned())
        } else {
            None
        }
    }

    /// Número de parámetros
    pub fn num_parameters(&self) -> usize {
        self.vocab_size * self.d_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_layer() {
        let vocab_size = 1000;
        let d_model = 128;
        let emb = EmbeddingLayer::new(vocab_size, d_model);

        let token_ids = vec![0, 10, 50, 100];
        let output = emb.forward(&token_ids);

        assert_eq!(output.shape(), &[4, d_model]);
    }

    #[test]
    fn test_get_embedding() {
        let emb = EmbeddingLayer::new(100, 64);
        
        let emb_0 = emb.get_embedding(0);
        assert!(emb_0.is_some());
        assert_eq!(emb_0.unwrap().len(), 64);

        let emb_invalid = emb.get_embedding(1000);
        assert!(emb_invalid.is_none());
    }
}
