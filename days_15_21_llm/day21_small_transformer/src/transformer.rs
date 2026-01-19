// Arquitectura Transformer completa

use ndarray::{Array1, Array2};
use crate::embedding::EmbeddingLayer;
use crate::positional::PositionalEncoding;
use crate::decoder::Decoder;
use crate::utils::softmax;

/// Transformer Language Model (Decoder-only)
///
/// Arquitectura simplificada para generación de texto
pub struct TransformerLM {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,

    // Componentes
    pub embedding: EmbeddingLayer,
    pub positional_encoding: PositionalEncoding,
    pub decoder: Decoder,
    pub output_projection: Array2<f32>,  // (d_model, vocab_size)
}

impl TransformerLM {
    /// Crea un nuevo Transformer Language Model
    ///
    /// # Argumentos
    /// * `vocab_size` - Tamaño del vocabulario
    /// * `d_model` - Dimensión del modelo
    /// * `num_heads` - Número de cabezas de atención
    /// * `num_layers` - Número de capas decoder
    /// * `max_seq_len` - Longitud máxima de secuencia
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        max_seq_len: usize,
    ) -> Self {
        let d_ff = d_model * 4;  // Estándar: 4x d_model

        // Inicializar componentes
        let embedding = EmbeddingLayer::new(vocab_size, d_model);
        let positional_encoding = PositionalEncoding::new(max_seq_len, d_model);
        let decoder = Decoder::new(num_layers, d_model, num_heads, d_ff);

        // Proyección final a vocabulario
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let limit = (1.0 / d_model as f32).sqrt();
        let output_projection = Array2::from_shape_fn((d_model, vocab_size), |_| {
            rng.gen_range(-limit..limit)
        });

        Self {
            vocab_size,
            d_model,
            num_heads,
            num_layers,
            d_ff,
            max_seq_len,
            embedding,
            positional_encoding,
            decoder,
            output_projection,
        }
    }

    /// Forward pass
    ///
    /// # Argumentos
    /// * `input_ids` - IDs de tokens de entrada
    ///
    /// # Retorna
    /// Logits (seq_len, vocab_size)
    pub fn forward(&self, input_ids: &[usize]) -> Array2<f32> {
        // 1. Embedding lookup
        let embeddings = self.embedding.forward(input_ids);

        // 2. Añadir positional encoding
        let x = self.positional_encoding.apply(&embeddings);

        // 3. Decoder stack
        let decoder_output = self.decoder.forward(&x);

        // 4. Proyección a vocabulario
        decoder_output.dot(&self.output_projection)
    }

    /// Predice el siguiente token (greedy)
    ///
    /// # Argumentos
    /// * `input_ids` - Secuencia de entrada
    ///
    /// # Retorna
    /// ID del token predicho
    pub fn predict_next(&self, input_ids: &[usize]) -> usize {
        let logits = self.forward(input_ids);
        let last_logits = logits.row(logits.nrows() - 1);

        // Encontrar argmax
        let mut max_idx = 0;
        let mut max_val = last_logits[0];

        for (i, &val) in last_logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        max_idx
    }

    /// Número total de parámetros
    pub fn num_parameters(&self) -> usize {
        self.embedding.num_parameters() +
        self.decoder.num_parameters() +
        (self.d_model * self.vocab_size)  // output_projection
    }

    /// Información del modelo
    pub fn info(&self) -> String {
        format!(
            "TransformerLM(\n\
             \tvocab_size={},\n\
             \td_model={},\n\
             \tnum_heads={},\n\
             \tnum_layers={},\n\
             \td_ff={},\n\
             \tmax_seq_len={},\n\
             \ttotal_params={}\n\
             )",
            self.vocab_size,
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.d_ff,
            self.max_seq_len,
            self.num_parameters()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_lm() {
        let model = TransformerLM::new(
            1000,  // vocab_size
            128,   // d_model
            4,     // num_heads
            2,     // num_layers
            128,   // max_seq_len
        );

        let input_ids = vec![1, 5, 10, 20];
        let logits = model.forward(&input_ids);

        assert_eq!(logits.shape(), &[4, 1000]);
    }

    #[test]
    fn test_predict_next() {
        let model = TransformerLM::new(100, 64, 4, 1, 64);
        let input_ids = vec![1, 2, 3];
        let next_token = model.predict_next(&input_ids);

        assert!(next_token < 100);
    }
}
