// Transformer Encoder

use ndarray::Array2;
use crate::attention::MultiHeadAttention;
use crate::feedforward::FeedForward;
use crate::layer_norm::LayerNorm;

/// Encoder Layer
///
/// EncoderLayer = LayerNorm(x + MultiHeadAttention(x)) → LayerNorm(x + FFN(x))
pub struct EncoderLayer {
    pub attention: MultiHeadAttention,
    pub feedforward: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
}

impl EncoderLayer {
    /// Crea una nueva capa encoder
    ///
    /// # Argumentos
    /// * `d_model` - Dimensión del modelo
    /// * `num_heads` - Número de cabezas de atención
    /// * `d_ff` - Dimensión de la capa feed-forward
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads),
            feedforward: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
        }
    }

    /// Forward pass
    ///
    /// # Argumentos
    /// * `x` - Input (seq_len, d_model)
    ///
    /// # Retorna
    /// Output (seq_len, d_model)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Self-attention con residual connection
        let (attn_output, _) = self.attention.forward(x, None);
        let x = x + &attn_output;
        let x = self.norm1.forward(&x);

        // Feed-forward con residual connection
        let ff_output = self.feedforward.forward(&x);
        let x = &x + &ff_output;
        let x = self.norm2.forward(&x);

        x
    }

    /// Número de parámetros
    pub fn num_parameters(&self) -> usize {
        self.attention.num_parameters() +
        self.feedforward.num_parameters() +
        self.norm1.num_parameters() +
        self.norm2.num_parameters()
    }
}

/// Encoder Stack (múltiples capas encoder)
pub struct Encoder {
    pub layers: Vec<EncoderLayer>,
    pub num_layers: usize,
}

impl Encoder {
    /// Crea un nuevo encoder stack
    ///
    /// # Argumentos
    /// * `num_layers` - Número de capas encoder
    /// * `d_model` - Dimensión del modelo
    /// * `num_heads` - Número de cabezas de atención
    /// * `d_ff` - Dimensión feed-forward
    pub fn new(num_layers: usize, d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        let mut layers = Vec::new();

        for _ in 0..num_layers {
            layers.push(EncoderLayer::new(d_model, num_heads, d_ff));
        }

        Self {
            layers,
            num_layers,
        }
    }

    /// Forward pass a través de todas las capas
    ///
    /// # Argumentos
    /// * `x` - Input (seq_len, d_model)
    ///
    /// # Retorna
    /// Output (seq_len, d_model)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut output = x.clone();

        for layer in &self.layers {
            output = layer.forward(&output);
        }

        output
    }

    /// Número total de parámetros
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_layer() {
        let d_model = 64;
        let num_heads = 4;
        let d_ff = 256;

        let layer = EncoderLayer::new(d_model, num_heads, d_ff);
        let x = Array2::zeros((10, d_model));
        let output = layer.forward(&x);

        assert_eq!(output.shape(), &[10, d_model]);
    }

    #[test]
    fn test_encoder() {
        let encoder = Encoder::new(2, 64, 4, 256);
        let x = Array2::zeros((10, 64));
        let output = encoder.forward(&x);

        assert_eq!(output.shape(), &[10, 64]);
    }
}
