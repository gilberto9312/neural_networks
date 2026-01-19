// Transformer Decoder

use ndarray::Array2;
use crate::attention::MultiHeadAttention;
use crate::feedforward::FeedForward;
use crate::layer_norm::LayerNorm;
use crate::utils::create_causal_mask;

/// Decoder Layer
///
/// DecoderLayer tiene:
/// 1. Masked self-attention
/// 2. Cross-attention con encoder output (opcional para LM puro)
/// 3. Feed-forward network
pub struct DecoderLayer {
    pub self_attention: MultiHeadAttention,
    pub feedforward: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
}

impl DecoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attention: MultiHeadAttention::new(d_model, num_heads),
            feedforward: FeedForward::new(d_model, d_ff),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();
        let mask = create_causal_mask(seq_len);

        // Masked self-attention
        let (attn_output, _) = self.self_attention.forward(x, Some(&mask));
        let x = x + &attn_output;
        let x = self.norm1.forward(&x);

        // Feed-forward
        let ff_output = self.feedforward.forward(&x);
        let x = &x + &ff_output;
        self.norm2.forward(&x)
    }

    pub fn num_parameters(&self) -> usize {
        self.self_attention.num_parameters() +
        self.feedforward.num_parameters() +
        self.norm1.num_parameters() +
        self.norm2.num_parameters()
    }
}

pub struct Decoder {
    pub layers: Vec<DecoderLayer>,
}

impl Decoder {
    pub fn new(num_layers: usize, d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| DecoderLayer::new(d_model, num_heads, d_ff))
                .collect(),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        self.layers.iter().fold(x.clone(), |acc, layer| layer.forward(&acc))
    }

    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }
}
