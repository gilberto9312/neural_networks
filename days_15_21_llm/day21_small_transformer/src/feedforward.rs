// Feed-Forward Network para Transformer

use ndarray::Array2;
use rand::Rng;

/// Feed-Forward Network (FFN)
///
/// FFN(x) = max(0, xW1 + b1)W2 + b2
///
/// Típicamente: d_ff = 4 * d_model
pub struct FeedForward {
    pub d_model: usize,
    pub d_ff: usize,

    // Primera capa
    pub w1: Array2<f32>,
    pub b1: Array2<f32>,

    // Segunda capa
    pub w2: Array2<f32>,
    pub b2: Array2<f32>,

    // Gradientes
    pub w1_grad: Array2<f32>,
    pub b1_grad: Array2<f32>,
    pub w2_grad: Array2<f32>,
    pub b2_grad: Array2<f32>,
}

impl FeedForward {
    /// Crea una nueva capa Feed-Forward
    ///
    /// # Argumentos
    /// * `d_model` - Dimensión del modelo
    /// * `d_ff` - Dimensión de la capa oculta (típicamente 4 * d_model)
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();

        let w1 = Self::init_weights(d_model, d_ff, &mut rng);
        let b1 = Array2::zeros((1, d_ff));

        let w2 = Self::init_weights(d_ff, d_model, &mut rng);
        let b2 = Array2::zeros((1, d_model));

        Self {
            d_model,
            d_ff,
            w1: w1.clone(),
            b1: b1.clone(),
            w2: w2.clone(),
            b2: b2.clone(),
            w1_grad: Array2::zeros(w1.dim()),
            b1_grad: Array2::zeros(b1.dim()),
            w2_grad: Array2::zeros(w2.dim()),
            b2_grad: Array2::zeros(b2.dim()),
        }
    }

    fn init_weights(rows: usize, cols: usize, rng: &mut impl Rng) -> Array2<f32> {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| {
            rng.gen_range(-limit..limit)
        })
    }

    /// Forward pass
    ///
    /// # Argumentos
    /// * `x` - Input (seq_len, d_model)
    ///
    /// # Retorna
    /// Output (seq_len, d_model)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();

        // Primera capa: x * W1 + b1
        let hidden = x.dot(&self.w1);
        let mut hidden_bias = Array2::zeros((seq_len, self.d_ff));
        
        for i in 0..seq_len {
            for j in 0..self.d_ff {
                hidden_bias[[i, j]] = hidden[[i, j]] + self.b1[[0, j]];
            }
        }

        // ReLU activation
        let activated = hidden_bias.mapv(|v| v.max(0.0));

        // Segunda capa: hidden * W2 + b2
        let output = activated.dot(&self.w2);
        let mut output_bias = Array2::zeros((seq_len, self.d_model));
        
        for i in 0..seq_len {
            for j in 0..self.d_model {
                output_bias[[i, j]] = output[[i, j]] + self.b2[[0, j]];
            }
        }

        output_bias
    }

    /// Número de parámetros
    pub fn num_parameters(&self) -> usize {
        self.d_model * self.d_ff + self.d_ff +  // W1, b1
        self.d_ff * self.d_model + self.d_model  // W2, b2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward() {
        let d_model = 64;
        let d_ff = 256;
        let ff = FeedForward::new(d_model, d_ff);

        let x = Array2::zeros((10, d_model));
        let output = ff.forward(&x);

        assert_eq!(output.shape(), &[10, d_model]);
    }
}
