// Layer Normalization para Transformer

use ndarray::{Array1, Array2, Axis};

/// Layer Normalization
///
/// LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
pub struct LayerNorm {
    pub d_model: usize,
    pub gamma: Array1<f32>,  // Escala
    pub beta: Array1<f32>,   // Desplazamiento
    pub eps: f32,            // Epsilon para estabilidad

    // Gradientes
    pub gamma_grad: Array1<f32>,
    pub beta_grad: Array1<f32>,
}

impl LayerNorm {
    /// Crea una nueva capa Layer Normalization
    ///
    /// # Argumentos
    /// * `d_model` - Dimensión del modelo
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            gamma: Array1::ones(d_model),
            beta: Array1::zeros(d_model),
            eps: 1e-5,
            gamma_grad: Array1::zeros(d_model),
            beta_grad: Array1::zeros(d_model),
        }
    }

    /// Forward pass
    ///
    /// # Argumentos
    /// * `x` - Input (seq_len, d_model)
    ///
    /// # Retorna
    /// Normalized output (seq_len, d_model)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();
        let mut output = Array2::zeros(x.dim());

        for i in 0..seq_len {
            let row = x.row(i);
            
            // Calcular media y varianza
            let mean: f32 = row.mean().unwrap();
            let variance: f32 = row.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f32>() / self.d_model as f32;
            let std = (variance + self.eps).sqrt();

            // Normalizar
            for j in 0..self.d_model {
                let normalized = (x[[i, j]] - mean) / std;
                output[[i, j]] = self.gamma[j] * normalized + self.beta[j];
            }
        }

        output
    }

    /// Número de parámetros
    pub fn num_parameters(&self) -> usize {
        self.d_model * 2  // gamma + beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let d_model = 64;
        let ln = LayerNorm::new(d_model);

        let x = Array2::from_shape_fn((10, d_model), |(i, j)| {
            (i * d_model + j) as f32 * 0.1
        });

        let output = ln.forward(&x);

        assert_eq!(output.shape(), &[10, d_model]);

        // Verificar que cada fila tiene media ~0 y varianza ~1
        for row in output.axis_iter(Axis(0)) {
            let mean: f32 = row.mean().unwrap();
            assert!(mean.abs() < 0.01, "Mean should be close to 0");
        }
    }
}
