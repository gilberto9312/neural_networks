// Multi-Head Attention para Transformer

use ndarray::{Array2, Array1, Axis};
use crate::utils::{softmax, create_causal_mask};
use rand::Rng;

/// Scaled Dot-Product Attention
///
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
///
/// # Argumentos
/// * `q` - Queries (seq_len, d_k)
/// * `k` - Keys (seq_len, d_k)
/// * `v` - Values (seq_len, d_v)
/// * `mask` - Máscara causal opcional
pub fn scaled_dot_product_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    mask: Option<&Array2<f32>>,
) -> (Array2<f32>, Array2<f32>) {
    let d_k = k.ncols() as f32;
    let scale = 1.0 / d_k.sqrt();

    // QK^T
    let mut scores = q.dot(&k.t());
    
    // Escalar
    scores.mapv_inplace(|x| x * scale);

    // Aplicar máscara si existe
    if let Some(m) = mask {
        scores += m;
    }

    // Softmax
    let attention_weights = softmax(&scores);

    // Multiplicar por values
    let output = attention_weights.dot(v);

    (output, attention_weights)
}

/// Multi-Head Attention
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub d_v: usize,

    // Proyecciones para Q, K, V
    pub wq: Vec<Array2<f32>>,  // num_heads proyecciones
    pub wk: Vec<Array2<f32>>,
    pub wv: Vec<Array2<f32>>,

    // Proyección de salida
    pub wo: Array2<f32>,

    // Gradientes (para backprop)
    pub wq_grad: Vec<Array2<f32>>,
    pub wk_grad: Vec<Array2<f32>>,
    pub wv_grad: Vec<Array2<f32>>,
    pub wo_grad: Array2<f32>,
}

impl MultiHeadAttention {
    /// Crea una nueva capa Multi-Head Attention
    ///
    /// # Argumentos
    /// * `d_model` - Dimensión del modelo
    /// * `num_heads` - Número de cabezas de atención
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model debe ser divisible por num_heads");

        let d_k = d_model / num_heads;
        let d_v = d_model / num_heads;

        let mut rng = rand::thread_rng();

        // Inicializar pesos para cada cabeza
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();

        for _ in 0..num_heads {
            wq.push(Self::init_weights(d_model, d_k, &mut rng));
            wk.push(Self::init_weights(d_model, d_k, &mut rng));
            wv.push(Self::init_weights(d_model, d_v, &mut rng));
        }

        // Proyección de salida
        let wo = Self::init_weights(d_model, d_model, &mut rng);

        // Gradientes (inicializados en cero)
        let wq_grad = vec![Array2::zeros((d_model, d_k)); num_heads];
        let wk_grad = vec![Array2::zeros((d_model, d_k)); num_heads];
        let wv_grad = vec![Array2::zeros((d_model, d_v)); num_heads];
        let wo_grad = Array2::zeros((d_model, d_model));

        Self {
            num_heads,
            d_model,
            d_k,
            d_v,
            wq,
            wk,
            wv,
            wo,
            wq_grad,
            wk_grad,
            wv_grad,
            wo_grad,
        }
    }

    fn init_weights(rows: usize, cols: usize, rng: &mut impl Rng) -> Array2<f32> {
        let limit = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| {
            rng.gen_range(-limit..limit)
        })
    }

    /// Forward pass de Multi-Head Attention
    ///
    /// # Argumentos
    /// * `x` - Input (seq_len, d_model)
    /// * `mask` - Máscara causal opcional
    ///
    /// # Retorna
    /// (output, attention_weights)
    pub fn forward(&self, x: &Array2<f32>, mask: Option<&Array2<f32>>) -> (Array2<f32>, Vec<Array2<f32>>) {
        let seq_len = x.nrows();
        let mut head_outputs = Vec::new();
        let mut attention_weights = Vec::new();

        // Aplicar cada cabeza
        for i in 0..self.num_heads {
            // Proyectar a Q, K, V
            let q = x.dot(&self.wq[i]);
            let k = x.dot(&self.wk[i]);
            let v = x.dot(&self.wv[i]);

            // Scaled dot-product attention
            let (head_out, weights) = scaled_dot_product_attention(&q, &k, &v, mask);
            
            head_outputs.push(head_out);
            attention_weights.push(weights);
        }

        // Concatenar outputs de todas las cabezas
        let mut concat = Array2::zeros((seq_len, self.d_model));
        for (i, head_out) in head_outputs.iter().enumerate() {
            let start = i * self.d_k;
            let end = start + self.d_k;
            for row in 0..seq_len {
                for col in 0..self.d_k {
                    concat[[row, start + col]] = head_out[[row, col]];
                }
            }
        }

        // Proyección de salida
        let output = concat.dot(&self.wo);

        (output, attention_weights)
    }

    /// Número de parámetros entrenables
    pub fn num_parameters(&self) -> usize {
        let head_params = self.num_heads * (self.d_model * self.d_k * 3); // Q, K, V
        let output_params = self.d_model * self.d_model; // Wo
        head_params + output_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_attention() {
        let q = Array2::from_shape_vec((3, 4), vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]).unwrap();

        let k = q.clone();
        let v = Array2::from_shape_vec((3, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]).unwrap();

        let (output, weights) = scaled_dot_product_attention(&q, &k, &v, None);

        assert_eq!(output.shape(), &[3, 2]);
        assert_eq!(weights.shape(), &[3, 3]);

        // Verificar que los pesos suman 1
        for row in weights.axis_iter(Axis(0)) {
            let sum: f32 = row.sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_multi_head_attention() {
        let d_model = 64;
        let num_heads = 4;
        let mha = MultiHeadAttention::new(d_model, num_heads);

        let x = Array2::zeros((10, d_model));
        let (output, weights) = mha.forward(&x, None);

        assert_eq!(output.shape(), &[10, d_model]);
        assert_eq!(weights.len(), num_heads);
    }
}
