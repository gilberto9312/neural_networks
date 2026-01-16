// Multi-Head Attention
//
// Implementación de atención multi-cabeza que permite al modelo
// atender a información de diferentes subespacios de representación

use ndarray::Array2;
use crate::attention::scaled_dot_product_attention;

/// Multi-Head Attention
///
/// Permite al modelo atender simultáneamente a información de diferentes
/// subespacios de representación en diferentes posiciones.
///
/// MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
/// donde head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
pub struct MultiHeadAttention {
    /// Número de cabezas de atención
    pub num_heads: usize,
    /// Dimensión del modelo
    pub d_model: usize,
    /// Dimensión de cada cabeza (d_model / num_heads)
    pub d_k: usize,
    /// Matrices de proyección para queries (una por cabeza)
    pub w_q: Vec<Array2<f32>>,
    /// Matrices de proyección para keys (una por cabeza)
    pub w_k: Vec<Array2<f32>>,
    /// Matrices de proyección para values (una por cabeza)
    pub w_v: Vec<Array2<f32>>,
    /// Matriz de proyección de salida
    pub w_o: Array2<f32>,
}

impl MultiHeadAttention {
    /// Crea una nueva capa de Multi-Head Attention
    ///
    /// # Argumentos
    /// * `d_model` - Dimensión del modelo (debe ser divisible por num_heads)
    /// * `num_heads` - Número de cabezas de atención
    ///
    /// # Panics
    /// Si d_model no es divisible por num_heads
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model ({}) debe ser divisible por num_heads ({})",
            d_model,
            num_heads
        );

        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;

        let d_k = d_model / num_heads;

        // Inicialización Xavier/Glorot
        let limit = (6.0 / (d_model + d_k) as f32).sqrt();

        let mut w_q = Vec::with_capacity(num_heads);
        let mut w_k = Vec::with_capacity(num_heads);
        let mut w_v = Vec::with_capacity(num_heads);

        for _ in 0..num_heads {
            w_q.push(Array2::random((d_model, d_k), Uniform::new(-limit, limit)));
            w_k.push(Array2::random((d_model, d_k), Uniform::new(-limit, limit)));
            w_v.push(Array2::random((d_model, d_k), Uniform::new(-limit, limit)));
        }

        let w_o = Array2::random((d_model, d_model), Uniform::new(-limit, limit));

        Self {
            num_heads,
            d_model,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Forward pass de Multi-Head Attention
    ///
    /// # Argumentos
    /// * `input` - Entrada (seq_len, d_model)
    /// * `mask` - Máscara opcional para cada cabeza
    ///
    /// # Retorna
    /// Tupla (output, attention_weights_per_head)
    /// * output: Resultado de la atención (seq_len, d_model)
    /// * attention_weights_per_head: Vector con pesos de cada cabeza
    pub fn forward(
        &self,
        input: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let seq_len = input.shape()[0];

        let mut head_outputs = Vec::with_capacity(self.num_heads);
        let mut attention_weights = Vec::with_capacity(self.num_heads);

        // Procesar cada cabeza de atención
        for i in 0..self.num_heads {
            // Proyectar entrada a Q, K, V para esta cabeza
            let q = input.dot(&self.w_q[i]);
            let k = input.dot(&self.w_k[i]);
            let v = input.dot(&self.w_v[i]);

            // Aplicar atención
            let (head_output, attn_weights) =
                scaled_dot_product_attention(&q, &k, &v, mask);

            head_outputs.push(head_output);
            attention_weights.push(attn_weights);
        }

        // Concatenar las salidas de todas las cabezas
        let concat_output = concatenate_heads(&head_outputs);

        // Proyección final
        let output = concat_output.dot(&self.w_o);

        (output, attention_weights)
    }

    /// Forward pass con queries, keys y values separados (para encoder-decoder attention)
    ///
    /// # Argumentos
    /// * `queries` - Queries (seq_len_q, d_model)
    /// * `keys` - Keys (seq_len_k, d_model)
    /// * `values` - Values (seq_len_v, d_model)
    /// * `mask` - Máscara opcional
    ///
    /// # Retorna
    /// Tupla (output, attention_weights_per_head)
    pub fn forward_separate(
        &self,
        queries: &Array2<f32>,
        keys: &Array2<f32>,
        values: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        let mut attention_weights = Vec::with_capacity(self.num_heads);

        // Procesar cada cabeza de atención
        for i in 0..self.num_heads {
            // Proyectar a Q, K, V para esta cabeza
            let q = queries.dot(&self.w_q[i]);
            let k = keys.dot(&self.w_k[i]);
            let v = values.dot(&self.w_v[i]);

            // Aplicar atención
            let (head_output, attn_weights) =
                scaled_dot_product_attention(&q, &k, &v, mask);

            head_outputs.push(head_output);
            attention_weights.push(attn_weights);
        }

        // Concatenar las salidas de todas las cabezas
        let concat_output = concatenate_heads(&head_outputs);

        // Proyección final
        let output = concat_output.dot(&self.w_o);

        (output, attention_weights)
    }

    /// Obtiene información sobre la arquitectura
    pub fn info(&self) -> String {
        format!(
            "MultiHeadAttention(heads={}, d_model={}, d_k={})",
            self.num_heads, self.d_model, self.d_k
        )
    }
}

/// Concatena las salidas de múltiples cabezas de atención
///
/// # Argumentos
/// * `heads` - Vector de salidas de cabezas (cada una es seq_len x d_k)
///
/// # Retorna
/// Array concatenado (seq_len x (num_heads * d_k))
fn concatenate_heads(heads: &[Array2<f32>]) -> Array2<f32> {
    if heads.is_empty() {
        panic!("No hay cabezas para concatenar");
    }

    let seq_len = heads[0].shape()[0];
    let d_k = heads[0].shape()[1];
    let num_heads = heads.len();

    // Crear array de salida
    let mut output = Array2::zeros((seq_len, num_heads * d_k));

    // Copiar cada cabeza en su posición correspondiente
    for (i, head) in heads.iter().enumerate() {
        let start = i * d_k;
        let end = (i + 1) * d_k;

        for row in 0..seq_len {
            for col in 0..d_k {
                output[[row, start + col]] = head[[row, col]];
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_multihead_attention_creation() {
        let mha = MultiHeadAttention::new(64, 8);
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.d_model, 64);
        assert_eq!(mha.d_k, 8);
        assert_eq!(mha.w_q.len(), 8);
    }

    #[test]
    #[should_panic(expected = "debe ser divisible por")]
    fn test_multihead_attention_invalid_dimensions() {
        // 64 no es divisible por 7
        MultiHeadAttention::new(64, 7);
    }

    #[test]
    fn test_concatenate_heads() {
        let head1 = array![[1.0, 2.0], [3.0, 4.0]];
        let head2 = array![[5.0, 6.0], [7.0, 8.0]];

        let result = concatenate_heads(&[head1, head2]);

        assert_eq!(result.shape(), &[2, 4]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 2]], 5.0);
    }

    #[test]
    fn test_multihead_forward() {
        let mha = MultiHeadAttention::new(16, 4);
        let input = Array2::ones((5, 16)); // seq_len=5, d_model=16

        let (output, weights) = mha.forward(&input, None);

        assert_eq!(output.shape(), &[5, 16]);
        assert_eq!(weights.len(), 4); // 4 cabezas

        // Cada cabeza debe tener pesos de atención de forma (seq_len, seq_len)
        for weight in weights {
            assert_eq!(weight.shape(), &[5, 5]);

            // Los pesos deben sumar 1 por fila
            for i in 0..5 {
                let row_sum: f32 = weight.row(i).sum();
                assert!((row_sum - 1.0).abs() < 1e-5);
            }
        }
    }
}
