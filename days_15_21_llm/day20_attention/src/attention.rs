// Scaled Dot-Product Attention
//
// Implementación del mecanismo de atención básico:
// Attention(Q,K,V) = softmax(QK^T / √d_k)V

use ndarray::Array2;

/// Aplica la función softmax a una matriz a lo largo de la última dimensión
///
/// # Argumentos
/// * `x` - Array de entrada
///
/// # Retorna
/// Array con softmax aplicado
pub fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let max_vals = x.map_axis(ndarray::Axis(1), |row| {
        row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    });

    let exp_x = x.clone() - &max_vals.insert_axis(ndarray::Axis(1));
    let exp_x = exp_x.mapv(f32::exp);

    let sum_exp = exp_x.sum_axis(ndarray::Axis(1));

    exp_x / &sum_exp.insert_axis(ndarray::Axis(1))
}

/// Scaled Dot-Product Attention
///
/// Calcula la atención usando la fórmula:
/// Attention(Q, K, V) = softmax(QK^T / √d_k)V
///
/// # Argumentos
/// * `queries` - Matriz Q (seq_len_q, d_k)
/// * `keys` - Matriz K (seq_len_k, d_k)
/// * `values` - Matriz V (seq_len_v, d_v)
/// * `mask` - Máscara opcional para evitar atención a ciertas posiciones
///
/// # Retorna
/// Tupla (output, attention_weights)
/// * output: Resultado de la atención (seq_len_q, d_v)
/// * attention_weights: Pesos de atención (seq_len_q, seq_len_k)
pub fn scaled_dot_product_attention(
    queries: &Array2<f32>,
    keys: &Array2<f32>,
    values: &Array2<f32>,
    mask: Option<&Array2<f32>>,
) -> (Array2<f32>, Array2<f32>) {
    // d_k es la dimensión de las keys
    let d_k = keys.shape()[1] as f32;

    // Calcular QK^T
    let scores = queries.dot(&keys.t());

    // Escalar por √d_k
    let scaled_scores = scores / d_k.sqrt();

    // Aplicar máscara si existe (poner -inf en posiciones enmascaradas)
    let masked_scores = if let Some(m) = mask {
        scaled_scores + m
    } else {
        scaled_scores
    };

    // Aplicar softmax para obtener pesos de atención
    let attention_weights = softmax(&masked_scores);

    // Multiplicar por valores
    let output = attention_weights.dot(values);

    (output, attention_weights)
}

/// Estructura para el mecanismo de atención con parámetros entrenables
pub struct Attention {
    /// Dimensión de entrada
    pub d_model: usize,
    /// Dimensión de las queries y keys
    pub d_k: usize,
    /// Dimensión de los values
    pub d_v: usize,
    /// Matriz de proyección para queries (d_model, d_k)
    pub w_q: Array2<f32>,
    /// Matriz de proyección para keys (d_model, d_k)
    pub w_k: Array2<f32>,
    /// Matriz de proyección para values (d_model, d_v)
    pub w_v: Array2<f32>,
    /// Matriz de proyección de salida (d_v, d_model)
    pub w_o: Array2<f32>,
}

impl Attention {
    /// Crea una nueva capa de atención con parámetros aleatorios
    ///
    /// # Argumentos
    /// * `d_model` - Dimensión del modelo
    /// * `d_k` - Dimensión de queries y keys
    /// * `d_v` - Dimensión de values
    pub fn new(d_model: usize, d_k: usize, d_v: usize) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;

        // Inicialización Xavier/Glorot
        let limit = (6.0 / (d_model + d_k) as f32).sqrt();

        Self {
            d_model,
            d_k,
            d_v,
            w_q: Array2::random((d_model, d_k), Uniform::new(-limit, limit)),
            w_k: Array2::random((d_model, d_k), Uniform::new(-limit, limit)),
            w_v: Array2::random((d_model, d_v), Uniform::new(-limit, limit)),
            w_o: Array2::random((d_v, d_model), Uniform::new(-limit, limit)),
        }
    }

    /// Forward pass de la atención
    ///
    /// # Argumentos
    /// * `input` - Entrada (seq_len, d_model)
    /// * `mask` - Máscara opcional
    ///
    /// # Retorna
    /// Tupla (output, attention_weights)
    pub fn forward(
        &self,
        input: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> (Array2<f32>, Array2<f32>) {
        // Proyectar entrada a Q, K, V
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);

        // Aplicar atención
        let (attn_output, attn_weights) = scaled_dot_product_attention(&q, &k, &v, mask);

        // Proyección final
        let output = attn_output.dot(&self.w_o);

        (output, attn_weights)
    }
}

/// Constante para máscara de atención
/// Valor usado en los logits enmascarados para evitar atención
/// Este valor específico es usado en Gemma y otros modelos de producción
pub const K_MASK: f32 = -2.3819763e38;

/// Crea una máscara causal para atención (máscara triangular superior)
///
/// Previene que las posiciones atiendan a posiciones futuras.
/// Útil para modelos autorregresivos como decoders.
///
/// # Argumentos
/// * `size` - Tamaño de la secuencia
///
/// # Retorna
/// Máscara (size, size) con K_MASK en la parte superior
pub fn create_causal_mask(size: usize) -> Array2<f32> {
    let mut mask = Array2::zeros((size, size));

    for i in 0..size {
        for j in (i + 1)..size {
            mask[[i, j]] = K_MASK;
        }
    }

    mask
}

/// Crea una máscara de padding
///
/// Previene que se atienda a tokens de padding.
///
/// # Argumentos
/// * `seq_len` - Longitud de la secuencia
/// * `actual_len` - Longitud real (sin padding)
///
/// # Retorna
/// Máscara (seq_len, seq_len) con K_MASK en posiciones de padding
pub fn create_padding_mask(seq_len: usize, actual_len: usize) -> Array2<f32> {
    let mut mask = Array2::zeros((seq_len, seq_len));

    if actual_len < seq_len {
        for i in 0..seq_len {
            for j in actual_len..seq_len {
                mask[[i, j]] = K_MASK;
            }
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_softmax() {
        let x = array![[1.0, 2.0, 3.0]];
        let result = softmax(&x);

        // La suma debe ser 1
        let sum: f32 = result.sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Los valores deben estar entre 0 y 1
        assert!(result.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let q = array![[1.0, 0.0], [0.0, 1.0]];
        let k = array![[1.0, 0.0], [0.0, 1.0]];
        let v = array![[1.0, 2.0], [3.0, 4.0]];

        let (output, weights) = scaled_dot_product_attention(&q, &k, &v, None);

        // Verificar dimensiones
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(weights.shape(), &[2, 2]);

        // Los pesos deben sumar 1 por fila
        for i in 0..weights.shape()[0] {
            let row_sum: f32 = weights.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3);

        // La diagonal y debajo deben ser 0
        assert_eq!(mask[[0, 0]], 0.0);
        assert_eq!(mask[[1, 0]], 0.0);
        assert_eq!(mask[[1, 1]], 0.0);

        // Arriba de la diagonal debe ser K_MASK
        assert_eq!(mask[[0, 1]], K_MASK);
        assert_eq!(mask[[0, 2]], K_MASK);
    }
}
