// Utilidades matemáticas para el transformer

use ndarray::{Array1, Array2, Axis};

/// Softmax aplicado a la última dimensión de una matriz
///
/// # Argumentos
/// * `x` - Matriz de entrada (seq_len, seq_len) o (batch, seq_len, seq_len)
///
/// # Retorna
/// Matriz con softmax aplicado, mismas dimensiones que entrada
pub fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros(x.dim());

    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Array1<f32> = row.mapv(|v| (v - max_val).exp());
        let sum: f32 = exp_vals.sum();

        for (j, &val) in exp_vals.iter().enumerate() {
            result[[i, j]] = val / sum;
        }
    }

    result
}

/// Crea una máscara causal para prevenir que el modelo vea tokens futuros
///
/// # Argumentos
/// * `seq_len` - Longitud de la secuencia
///
/// # Retorna
/// Matriz (seq_len, seq_len) donde:
/// - 0.0 indica que el token es visible
/// - -inf indica que el token está enmascarado
pub fn create_causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::from_elem((seq_len, seq_len), f32::NEG_INFINITY);

    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 0.0;
        }
    }

    mask
}

/// Calcula la norma L2 de un vector
pub fn l2_norm(x: &Array1<f32>) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Inicializa pesos usando Xavier/Glorot initialization
///
/// # Argumentos
/// * `shape` - Forma de la matriz (rows, cols)
/// * `rng` - Generador de números aleatorios
///
/// # Retorna
/// Matriz inicializada
pub fn xavier_init(shape: (usize, usize), rng: &mut impl rand::Rng) -> Array2<f32> {
    use rand::distributions::{Distribution, Uniform};

    let (fan_in, fan_out) = shape;
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let dist = Uniform::new(-limit, limit);

    Array2::from_shape_fn(shape, |_| dist.sample(rng))
}

/// Calcula cross-entropy loss entre predicciones y targets
///
/// # Argumentos
/// * `logits` - Predicciones del modelo (seq_len, vocab_size)
/// * `targets` - Índices de tokens target (seq_len,)
///
/// # Retorna
/// Loss promedio
pub fn cross_entropy_loss(logits: &Array2<f32>, targets: &[usize]) -> f32 {
    let mut total_loss = 0.0;
    let epsilon = 1e-10; // Para estabilidad numérica

    for (i, &target_idx) in targets.iter().enumerate() {
        if i >= logits.nrows() {
            break;
        }

        // Aplicar softmax a logits
        let row = logits.row(i);
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&v| v / sum).collect();

        // Cross-entropy: -log(p_target)
        if target_idx < probs.len() {
            total_loss -= (probs[target_idx] + epsilon).ln();
        }
    }

    total_loss / targets.len() as f32
}

/// Calcula perplejidad desde el loss
///
/// # Argumentos
/// * `loss` - Cross-entropy loss
///
/// # Retorna
/// Perplejidad
pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_softmax() {
        let x = Array2::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            1.0, 1.0, 1.0,
        ]).unwrap();

        let result = softmax(&x);

        // Verificar que cada fila suma 1.0
        for row in result.axis_iter(Axis(0)) {
            let sum: f32 = row.sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3);

        // Verificar diagonal inferior es 0.0
        assert_eq!(mask[[0, 0]], 0.0);
        assert_eq!(mask[[1, 0]], 0.0);
        assert_eq!(mask[[1, 1]], 0.0);
        assert_eq!(mask[[2, 0]], 0.0);
        assert_eq!(mask[[2, 1]], 0.0);
        assert_eq!(mask[[2, 2]], 0.0);

        // Verificar diagonal superior es -inf
        assert!(mask[[0, 1]].is_infinite() && mask[[0, 1]].is_sign_negative());
        assert!(mask[[0, 2]].is_infinite() && mask[[0, 2]].is_sign_negative());
        assert!(mask[[1, 2]].is_infinite() && mask[[1, 2]].is_sign_negative());
    }

    #[test]
    fn test_l2_norm() {
        let x = arr1(&[3.0, 4.0]);
        assert_eq!(l2_norm(&x), 5.0);
    }
}
