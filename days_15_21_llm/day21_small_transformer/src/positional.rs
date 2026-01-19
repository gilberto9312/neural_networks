// Positional Encoding para Transformers

use ndarray::{Array1, Array2};

/// Positional Encoding Sinusoidal
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
pub struct PositionalEncoding {
    encoding: Array2<f32>,
    max_len: usize,
    d_model: usize,
}

impl PositionalEncoding {
    /// Crea positional encoding sinusoidal
    ///
    /// # Argumentos
    /// * `max_len` - Longitud máxima de secuencia
    /// * `d_model` - Dimensión del modelo
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let mut encoding = Array2::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let angle = pos as f32 / 10000_f32.powf((2 * i) as f32 / d_model as f32);
                
                // Sin para dimensiones pares
                encoding[[pos, 2 * i]] = angle.sin();
                
                // Cos para dimensiones impares
                if 2 * i + 1 < d_model {
                    encoding[[pos, 2 * i + 1]] = angle.cos();
                }
            }
        }

        Self {
            encoding,
            max_len,
            d_model,
        }
    }

    /// Aplica positional encoding a embeddings
    ///
    /// # Argumentos
    /// * `x` - Embeddings (seq_len, d_model)
    ///
    /// # Retorna
    /// Embeddings + positional encoding
    pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.nrows();
        assert!(seq_len <= self.max_len, "Secuencia demasiado larga");
        assert_eq!(x.ncols(), self.d_model, "Dimensiones incompatibles");

        let mut result = x.clone();
        
        for i in 0..seq_len {
            for j in 0..self.d_model {
                result[[i, j]] += self.encoding[[i, j]];
            }
        }

        result
    }

    /// Obtiene el encoding para una posición específica
    pub fn get_position(&self, pos: usize) -> Array1<f32> {
        assert!(pos < self.max_len);
        self.encoding.row(pos).to_owned()
    }

    /// Retorna el encoding completo
    pub fn get_encoding(&self) -> &Array2<f32> {
        &self.encoding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encoding() {
        let max_len = 100;
        let d_model = 128;
        let pe = PositionalEncoding::new(max_len, d_model);

        assert_eq!(pe.encoding.shape(), &[max_len, d_model]);

        // Test aplicar
        let x = Array2::ones((10, d_model));
        let result = pe.apply(&x);
        
        assert_eq!(result.shape(), &[10, d_model]);
    }

    #[test]
    fn test_get_position() {
        let pe = PositionalEncoding::new(50, 64);
        let pos_0 = pe.get_position(0);
        let pos_10 = pe.get_position(10);

        assert_eq!(pos_0.len(), 64);
        assert_eq!(pos_10.len(), 64);
        
        // Las posiciones deben ser diferentes
        assert_ne!(pos_0.sum(), pos_10.sum());
    }
}
