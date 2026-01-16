// Positional Encoding
//
// Codificación posicional sinusoidal para inyectar información
// sobre la posición de los tokens en la secuencia

use ndarray::Array2;

/// Genera codificación posicional sinusoidal
///
/// Usa la fórmula del paper "Attention is All You Need":
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// # Argumentos
/// * `max_len` - Longitud máxima de la secuencia
/// * `d_model` - Dimensión del modelo (embedding)
///
/// # Retorna
/// Matriz de codificación posicional (max_len, d_model)
pub fn sinusoidal_positional_encoding(max_len: usize, d_model: usize) -> Array2<f32> {
    let mut pe = Array2::zeros((max_len, d_model));

    for pos in 0..max_len {
        for i in 0..d_model {
            let angle = pos as f32 / 10000_f32.powf((2 * (i / 2)) as f32 / d_model as f32);

            if i % 2 == 0 {
                // Posiciones pares: seno
                pe[[pos, i]] = angle.sin();
            } else {
                // Posiciones impares: coseno
                pe[[pos, i]] = angle.cos();
            }
        }
    }

    pe
}

/// Genera codificación posicional aprendible (inicializada aleatoriamente)
///
/// A diferencia de la codificación sinusoidal fija, esta puede entrenarse.
///
/// # Argumentos
/// * `max_len` - Longitud máxima de la secuencia
/// * `d_model` - Dimensión del modelo
///
/// # Retorna
/// Matriz de codificación posicional (max_len, d_model)
pub fn learnable_positional_encoding(max_len: usize, d_model: usize) -> Array2<f32> {
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;

    // Inicialización con distribución normal pequeña
    Array2::random((max_len, d_model), Normal::new(0.0, 0.02).unwrap())
}

/// Añade codificación posicional a embeddings
///
/// # Argumentos
/// * `embeddings` - Embeddings de entrada (batch_size, seq_len, d_model) o (seq_len, d_model)
/// * `pos_encoding` - Codificación posicional (max_len, d_model)
///
/// # Retorna
/// Embeddings con posición añadida
pub fn add_positional_encoding(
    embeddings: &Array2<f32>,
    pos_encoding: &Array2<f32>,
) -> Array2<f32> {
    let seq_len = embeddings.shape()[0];
    let d_model = embeddings.shape()[1];

    assert!(
        seq_len <= pos_encoding.shape()[0],
        "Secuencia muy larga para el positional encoding disponible"
    );
    assert_eq!(
        d_model,
        pos_encoding.shape()[1],
        "d_model debe coincidir entre embeddings y pos_encoding"
    );

    // Tomar solo las primeras seq_len posiciones
    let pe_slice = pos_encoding.slice(ndarray::s![0..seq_len, ..]).to_owned();

    embeddings + &pe_slice
}

/// Estructura para manejar codificación posicional
pub struct PositionalEncoding {
    /// Codificación posicional precomputada
    encoding: Array2<f32>,
    /// Tipo de codificación (sinusoidal o aprendible)
    encoding_type: EncodingType,
}

/// Tipo de codificación posicional
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EncodingType {
    /// Codificación sinusoidal fija (no entrenable)
    Sinusoidal,
    /// Codificación aprendible (entrenable)
    Learnable,
}

impl PositionalEncoding {
    /// Crea una nueva codificación posicional sinusoidal
    ///
    /// # Argumentos
    /// * `max_len` - Longitud máxima de la secuencia
    /// * `d_model` - Dimensión del modelo
    pub fn new_sinusoidal(max_len: usize, d_model: usize) -> Self {
        Self {
            encoding: sinusoidal_positional_encoding(max_len, d_model),
            encoding_type: EncodingType::Sinusoidal,
        }
    }

    /// Crea una nueva codificación posicional aprendible
    ///
    /// # Argumentos
    /// * `max_len` - Longitud máxima de la secuencia
    /// * `d_model` - Dimensión del modelo
    pub fn new_learnable(max_len: usize, d_model: usize) -> Self {
        Self {
            encoding: learnable_positional_encoding(max_len, d_model),
            encoding_type: EncodingType::Learnable,
        }
    }

    /// Aplica la codificación posicional a los embeddings
    ///
    /// # Argumentos
    /// * `embeddings` - Embeddings de entrada (seq_len, d_model)
    ///
    /// # Retorna
    /// Embeddings con posición añadida
    pub fn apply(&self, embeddings: &Array2<f32>) -> Array2<f32> {
        add_positional_encoding(embeddings, &self.encoding)
    }

    /// Obtiene la codificación para una posición específica
    ///
    /// # Argumentos
    /// * `pos` - Posición en la secuencia
    ///
    /// # Retorna
    /// Vector de codificación para esa posición
    pub fn get_position(&self, pos: usize) -> ndarray::ArrayView1<f32> {
        assert!(pos < self.encoding.shape()[0], "Posición fuera de rango");
        self.encoding.row(pos)
    }

    /// Devuelve el tipo de codificación
    pub fn encoding_type(&self) -> EncodingType {
        self.encoding_type
    }

    /// Devuelve la matriz de codificación completa
    pub fn encoding(&self) -> &Array2<f32> {
        &self.encoding
    }

    /// Información sobre la codificación
    pub fn info(&self) -> String {
        format!(
            "PositionalEncoding(type={:?}, max_len={}, d_model={})",
            self.encoding_type,
            self.encoding.shape()[0],
            self.encoding.shape()[1]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sinusoidal_encoding() {
        let pe = sinusoidal_positional_encoding(10, 8);

        assert_eq!(pe.shape(), &[10, 8]);

        // Verificar que los valores estén en rango [-1, 1] (seno y coseno)
        for &val in pe.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }

        // Primera posición, primera dimensión debe ser sin(0) = 0
        assert!((pe[[0, 0]] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_learnable_encoding() {
        let pe = learnable_positional_encoding(10, 8);

        assert_eq!(pe.shape(), &[10, 8]);

        // Los valores deben ser pequeños (inicialización con std=0.02)
        for &val in pe.iter() {
            assert!(val.abs() < 1.0); // Muy probable con std=0.02
        }
    }

    #[test]
    fn test_add_positional_encoding() {
        let embeddings = Array2::ones((5, 8));
        let pe = sinusoidal_positional_encoding(10, 8);

        let result = add_positional_encoding(&embeddings, &pe);

        assert_eq!(result.shape(), &[5, 8]);

        // Los valores deben ser diferentes a los embeddings originales
        assert_ne!(result, embeddings);
    }

    #[test]
    fn test_positional_encoding_struct() {
        let pe = PositionalEncoding::new_sinusoidal(10, 8);

        assert_eq!(pe.encoding_type(), EncodingType::Sinusoidal);
        assert_eq!(pe.encoding().shape(), &[10, 8]);

        let embeddings = Array2::ones((5, 8));
        let result = pe.apply(&embeddings);

        assert_eq!(result.shape(), &[5, 8]);
    }

    #[test]
    fn test_get_position() {
        let pe = PositionalEncoding::new_sinusoidal(10, 8);

        let pos_0 = pe.get_position(0);
        let pos_1 = pe.get_position(1);

        // Las posiciones deben ser diferentes
        assert_ne!(pos_0.to_vec(), pos_1.to_vec());
    }

    #[test]
    #[should_panic(expected = "Posición fuera de rango")]
    fn test_get_position_out_of_range() {
        let pe = PositionalEncoding::new_sinusoidal(10, 8);
        pe.get_position(10); // Fuera de rango
    }
}
