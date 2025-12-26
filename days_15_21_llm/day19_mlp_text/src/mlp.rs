// mlp.rs
// Implementación de Multi-Layer Perceptron (MLP)

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Capa individual de la red neuronal
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    // Guardamos valores para backpropagation
    last_input: Option<Array2<f32>>,
    last_z: Option<Array2<f32>>,      // Antes de activación
    last_output: Option<Array2<f32>>, // Después de activación
}

impl Layer {
    /// Crea una nueva capa con inicialización aleatoria He
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Inicialización He para ReLU
        let scale = (2.0 / input_size as f32).sqrt();
        let weights = Array2::random(
            (input_size, output_size),
            Uniform::new(-scale, scale)
        );
        let biases = Array1::zeros(output_size);

        Layer {
            weights,
            biases,
            last_input: None,
            last_z: None,
            last_output: None,
        }
    }

    /// Forward pass: calcula la suma ponderada
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Guardar input para backprop
        self.last_input = Some(input.clone());

        // z = X * W + b
        let z = input.dot(&self.weights) + &self.biases;
        self.last_z = Some(z.clone());

        z
    }

    /// Actualiza los pesos usando los gradientes
    pub fn update_weights(
        &mut self,
        grad_weights: &Array2<f32>,
        grad_biases: &Array1<f32>,
        learning_rate: f32,
    ) {
        self.weights = &self.weights - &(grad_weights * learning_rate);
        self.biases = &self.biases - &(grad_biases * learning_rate);
    }
}

/// Red neuronal multi-capa (MLP)
pub struct MLP {
    layers: Vec<Layer>,
    use_relu: Vec<bool>, // true para ReLU, false para ninguna activación
}

impl MLP {
    /// Crea un nuevo MLP con las dimensiones especificadas
    ///
    /// # Argumentos
    /// * `layer_sizes` - Lista de tamaños [input, hidden1, hidden2, ..., output]
    ///
    /// # Ejemplo
    /// ```
    /// // Red con entrada de 128, capa oculta de 64, salida de 3
    /// let mlp = MLP::new(&[128, 64, 3]);
    /// ```
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut use_relu = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
            // ReLU en capas ocultas, ninguna activación en la última
            use_relu.push(i < layer_sizes.len() - 2);
        }

        MLP { layers, use_relu }
    }

    /// Forward pass a través de todas las capas
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mut current = input.clone();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            current = layer.forward(&current);

            // Aplicar ReLU si corresponde
            if self.use_relu[i] {
                current = relu(&current);
            }

            // Guardar output de esta capa
            layer.last_output = Some(current.clone());
        }

        current
    }

    /// Backward pass con gradientes
    ///
    /// # Retorna
    /// Gradiente con respecto a la entrada del MLP
    pub fn backward(&mut self, mut grad_output: Array2<f32>, learning_rate: f32) -> Array2<f32> {
        // Recorrer capas en reversa
        for i in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[i];

            // Si usamos ReLU en esta capa, aplicar su derivada al gradiente entrante
            // La derivada debe aplicarse sobre la salida de la capa (después de ReLU)
            if self.use_relu[i] {
                if let Some(ref output) = layer.last_output {
                    // ReLU derivative: 1 si output > 0, 0 si output <= 0
                    grad_output = grad_output * output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                }
            }

            // Calcular gradientes de pesos y biases
            if let Some(ref input) = layer.last_input {
                // grad_W = X^T * grad_output
                let grad_weights = input.t().dot(&grad_output);

                // grad_b = sum(grad_output, axis=0)
                let grad_biases = grad_output.sum_axis(ndarray::Axis(0));

                // Actualizar pesos
                layer.update_weights(&grad_weights, &grad_biases, learning_rate);

                // Propagar gradiente hacia atrás (antes de la no-linealidad de la capa anterior)
                grad_output = grad_output.dot(&layer.weights.t());
            }
        }

        // Retornar el gradiente con respecto a la entrada
        grad_output
    }

    /// Predice la salida (sin guardar estados intermedios)
    pub fn predict(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut current = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            current = current.dot(&layer.weights) + &layer.biases;

            if self.use_relu[i] {
                current = relu(&current);
            }
        }

        current
    }
}

/// Función de activación ReLU
pub fn relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Derivada de ReLU
pub fn relu_derivative(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

/// Función Softmax para clasificación multi-clase
pub fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros(x.dim());

    for (i, row) in x.axis_iter(ndarray::Axis(0)).enumerate() {
        // Restar el máximo para estabilidad numérica
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_row: Array1<f32> = row.mapv(|v| (v - max_val).exp());
        let sum_exp: f32 = exp_row.sum();

        for (j, &val) in exp_row.iter().enumerate() {
            result[[i, j]] = val / sum_exp;
        }
    }

    result
}

/// Función Sigmoid
pub fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -2.0, 3.0, 0.5]).unwrap();
        let result = relu(&x);
        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[0, 1]], 0.0);
        assert_eq!(result[[0, 2]], 1.0);
    }

    #[test]
    fn test_softmax() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]).unwrap();
        let result = softmax(&x);

        // La suma de cada fila debe ser 1.0
        for row in result.axis_iter(ndarray::Axis(0)) {
            let sum: f32 = row.sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
