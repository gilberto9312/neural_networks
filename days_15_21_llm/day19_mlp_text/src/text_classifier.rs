// text_classifier.rs
// Clasificador de texto con capa de embeddings + MLP

use crate::mlp::{MLP, softmax};
use crate::batch::{Batch, average_embeddings};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Capa de Embeddings
pub struct EmbeddingLayer {
    pub embeddings: Array2<f32>,  // (vocab_size, embedding_dim)
    vocab_size: usize,
    embedding_dim: usize,
    // Gradientes acumulados para actualización
    grad_embeddings: Array2<f32>,
}

impl EmbeddingLayer {
    /// Crea una nueva capa de embeddings con inicialización aleatoria
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Inicialización aleatoria pequeña
        let embeddings = Array2::random(
            (vocab_size, embedding_dim),
            Uniform::new(-0.1, 0.1)
        );
        let grad_embeddings = Array2::zeros((vocab_size, embedding_dim));

        EmbeddingLayer {
            embeddings,
            vocab_size,
            embedding_dim,
            grad_embeddings,
        }
    }

    /// Lookup de embeddings para una lista de token IDs
    /// Retorna el promedio de embeddings
    pub fn forward(&self, token_ids: &[usize]) -> Array1<f32> {
        let avg = average_embeddings(&self.embeddings, token_ids);
        Array1::from_vec(avg)
    }

    /// Forward para un batch completo
    pub fn forward_batch(&self, batch_tokens: &[Vec<usize>]) -> Array2<f32> {
        let batch_size = batch_tokens.len();
        let mut result = Array2::zeros((batch_size, self.embedding_dim));

        for (i, tokens) in batch_tokens.iter().enumerate() {
            let avg = self.forward(tokens);
            for (j, &val) in avg.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        result
    }

    /// Backward pass para actualizar embeddings
    pub fn backward(
        &mut self,
        batch_tokens: &[Vec<usize>],
        grad_output: &Array2<f32>,
        learning_rate: f32,
    ) {
        // Reiniciar gradientes acumulados
        self.grad_embeddings.fill(0.0);

        // Acumular gradientes para cada ejemplo en el batch
        for (i, tokens) in batch_tokens.iter().enumerate() {
            let count = tokens.len() as f32;
            if count == 0.0 {
                continue;
            }

            // El gradiente se distribuye igualmente entre todos los tokens
            for &token_id in tokens {
                if token_id < self.vocab_size {
                    for j in 0..self.embedding_dim {
                        self.grad_embeddings[[token_id, j]] += grad_output[[i, j]] / count;
                    }
                }
            }
        }

        // Actualizar embeddings
        self.embeddings = &self.embeddings - &(&self.grad_embeddings * learning_rate);
    }
}

/// Clasificador de texto completo: Embedding + MLP
pub struct TextClassifier {
    embedding_layer: EmbeddingLayer,
    mlp: MLP,
    num_classes: usize,
}

impl TextClassifier {
    /// Crea un nuevo clasificador de texto
    ///
    /// # Argumentos
    /// * `vocab_size` - Tamaño del vocabulario
    /// * `embedding_dim` - Dimensión de los embeddings
    /// * `hidden_dims` - Dimensiones de las capas ocultas del MLP
    /// * `num_classes` - Número de clases para clasificación
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        hidden_dims: Vec<usize>,
        num_classes: usize,
    ) -> Self {
        let embedding_layer = EmbeddingLayer::new(vocab_size, embedding_dim);

        // Construir arquitectura del MLP: embedding_dim -> hidden -> ... -> num_classes
        let mut layer_sizes = vec![embedding_dim];
        layer_sizes.extend(hidden_dims);
        layer_sizes.push(num_classes);

        let mlp = MLP::new(&layer_sizes);

        TextClassifier {
            embedding_layer,
            mlp,
            num_classes,
        }
    }

    /// Forward pass completo
    pub fn forward(&mut self, batch: &Batch) -> Array2<f32> {
        // 1. Pasar por la capa de embeddings
        let embeddings = self.embedding_layer.forward_batch(&batch.inputs);

        // 2. Pasar por el MLP
        let logits = self.mlp.forward(&embeddings);

        // 3. Aplicar Softmax para obtener probabilidades
        softmax(&logits)
    }

    /// Entrena el modelo con un batch
    ///
    /// # Retorna
    /// La pérdida (cross-entropy) del batch
    pub fn train(&mut self, batch: &Batch, learning_rate: f32) -> f32 {
        // Forward pass (sin softmax para entrenamiento)
        let embeddings = self.embedding_layer.forward_batch(&batch.inputs);
        let logits = self.mlp.forward(&embeddings);

        // Aplicar softmax para calcular pérdida
        let predictions = softmax(&logits);
        let loss = cross_entropy_loss(&predictions, &batch.targets);

        // Backward pass: gradiente de cross-entropy + softmax
        // Para cross-entropy con softmax, el gradiente es: (predictions - one_hot(targets)) / batch_size
        let batch_size = batch.targets.len() as f32;
        let mut grad_logits = predictions.clone();

        for (i, &target) in batch.targets.iter().enumerate() {
            if i < grad_logits.nrows() && target < grad_logits.ncols() {
                grad_logits[[i, target]] -= 1.0;
            }
        }

        // Normalizar por batch size
        grad_logits = grad_logits / batch_size;

        // Backprop a través del MLP (retorna gradiente respecto a embeddings)
        let grad_embeddings = self.mlp.backward(grad_logits, learning_rate);

        // Backprop a través de embeddings
        self.embedding_layer.backward(&batch.inputs, &grad_embeddings, learning_rate);

        loss
    }

    /// Predice la clase para un texto (representado como token IDs)
    pub fn predict(&self, token_ids: &[usize]) -> usize {
        let embedding = self.embedding_layer.forward(token_ids);
        let embedding_batch = embedding.insert_axis(ndarray::Axis(0));
        let logits = self.mlp.predict(&embedding_batch);

        // Aplicar softmax y obtener la clase con mayor probabilidad
        let probs = softmax(&logits);
        let row = probs.row(0);

        // Encontrar el índice con mayor probabilidad
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (idx, &val) in row.iter().enumerate() {
            if val.is_finite() && val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        max_idx
    }

    /// Calcula la accuracy en un conjunto de datos
    pub fn accuracy(&self, data: &[(Vec<usize>, usize)]) -> f32 {
        let mut correct = 0;
        let total = data.len();

        for (tokens, true_label) in data {
            let predicted = self.predict(tokens);
            if predicted == *true_label {
                correct += 1;
            }
        }

        correct as f32 / total as f32
    }
}

/// Calcula la pérdida Cross-Entropy
///
/// # Argumentos
/// * `predictions` - Probabilidades predichas (batch_size, num_classes) después de softmax
/// * `targets` - Índices de las clases verdaderas
pub fn cross_entropy_loss(predictions: &Array2<f32>, targets: &[usize]) -> f32 {
    let batch_size = predictions.nrows();
    let mut total_loss = 0.0;

    for (i, &target) in targets.iter().enumerate() {
        if i >= batch_size {
            break;
        }
        // -log(p_target)
        let prob = predictions[[i, target]].max(1e-10); // Evitar log(0)
        total_loss -= prob.ln();
    }

    total_loss / batch_size as f32
}

/// Calcula el gradiente de Cross-Entropy respecto a las predicciones
///
/// Para Cross-Entropy con Softmax, el gradiente es simplemente:
/// grad = predictions - one_hot(targets)
pub fn cross_entropy_gradient(predictions: &Array2<f32>, targets: &[usize]) -> Array2<f32> {
    let batch_size = predictions.nrows();
    let num_classes = predictions.ncols();
    let mut grad = predictions.clone();

    for (i, &target) in targets.iter().enumerate() {
        if i >= batch_size {
            break;
        }
        grad[[i, target]] -= 1.0;
    }

    // Normalizar por el tamaño del batch
    grad / batch_size as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_layer() {
        let emb = EmbeddingLayer::new(100, 64);
        let tokens = vec![1, 5, 10];
        let result = emb.forward(&tokens);

        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_text_classifier() {
        let mut classifier = TextClassifier::new(100, 32, vec![16], 3);

        let mut batch = Batch::new();
        batch.add(vec![1, 2, 3], 0);
        batch.add(vec![4, 5], 1);

        let predictions = classifier.forward(&batch);

        assert_eq!(predictions.shape(), &[2, 3]);

        // La suma de cada fila debe ser ~1.0 (softmax)
        for row in predictions.axis_iter(ndarray::Axis(0)) {
            let sum: f32 = row.sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cross_entropy() {
        let predictions = Array2::from_shape_vec(
            (2, 3),
            vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1]
        ).unwrap();

        let targets = vec![0, 1];
        let loss = cross_entropy_loss(&predictions, &targets);

        assert!(loss > 0.0);
        assert!(loss < 1.0); // Pérdida baja porque las predicciones son correctas
    }
}
