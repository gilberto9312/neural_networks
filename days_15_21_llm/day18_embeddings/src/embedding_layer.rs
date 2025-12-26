// Capa de embeddings (lookup table)
// Implementación de matriz de embeddings y operaciones básicas

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Estructura que representa una capa de embeddings
/// Cada fila de la matriz es el vector embedding de un token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLayer {
    /// Matriz de embeddings (vocab_size x embedding_dim)
    pub embeddings: Array2<f32>,
    /// Mapeo de token a índice
    pub token_to_id: HashMap<String, usize>,
    /// Mapeo de índice a token
    pub id_to_token: Vec<String>,
    /// Dimensión de cada embedding
    pub embedding_dim: usize,
}

impl EmbeddingLayer {
    /// Crea una nueva capa de embeddings con inicialización aleatoria
    ///
    /// # Argumentos
    /// * `vocab` - Lista de tokens del vocabulario
    /// * `embedding_dim` - Dimensión de cada vector embedding
    pub fn new(vocab: Vec<String>, embedding_dim: usize) -> Self {
        let vocab_size = vocab.len();

        // Inicializar embeddings con valores aleatorios entre -0.5 y 0.5
        let embeddings = Array2::random(
            (vocab_size, embedding_dim),
            Uniform::new(-0.5, 0.5)
        );

        // Crear mapeos token <-> id
        let mut token_to_id = HashMap::new();
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), id);
        }

        EmbeddingLayer {
            embeddings,
            token_to_id,
            id_to_token: vocab,
            embedding_dim,
        }
    }

    /// Obtiene el embedding de un token por su nombre
    ///
    /// # Argumentos
    /// * `token` - Token del cual obtener el embedding
    ///
    /// # Retorna
    /// El vector embedding del token, o None si no existe
    pub fn get_embedding(&self, token: &str) -> Option<Array1<f32>> {
        self.token_to_id.get(token).map(|&id| {
            self.embeddings.row(id).to_owned()
        })
    }

    /// Obtiene el embedding de un token por su índice
    ///
    /// # Argumentos
    /// * `id` - Índice del token
    ///
    /// # Retorna
    /// El vector embedding del token
    pub fn get_embedding_by_id(&self, id: usize) -> Array1<f32> {
        self.embeddings.row(id).to_owned()
    }

    /// Obtiene el índice de un token
    pub fn get_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    /// Obtiene el token de un índice
    pub fn get_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    /// Actualiza el embedding de un token
    ///
    /// # Argumentos
    /// * `token` - Token a actualizar
    /// * `new_embedding` - Nuevo vector embedding
    pub fn update_embedding(&mut self, token: &str, new_embedding: &Array1<f32>) {
        if let Some(&id) = self.token_to_id.get(token) {
            for (i, &val) in new_embedding.iter().enumerate() {
                self.embeddings[[id, i]] = val;
            }
        }
    }

    /// Actualiza el embedding por índice
    pub fn update_embedding_by_id(&mut self, id: usize, new_embedding: &Array1<f32>) {
        for (i, &val) in new_embedding.iter().enumerate() {
            self.embeddings[[id, i]] = val;
        }
    }

    /// Obtiene el tamaño del vocabulario
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Normaliza todos los embeddings a longitud unitaria
    /// Útil para mejorar la calidad de la similitud coseno
    pub fn normalize_embeddings(&mut self) {
        for i in 0..self.vocab_size() {
            let mut row = self.embeddings.row_mut(i);
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                row /= norm;
            }
        }
    }

    /// Guarda los embeddings en formato JSON
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Carga embeddings desde un archivo JSON
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let layer: EmbeddingLayer = serde_json::from_str(&json)?;
        Ok(layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_layer_creation() {
        let vocab = vec!["apple".to_string(), "banana".to_string(), "orange".to_string()];
        let layer = EmbeddingLayer::new(vocab.clone(), 50);

        assert_eq!(layer.vocab_size(), 3);
        assert_eq!(layer.embedding_dim, 50);
        assert_eq!(layer.embeddings.shape(), &[3, 50]);
    }

    #[test]
    fn test_get_embedding() {
        let vocab = vec!["apple".to_string(), "banana".to_string()];
        let layer = EmbeddingLayer::new(vocab, 10);

        let apple_emb = layer.get_embedding("apple");
        assert!(apple_emb.is_some());
        assert_eq!(apple_emb.unwrap().len(), 10);

        let none_emb = layer.get_embedding("nonexistent");
        assert!(none_emb.is_none());
    }

    #[test]
    fn test_normalization() {
        let vocab = vec!["test".to_string()];
        let mut layer = EmbeddingLayer::new(vocab, 3);

        // Establecer valores conocidos
        layer.embeddings[[0, 0]] = 3.0;
        layer.embeddings[[0, 1]] = 4.0;
        layer.embeddings[[0, 2]] = 0.0;

        layer.normalize_embeddings();

        let norm = layer.embeddings.row(0).iter()
            .map(|x| x * x).sum::<f32>().sqrt();

        assert!((norm - 1.0).abs() < 1e-6);
    }
}
