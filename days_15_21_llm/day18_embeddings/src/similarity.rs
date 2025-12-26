// Cálculo de similitud coseno y operaciones vectoriales
// Basado en el Lab 2.5 de Google DeepMind

use ndarray::Array1;
use crate::embedding_layer::EmbeddingLayer;

/// Calcula la similitud coseno entre dos vectores
///
/// La similitud coseno se define como:
/// cos(u, v) = (u · v) / (||u|| × ||v||)
///
/// Donde:
/// - u · v es el producto punto
/// - ||u|| y ||v|| son las magnitudes (normas L2) de los vectores
///
/// # Retorna
/// Un valor entre -1 y 1:
/// - 1: Vectores idénticos (mismo dirección)
/// - 0: Vectores ortogonales (no relacionados)
/// - -1: Vectores opuestos (significados contrarios)
pub fn cosine_similarity(u: &Array1<f32>, v: &Array1<f32>) -> f32 {
    // Producto punto: u · v
    let dot_product: f32 = u.iter()
        .zip(v.iter())
        .map(|(a, b)| a * b)
        .sum();

    // Normas L2 (magnitudes)
    let norm_u: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Evitar división por cero
    if norm_u == 0.0 || norm_v == 0.0 {
        return 0.0;
    }

    // Similitud coseno
    dot_product / (norm_u * norm_v)
}

/// Encuentra los k tokens más similares a un token dado
///
/// # Argumentos
/// * `embeddings` - Capa de embeddings
/// * `token` - Token de referencia
/// * `k` - Número de vecinos cercanos a retornar
///
/// # Retorna
/// Vector de tuplas (token, similitud) ordenadas de mayor a menor similitud
pub fn find_similar_tokens(
    embeddings: &EmbeddingLayer,
    token: &str,
    k: usize
) -> Vec<(String, f32)> {
    // Obtener el embedding del token de referencia
    let target_embedding = match embeddings.get_embedding(token) {
        Some(emb) => emb,
        None => {
            println!("⚠️  Token '{}' no encontrado en el vocabulario", token);
            return Vec::new();
        }
    };

    // Calcular similitudes con todos los tokens
    let mut similarities: Vec<(String, f32)> = Vec::new();

    for i in 0..embeddings.vocab_size() {
        let current_token = embeddings.get_token(i).unwrap();

        // Saltar el token mismo
        if current_token == token {
            continue;
        }

        let current_embedding = embeddings.get_embedding_by_id(i);
        let similarity = cosine_similarity(&target_embedding, &current_embedding);

        similarities.push((current_token.to_string(), similarity));
    }

    // Ordenar por similitud descendente
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Retornar top k
    similarities.into_iter().take(k).collect()
}

/// Imprime la similitud entre dos tokens
pub fn print_similarity(
    embeddings: &EmbeddingLayer,
    token1: &str,
    token2: &str
) -> Option<f32> {
    let emb1 = embeddings.get_embedding(token1)?;
    let emb2 = embeddings.get_embedding(token2)?;

    let similarity = cosine_similarity(&emb1, &emb2);

    println!(
        "Similitud coseno entre \"{}\" y \"{}\" = {:.4}",
        token1, token2, similarity
    );

    Some(similarity)
}

/// Realiza operaciones de analogía: word1 - word2 + word3 ≈ ?
///
/// Ejemplo clásico: "rey" - "hombre" + "mujer" ≈ "reina"
///
/// # Argumentos
/// * `embeddings` - Capa de embeddings
/// * `word1` - Primera palabra (ej: "rey")
/// * `word2` - Palabra a restar (ej: "hombre")
/// * `word3` - Palabra a sumar (ej: "mujer")
/// * `k` - Número de resultados a retornar
///
/// # Retorna
/// Vector de tuplas (token, similitud) con los k tokens más cercanos
pub fn word_analogy(
    embeddings: &EmbeddingLayer,
    word1: &str,
    word2: &str,
    word3: &str,
    k: usize
) -> Vec<(String, f32)> {
    // Obtener embeddings
    let emb1 = match embeddings.get_embedding(word1) {
        Some(e) => e,
        None => {
            println!("⚠️  Token '{}' no encontrado", word1);
            return Vec::new();
        }
    };

    let emb2 = match embeddings.get_embedding(word2) {
        Some(e) => e,
        None => {
            println!("⚠️  Token '{}' no encontrado", word2);
            return Vec::new();
        }
    };

    let emb3 = match embeddings.get_embedding(word3) {
        Some(e) => e,
        None => {
            println!("⚠️  Token '{}' no encontrado", word3);
            return Vec::new();
        }
    };

    // Calcular: word1 - word2 + word3
    let result_embedding = &emb1 - &emb2 + &emb3;

    // Encontrar tokens más cercanos al resultado
    let mut similarities: Vec<(String, f32)> = Vec::new();

    for i in 0..embeddings.vocab_size() {
        let current_token = embeddings.get_token(i).unwrap();

        // Saltar las palabras de entrada
        if current_token == word1 || current_token == word2 || current_token == word3 {
            continue;
        }

        let current_embedding = embeddings.get_embedding_by_id(i);
        let similarity = cosine_similarity(&result_embedding, &current_embedding);

        similarities.push((current_token.to_string(), similarity));
    }

    // Ordenar por similitud descendente
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Retornar top k
    similarities.into_iter().take(k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cosine_similarity_identical() {
        let v1 = array![1.0, 2.0, 3.0];
        let v2 = array![1.0, 2.0, 3.0];

        let sim = cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = array![1.0, 0.0, 0.0];
        let v2 = array![0.0, 1.0, 0.0];

        let sim = cosine_similarity(&v1, &v2);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let v1 = array![1.0, 2.0, 3.0];
        let v2 = array![-1.0, -2.0, -3.0];

        let sim = cosine_similarity(&v1, &v2);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_similar_tokens() {
        let vocab = vec![
            "apple".to_string(),
            "banana".to_string(),
            "car".to_string(),
            "orange".to_string(),
        ];

        let embeddings = EmbeddingLayer::new(vocab, 10);
        let similar = find_similar_tokens(&embeddings, "apple", 2);

        assert_eq!(similar.len(), 2);
    }
}
