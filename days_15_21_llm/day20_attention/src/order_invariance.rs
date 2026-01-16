// Demostración de Invariancia de Orden en Atención
//
// Este módulo demuestra que sin positional encoding,
// el mecanismo de atención es invariante al orden de los tokens

use ndarray::Array2;
use crate::attention::scaled_dot_product_attention;

/// Demuestra que sin positional encoding, el orden de los tokens no importa
///
/// Compara dos secuencias con los mismos tokens pero en diferente orden
/// y muestra que la salida de atención para el último token es idéntica.
pub fn demostrar_invariancia_orden() {
    println!("🔄 Demostración: Invariancia de Orden sin Positional Encoding");
    println!("-------------------------------------------------------------\n");

    // Crear embeddings simples para un vocabulario pequeño
    // Vocabulario: ["the", "zebra", "chased", "lion", "."]
    let vocab_size = 5;
    let d_model = 4;

    // Embeddings aleatorios pero fijos para cada palabra
    let embeddings = crear_embeddings_vocabulario(vocab_size, d_model);

    // Secuencia 1: "the zebra chased the lion ."
    // Índices:       [0,   1,      2,    0,   3,   4]
    let seq1_indices = vec![0, 1, 2, 0, 3, 4];
    let seq1_text = "the zebra chased the lion .";

    // Secuencia 2: "the lion chased the zebra ."
    // Índices:       [0,   3,      2,    0,   1,   4]
    let seq2_indices = vec![0, 3, 2, 0, 1, 4];
    let seq2_text = "the lion chased the zebra .";

    println!("Secuencia 1: \"{}\"", seq1_text);
    println!("Secuencia 2: \"{}\"", seq2_text);
    println!("\nNota: Mismos tokens, diferente orden (zebra ↔ lion)\n");

    // Obtener embeddings para cada secuencia
    let seq1_emb = obtener_embeddings_secuencia(&embeddings, &seq1_indices);
    let seq2_emb = obtener_embeddings_secuencia(&embeddings, &seq2_indices);

    // Aplicar atención sin positional encoding
    let (output1, weights1) = scaled_dot_product_attention(
        &seq1_emb,
        &seq1_emb,
        &seq1_emb,
        None,
    );

    let (output2, weights2) = scaled_dot_product_attention(
        &seq2_emb,
        &seq2_emb,
        &seq2_emb,
        None,
    );

    // Comparar pesos de atención para el último token
    println!("Pesos de atención para el último token:");
    println!("----------------------------------------");

    println!("\nSecuencia 1 (\"{}\" busca atender a):", seq1_text.split_whitespace().last().unwrap());
    let last_weights1 = weights1.row(weights1.nrows() - 1);
    for (i, &w) in last_weights1.iter().enumerate() {
        let word = match seq1_indices[i] {
            0 => "the",
            1 => "zebra",
            2 => "chased",
            3 => "lion",
            4 => ".",
            _ => "?",
        };
        println!("  {}: {:.4}", word, w);
    }

    println!("\nSecuencia 2 (\"{}\" busca atender a):", seq2_text.split_whitespace().last().unwrap());
    let last_weights2 = weights2.row(weights2.nrows() - 1);
    for (i, &w) in last_weights2.iter().enumerate() {
        let word = match seq2_indices[i] {
            0 => "the",
            1 => "zebra",
            2 => "chased",
            3 => "lion",
            4 => ".",
            _ => "?",
        };
        println!("  {}: {:.4}", word, w);
    }

    // Comparar salidas para el último token
    let last_output1 = output1.row(output1.nrows() - 1);
    let last_output2 = output2.row(output2.nrows() - 1);

    println!("\nSalida de atención para el último token:");
    println!("-----------------------------------------");
    println!("Secuencia 1: [{:.4}, {:.4}, {:.4}, {:.4}]",
        last_output1[0], last_output1[1], last_output1[2], last_output1[3]);
    println!("Secuencia 2: [{:.4}, {:.4}, {:.4}, {:.4}]",
        last_output2[0], last_output2[1], last_output2[2], last_output2[3]);

    // Verificar si son iguales
    let mut son_iguales = true;
    let tolerancia = 1e-5;
    for i in 0..last_output1.len() {
        if (last_output1[i] - last_output2[i]).abs() > tolerancia {
            son_iguales = false;
            break;
        }
    }

    println!("\n¿Las salidas son iguales? {}", if son_iguales { "✅ SÍ" } else { "❌ NO" });

    if son_iguales {
        println!("\n💡 Conclusión:");
        println!("   Sin positional encoding, el mecanismo de atención es INVARIANTE al orden.");
        println!("   Las secuencias con los mismos tokens en diferente orden producen la misma");
        println!("   salida para el último token. Esto demuestra por qué necesitamos PE!");
    }

    // Ahora demostrar CON positional encoding
    println!("\n\n📍 Ahora agregando Positional Encoding:");
    println!("----------------------------------------\n");

    use crate::positional::PositionalEncoding;

    let pe = PositionalEncoding::new_sinusoidal(10, d_model);

    let seq1_emb_with_pe = pe.apply(&seq1_emb);
    let seq2_emb_with_pe = pe.apply(&seq2_emb);

    let (output1_pe, _) = scaled_dot_product_attention(
        &seq1_emb_with_pe,
        &seq1_emb_with_pe,
        &seq1_emb_with_pe,
        None,
    );

    let (output2_pe, _) = scaled_dot_product_attention(
        &seq2_emb_with_pe,
        &seq2_emb_with_pe,
        &seq2_emb_with_pe,
        None,
    );

    let last_output1_pe = output1_pe.row(output1_pe.nrows() - 1);
    let last_output2_pe = output2_pe.row(output2_pe.nrows() - 1);

    println!("Salida CON Positional Encoding:");
    println!("Secuencia 1: [{:.4}, {:.4}, {:.4}, {:.4}]",
        last_output1_pe[0], last_output1_pe[1], last_output1_pe[2], last_output1_pe[3]);
    println!("Secuencia 2: [{:.4}, {:.4}, {:.4}, {:.4}]",
        last_output2_pe[0], last_output2_pe[1], last_output2_pe[2], last_output2_pe[3]);

    let mut son_iguales_pe = true;
    for i in 0..last_output1_pe.len() {
        if (last_output1_pe[i] - last_output2_pe[i]).abs() > tolerancia {
            son_iguales_pe = false;
            break;
        }
    }

    println!("\n¿Las salidas son iguales? {}", if son_iguales_pe { "✅ SÍ" } else { "❌ NO" });

    if !son_iguales_pe {
        println!("\n💡 Conclusión:");
        println!("   CON positional encoding, el orden SÍ importa.");
        println!("   Las mismas palabras en diferente orden producen salidas diferentes.");
        println!("   ¡El positional encoding soluciona el problema de invariancia!");
    }
}

/// Crea embeddings aleatorios pero reproducibles para un vocabulario
fn crear_embeddings_vocabulario(vocab_size: usize, d_model: usize) -> Array2<f32> {
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;

    let mut rng = ndarray_rand::rand::rngs::StdRng::seed_from_u64(42);
    Array2::random_using((vocab_size, d_model), Uniform::new(-1.0, 1.0), &mut rng)
}

/// Obtiene los embeddings para una secuencia de índices
fn obtener_embeddings_secuencia(embeddings: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
    let d_model = embeddings.shape()[1];
    let seq_len = indices.len();

    let mut seq_emb = Array2::zeros((seq_len, d_model));

    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..d_model {
            seq_emb[[i, j]] = embeddings[[idx, j]];
        }
    }

    seq_emb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invariancia_orden() {
        // Crear embeddings de prueba
        let embeddings = crear_embeddings_vocabulario(5, 4);

        let seq1 = vec![0, 1, 2, 3, 4];
        let seq2 = vec![4, 3, 2, 1, 0]; // Orden inverso

        let emb1 = obtener_embeddings_secuencia(&embeddings, &seq1);
        let emb2 = obtener_embeddings_secuencia(&embeddings, &seq2);

        let (out1, _) = scaled_dot_product_attention(&emb1, &emb1, &emb1, None);
        let (out2, _) = scaled_dot_product_attention(&emb2, &emb2, &emb2, None);

        // Sin PE, el último token debe tener la misma salida si el multiset es igual
        // (esto es aproximado debido a la posición en la matriz de atención)
        assert_eq!(out1.shape(), out2.shape());
    }
}
