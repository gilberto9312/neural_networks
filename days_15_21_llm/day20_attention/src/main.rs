// Mecanismo de Atención
// Día 20: Mecanismo de Atención - Componente fundamental de Transformers

mod attention;
mod multihead;
mod positional;
mod visualize;
mod order_invariance;
mod params;

use ndarray::Array2;
use attention::{scaled_dot_product_attention, create_causal_mask, Attention};
use multihead::MultiHeadAttention;
use positional::PositionalEncoding;
use visualize::{
    plot_attention_heatmap, plot_multihead_attention, plot_positional_encoding,
    plot_attention_with_tokens,
};

fn main() {
    println!("⚡ Attention - Día 20: Mecanismo de Atención");
    println!("==============================================\n");

    // Ejemplo 1: Scaled Dot-Product Attention básico
    println!("📊 Ejemplo 1: Scaled Dot-Product Attention");
    println!("-------------------------------------------");
    ejemplo_scaled_attention();

    println!("\n");

    // Ejemplo 2: Atención con máscara causal
    println!("🎭 Ejemplo 2: Atención con Máscara Causal");
    println!("------------------------------------------");
    ejemplo_causal_mask();

    println!("\n");

    // Ejemplo 3: Multi-Head Attention
    println!("🔀 Ejemplo 3: Multi-Head Attention");
    println!("-----------------------------------");
    ejemplo_multihead();

    println!("\n");

    // Ejemplo 4: Positional Encoding
    println!("📍 Ejemplo 4: Positional Encoding");
    println!("----------------------------------");
    ejemplo_positional_encoding();

    println!("\n");

    // Ejemplo 5: Pipeline completo
    println!("🚀 Ejemplo 5: Pipeline Completo (Embeddings + PE + Attention)");
    println!("--------------------------------------------------------------");
    ejemplo_pipeline_completo();

    println!("\n");

    // Ejemplo 6: Invariancia de Orden
    println!("🔄 Ejemplo 6: Demostración de Invariancia de Orden");
    println!("---------------------------------------------------");
    order_invariance::demostrar_invariancia_orden();

    println!("\n");

    // Ejemplo 7: Conteo de Parámetros
    println!("📊 Ejemplo 7: Análisis de Parámetros Entrenables");
    println!("-------------------------------------------------");
    params::generar_reporte(256, 8, 512, false);
    params::comparar_configuraciones();

    println!("\n==============================================");
    println!("✨ Todos los ejemplos completados exitosamente!");
    println!("\nArchivos generados:");
    println!("  - attention_basic.png");
    println!("  - attention_causal.png");
    println!("  - multihead_attention.png");
    println!("  - positional_encoding.png");
    println!("  - attention_with_tokens.png");
    println!("\n💡 Conceptos demostrados:");
    println!("  1. Scaled Dot-Product Attention");
    println!("  2. Máscaras Causales (para decoders)");
    println!("  3. Multi-Head Attention (8 cabezas)");
    println!("  4. Positional Encoding (sinusoidal)");
    println!("  5. Pipeline completo de atención");
    println!("  6. Invariancia de orden sin PE");
    println!("  7. Análisis de parámetros entrenables");
}

/// Ejemplo 1: Scaled Dot-Product Attention básico
fn ejemplo_scaled_attention() {
    // Crear queries, keys y values simples
    let q = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 0.0, 0.0, 0.0, // Query 1: busca primera dimensión
            0.0, 1.0, 0.0, 0.0, // Query 2: busca segunda dimensión
            0.0, 0.0, 1.0, 0.0, // Query 3: busca tercera dimensión
        ],
    )
    .unwrap();

    let k = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 0.0, 0.0, 0.0, // Key 1
            0.0, 1.0, 0.0, 0.0, // Key 2
            0.0, 0.0, 1.0, 0.0, // Key 3
        ],
    )
    .unwrap();

    let v = Array2::from_shape_vec(
        (3, 2),
        vec![
            1.0, 2.0, // Value 1
            3.0, 4.0, // Value 2
            5.0, 6.0, // Value 3
        ],
    )
    .unwrap();

    println!("Dimensiones:");
    println!("  Q: {:?}", q.shape());
    println!("  K: {:?}", k.shape());
    println!("  V: {:?}", v.shape());

    // Aplicar atención
    let (output, weights) = scaled_dot_product_attention(&q, &k, &v, None);

    println!("\nPesos de atención:");
    for i in 0..weights.shape()[0] {
        print!("  Query {}: [", i);
        for j in 0..weights.shape()[1] {
            print!("{:.3}", weights[[i, j]]);
            if j < weights.shape()[1] - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    println!("\nSalida de atención:");
    println!("{:?}", output);

    // Visualizar
    if let Err(e) = plot_attention_heatmap(&weights, "attention_basic.png", "Scaled Dot-Product Attention", None) {
        eprintln!("Error generando visualización: {}", e);
    } else {
        println!("\n✅ Visualización guardada en: attention_basic.png");
    }
}

/// Ejemplo 2: Atención con máscara causal
fn ejemplo_causal_mask() {
    // Crear secuencia simple
    let seq_len = 5;
    let d_model = 8;

    // Crear queries, keys, values (simulando una secuencia)
    let mut q = Array2::zeros((seq_len, d_model));
    let mut k = Array2::zeros((seq_len, d_model));
    let mut v = Array2::zeros((seq_len, d_model));

    for i in 0..seq_len {
        for j in 0..d_model {
            q[[i, j]] = (i as f32 + j as f32) * 0.1;
            k[[i, j]] = (i as f32 + j as f32) * 0.1;
            v[[i, j]] = (i + 1) as f32;
        }
    }

    // Crear máscara causal
    let mask = create_causal_mask(seq_len);

    println!("Máscara causal (5x5):");
    println!("(0 = permitido, -inf = bloqueado)");
    for i in 0..seq_len {
        print!("  ");
        for j in 0..seq_len {
            if mask[[i, j]].is_infinite() {
                print!("  X ");
            } else {
                print!("  ✓ ");
            }
        }
        println!();
    }

    // Aplicar atención con máscara
    let (output, weights) = scaled_dot_product_attention(&q, &k, &v, Some(&mask));

    println!("\nPesos de atención con máscara causal:");
    for i in 0..weights.shape()[0] {
        print!("  Pos {}: [", i);
        for j in 0..weights.shape()[1] {
            print!("{:.3}", weights[[i, j]]);
            if j < weights.shape()[1] - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    // Visualizar
    if let Err(e) = plot_attention_heatmap(&weights, "attention_causal.png", "Atención con Máscara Causal", None) {
        eprintln!("Error generando visualización: {}", e);
    } else {
        println!("\n✅ Visualización guardada en: attention_causal.png");
    }
}

/// Ejemplo 3: Multi-Head Attention
fn ejemplo_multihead() {
    let d_model = 64;
    let num_heads = 8;
    let seq_len = 10;

    println!("Configuración:");
    println!("  d_model: {}", d_model);
    println!("  num_heads: {}", num_heads);
    println!("  d_k por cabeza: {}", d_model / num_heads);
    println!("  seq_len: {}", seq_len);

    // Crear Multi-Head Attention
    let mha = MultiHeadAttention::new(d_model, num_heads);
    println!("\n{}", mha.info());

    // Crear entrada de ejemplo
    let mut input = Array2::zeros((seq_len, d_model));
    for i in 0..seq_len {
        for j in 0..d_model {
            input[[i, j]] = ((i * d_model + j) as f32 * 0.01).sin();
        }
    }

    // Forward pass
    let (output, attention_weights) = mha.forward(&input, None);

    println!("\nDimensiones de salida: {:?}", output.shape());
    println!("Número de cabezas: {}", attention_weights.len());

    // Mostrar estadísticas de cada cabeza
    println!("\nEstadísticas por cabeza:");
    for (i, weights) in attention_weights.iter().enumerate() {
        let mean: f32 = weights.mean().unwrap();
        let max: f32 = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min: f32 = weights.iter().cloned().fold(f32::INFINITY, f32::min);

        println!(
            "  Cabeza {}: mean={:.4}, min={:.4}, max={:.4}",
            i, mean, min, max
        );
    }

    // Visualizar todas las cabezas
    if let Err(e) = plot_multihead_attention(&attention_weights, "multihead_attention.png", "Multi-Head Attention (8 cabezas)") {
        eprintln!("Error generando visualización: {}", e);
    } else {
        println!("\n✅ Visualización guardada en: multihead_attention.png");
    }
}

/// Ejemplo 4: Positional Encoding
fn ejemplo_positional_encoding() {
    let max_len = 50;
    let d_model = 128;

    println!("Configuración:");
    println!("  max_len: {}", max_len);
    println!("  d_model: {}", d_model);

    // Crear positional encoding sinusoidal
    let pe = PositionalEncoding::new_sinusoidal(max_len, d_model);
    println!("\n{}", pe.info());

    // Mostrar algunos valores de ejemplo
    println!("\nPrimeros valores de PE para posición 0:");
    let pos_0 = pe.get_position(0);
    print!("  [");
    for i in 0..8 {
        print!("{:.3}", pos_0[i]);
        if i < 7 {
            print!(", ");
        }
    }
    println!(", ...]");

    println!("\nPrimeros valores de PE para posición 10:");
    let pos_10 = pe.get_position(10);
    print!("  [");
    for i in 0..8 {
        print!("{:.3}", pos_10[i]);
        if i < 7 {
            print!(", ");
        }
    }
    println!(", ...]");

    // Aplicar a embeddings de ejemplo
    let embeddings = Array2::ones((10, d_model));
    let embeddings_with_pos = pe.apply(&embeddings);

    println!("\nDimensiones después de añadir PE: {:?}", embeddings_with_pos.shape());

    // Visualizar
    if let Err(e) = plot_positional_encoding(pe.encoding(), "positional_encoding.png", "Positional Encoding Sinusoidal") {
        eprintln!("Error generando visualización: {}", e);
    } else {
        println!("\n✅ Visualización guardada en: positional_encoding.png");
    }
}

/// Ejemplo 5: Pipeline completo (Embeddings + PE + Attention)
fn ejemplo_pipeline_completo() {
    let seq_len = 6;
    let d_model = 32;

    println!("Simulando: 'El gato come pescado ahora'");
    println!("Configuración:");
    println!("  seq_len: {} tokens", seq_len);
    println!("  d_model: {}", d_model);

    // 1. Embeddings simulados (normalmente vendrían de una capa de embedding)
    let mut embeddings = Array2::zeros((seq_len, d_model));
    for i in 0..seq_len {
        for j in 0..d_model {
            // Cada token tiene un patrón diferente
            embeddings[[i, j]] = ((i * 10 + j) as f32 * 0.05).sin();
        }
    }
    println!("\n1. Embeddings creados: {:?}", embeddings.shape());

    // 2. Añadir positional encoding
    let pe = PositionalEncoding::new_sinusoidal(100, d_model);
    let embeddings_with_pos = pe.apply(&embeddings);
    println!("2. Positional encoding añadido: {:?}", embeddings_with_pos.shape());

    // 3. Aplicar Self-Attention
    let attention = Attention::new(d_model, d_model, d_model);
    let (output, weights) = attention.forward(&embeddings_with_pos, None);
    println!("3. Self-Attention aplicado: {:?}", output.shape());

    // Analizar pesos de atención
    println!("\n📊 Análisis de pesos de atención:");
    println!("(Cada fila muestra qué tan fuerte cada token atiende a otros tokens)");

    let tokens = ["El", "gato", "come", "pescado", "ahora", "EOS"];

    println!("\nMatriz de atención:");
    print!("       ");
    for token in &tokens {
        print!("{:>8}", token);
    }
    println!();

    for i in 0..seq_len {
        print!("{:>6} ", tokens[i]);
        for j in 0..seq_len {
            print!(" {:.4} ", weights[[i, j]]);
        }
        println!();
    }

    // Visualizar con tokens
    if let Err(e) = plot_attention_with_tokens(
        &weights,
        &tokens,
        "attention_with_tokens.png",
        "Self-Attention: 'El gato come pescado ahora'"
    ) {
        eprintln!("Error generando visualización: {}", e);
    } else {
        println!("\n✅ Visualización guardada en: attention_with_tokens.png");
    }

    // Interpretación
    println!("\n💡 Interpretación:");
    println!("  - Valores altos en la diagonal: cada token se atiende a sí mismo");
    println!("  - Valores altos fuera de diagonal: relaciones entre tokens");
    println!("  - En transformers reales, esto captura dependencias sintácticas y semánticas");
}
