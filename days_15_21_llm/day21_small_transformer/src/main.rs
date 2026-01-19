// Small Transformer LLM
// Día 21: Transformer Completo - Proyecto Final del Plan de 21 Días

mod attention;
mod positional;
mod feedforward;
mod layer_norm;
mod utils;
mod embedding;
mod encoder;
mod decoder;
mod transformer;
mod tokenizer;
mod dataset;
mod training;
mod generation;
mod checkpoint;

use std::env;
use transformer::TransformerLM;
use tokenizer::SimpleTokenizer;
use dataset::{TextDataset, AFRICA_GALORE_SAMPLE};
use checkpoint::ModelConfig;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "demo" => demo(),
        "train" => train_model(),
        "generate" => {
            if args.len() < 3 {
                eprintln!("❌ Uso: cargo run --release -- generate \"<prompt>\"");
                return;
            }
            let prompt = &args[2];
            generate_text(prompt);
        }
        "info" => show_info(),
        _ => {
            eprintln!("❌ Comando desconocido: {}", args[1]);
            print_usage();
        }
    }
}

fn print_usage() {
    println!("🚀 Small Transformer LLM - Día 21");
    println!("═══════════════════════════════════════════════════════════");
    println!("\nProyecto final del plan de 21 días de aprendizaje de LLMs\n");
    println!("COMANDOS:");
    println!("  demo                     Demostración completa");
    println!("  train                    Entrenar modelo");
    println!("  generate \"<prompt>\"      Generar texto desde prompt");
    println!("  info                     Información del modelo\n");
    println!("EJEMPLOS:");
    println!("  cargo run --release -- demo");
    println!("  cargo run --release -- generate \"Abeni was\"");
    println!("  cargo run --release -- train");
}

fn demo() {
    println!("\n🎯 DEMOSTRACIÓN COMPLETA - Small Transformer LLM");
    println!("═══════════════════════════════════════════════════════════\n");

    // 1. Preparar datos
    println!("📚 Paso 1/5: Preparando datos...");
    println!("─────────────────────────────────────────────────────────");

    let corpus = vec![AFRICA_GALORE_SAMPLE.to_string()];
    let tokenizer = SimpleTokenizer::from_corpus(&corpus, 2);

    println!("✓ Vocabulario construido: {} tokens", tokenizer.vocab_size());
    println!("✓ Tokens especiales: PAD={}, UNK={}, BOS={}, EOS={}",
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id
    );

    // 2. Crear modelo
    println!("\n🏗️  Paso 2/5: Creando modelo transformer...");
    println!("─────────────────────────────────────────────────────────");

    let model = TransformerLM::new(
        tokenizer.vocab_size(),
        128,  // d_model
        4,    // num_heads
        2,    // num_layers
        64,   // max_seq_len
    );

    println!("{}", model.info());

    // 3. Preparar dataset
    println!("\n📊 Paso 3/5: Preparando dataset...");
    println!("─────────────────────────────────────────────────────────");

    let dataset = TextDataset::from_text(AFRICA_GALORE_SAMPLE, &tokenizer, 32);
    println!("✓ Dataset creado: {} secuencias de longitud 32", dataset.len());

    // 4. Entrenamiento (simulado)
    println!("\n🎓 Paso 4/5: Demostración de entrenamiento...");
    println!("─────────────────────────────────────────────────────────");

    training::train(&model, &dataset, 20, 4);

    // 5. Generación
    println!("\n✨ Paso 5/5: Generación de texto...");
    println!("─────────────────────────────────────────────────────────");

    let prompts = vec![
        "Abeni was",
        "The marketplace",
        "Children sang",
    ];

    for prompt in prompts {
        println!("\n📝 Prompt: \"{}\"", prompt);
        let generated = generation::generate(&model, &tokenizer, prompt, 15, 1.0);
        println!("   → {}", generated);
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("✅ DEMOSTRACIÓN COMPLETADA");
    println!("\n💡 CONCEPTOS DEMOSTRADOS:");
    println!("   1. Tokenización con vocabulario basado en palabras");
    println!("   2. Arquitectura Transformer (decoder-only)");
    println!("   3. Multi-Head Attention con máscara causal");
    println!("   4. Positional Encoding sinusoidal");
    println!("   5. Feed-Forward Networks y Layer Normalization");
    println!("   6. Generación autoregressiva de texto");
    println!("\n⚠️  NOTA: Este es un modelo educativo simplificado.");
    println!("   Para producción se requeriría entrenamiento con backprop completo.");
    println!("═══════════════════════════════════════════════════════════\n");
}

fn train_model() {
    println!("\n🎓 MODO ENTRENAMIENTO");
    println!("═══════════════════════════════════════════════════════════\n");

    let corpus = vec![AFRICA_GALORE_SAMPLE.to_string()];
    let tokenizer = SimpleTokenizer::from_corpus(&corpus, 2);
    let model = TransformerLM::new(tokenizer.vocab_size(), 128, 4, 2, 64);
    let dataset = TextDataset::from_text(AFRICA_GALORE_SAMPLE, &tokenizer, 32);

    training::train(&model, &dataset, 50, 8);

    println!("\n💾 Guardando configuración...");
    let config = ModelConfig {
        vocab_size: model.vocab_size,
        d_model: model.d_model,
        num_heads: model.num_heads,
        num_layers: model.num_layers,
        max_seq_len: model.max_seq_len,
    };

    if let Err(e) = checkpoint::save_config(&config, "model_config.json") {
        eprintln!("⚠️  Error al guardar configuración: {}", e);
    } else {
        println!("✓ Configuración guardada en model_config.json");
    }

    if let Err(e) = checkpoint::save_tokenizer(&tokenizer, "tokenizer.json") {
        eprintln!("⚠️  Error al guardar tokenizer: {}", e);
    } else {
        println!("✓ Tokenizer guardado en tokenizer.json");
    }
}

fn generate_text(prompt: &str) {
    println!("\n✨ MODO GENERACIÓN");
    println!("═══════════════════════════════════════════════════════════");

    let corpus = vec![AFRICA_GALORE_SAMPLE.to_string()];
    let tokenizer = SimpleTokenizer::from_corpus(&corpus, 2);
    let model = TransformerLM::new(tokenizer.vocab_size(), 128, 4, 2, 64);

    println!("\nPrompt: \"{}\"", prompt);
    println!("Generando 20 tokens...\n");

    let generated = generation::generate(&model, &tokenizer, prompt, 20, 1.0);

    println!("\n─────────────────────────────────────────────────────────");
    println!("RESULTADO:");
    println!("{}", generated);
    println!("─────────────────────────────────────────────────────────\n");
}

fn show_info() {
    println!("\n📊 INFORMACIÓN DEL MODELO");
    println!("═══════════════════════════════════════════════════════════\n");

    let corpus = vec![AFRICA_GALORE_SAMPLE.to_string()];
    let tokenizer = SimpleTokenizer::from_corpus(&corpus, 2);
    let model = TransformerLM::new(tokenizer.vocab_size(), 128, 4, 2, 64);

    println!("{}\n", model.info());

    println!("ARQUITECTURA:");
    println!("  • Decoder-only (GPT-style)");
    println!("  • {} capas decoder", model.num_layers);
    println!("  • {} cabezas de atención", model.num_heads);
    println!("  • Dimensión por cabeza: {}", model.d_model / model.num_heads);
    println!("  • Dimensión feed-forward: {}", model.d_ff);
    println!("\nCOMPONENTES:");
    println!("  ✓ Embedding Layer");
    println!("  ✓ Positional Encoding (sinusoidal)");
    println!("  ✓ Multi-Head Attention con máscara causal");
    println!("  ✓ Feed-Forward Networks (ReLU)");
    println!("  ✓ Layer Normalization");
    println!("  ✓ Residual Connections");
    println!("\n═══════════════════════════════════════════════════════════\n");
}
