// Word Embeddings y Similitud - Día 18
// Implementación de embeddings con Skip-gram simplificado y operaciones vectoriales

mod embedding_layer;
mod similarity;
mod visualize;

use embedding_layer::EmbeddingLayer;
use similarity::{find_similar_tokens, print_similarity, word_analogy};
use std::env;
use std::collections::HashMap;
use rand::Rng;

/// Entrena embeddings usando Skip-gram simplificado
///
/// Skip-gram predice palabras del contexto dada una palabra central
/// Simplificación: usamos ventana de contexto pequeña y actualización directa
fn train_skipgram(text: &str, embedding_dim: usize, epochs: usize, window_size: usize) -> EmbeddingLayer {
    println!("📚 Entrenando embeddings con Skip-gram...");
    println!("   Dimensión: {}", embedding_dim);
    println!("   Épocas: {}", epochs);
    println!("   Ventana de contexto: {}", window_size);

    // Tokenizar texto (simplificado: por espacios)
    let tokens: Vec<String> = text
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    println!("   Total de tokens: {}", tokens.len());

    // Construir vocabulario
    let mut vocab_counts: HashMap<String, usize> = HashMap::new();
    for token in &tokens {
        *vocab_counts.entry(token.clone()).or_insert(0) += 1;
    }

    // Filtrar tokens poco frecuentes (mínimo 2 ocurrencias)
    let vocab: Vec<String> = vocab_counts
        .iter()
        .filter(|(_, &count)| count >= 2)
        .map(|(token, _)| token.clone())
        .collect();

    println!("   Tamaño del vocabulario: {}", vocab.len());

    // Inicializar embeddings
    let mut embeddings = EmbeddingLayer::new(vocab.clone(), embedding_dim);

    // Crear índices de tokens en el texto
    let token_ids: Vec<usize> = tokens
        .iter()
        .filter_map(|t| embeddings.get_id(t))
        .collect();

    // Learning rate con decaimiento
    let initial_lr = 0.01; // Reducido para estabilidad
    let negative_samples = 3; // Reducido de 5 a 3
    let mut rng = rand::thread_rng();

    println!("   Negative sampling: {} muestras por contexto", negative_samples);

    // Entrenamiento con Negative Sampling (versión estable)
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut samples = 0;

        // Decaer learning rate
        let learning_rate = initial_lr * (1.0 - epoch as f32 / epochs as f32).max(0.0001);

        // Para cada posición en el texto
        for (i, &center_id) in token_ids.iter().enumerate() {
            // Obtener contexto (palabras cercanas) - MUESTRAS POSITIVAS
            let start = if i >= window_size { i - window_size } else { 0 };
            let end = (i + window_size + 1).min(token_ids.len());

            // Recolectar IDs del contexto actual
            let mut context_ids = Vec::new();
            for j in start..end {
                if i != j {
                    context_ids.push(token_ids[j]);
                }
            }

            // 1. MUESTRAS POSITIVAS: acercar embeddings del contexto
            for &context_id in &context_ids {
                let mut center_emb = embeddings.get_embedding_by_id(center_id);
                let context_emb = embeddings.get_embedding_by_id(context_id);

                // Calcular dot product (similitud)
                let dot: f32 = center_emb.iter().zip(context_emb.iter()).map(|(a,b)| a*b).sum();

                // Acercar con factor que decrece si ya están cerca
                let positive_factor = (1.0 / (1.0 + dot.exp())).min(1.0);

                let diff = &context_emb - &center_emb;
                center_emb = &center_emb + &(&diff * (learning_rate * positive_factor));

                // Clip para evitar explosión
                for val in center_emb.iter_mut() {
                    *val = val.clamp(-10.0, 10.0);
                }

                embeddings.update_embedding_by_id(center_id, &center_emb);

                let loss = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
                total_loss += loss;
                samples += 1;
            }

            // 2. NEGATIVE SAMPLING: alejar embeddings de palabras NO relacionadas
            for _ in 0..negative_samples {
                // Muestrear palabra aleatoria del vocabulario
                let neg_id = rng.gen_range(0..embeddings.vocab_size());

                // Asegurarse de que no es la palabra central ni está en el contexto
                if neg_id == center_id || context_ids.contains(&neg_id) {
                    continue;
                }

                let mut center_emb = embeddings.get_embedding_by_id(center_id);
                let neg_emb = embeddings.get_embedding_by_id(neg_id);

                // Calcular dot product
                let dot: f32 = center_emb.iter().zip(neg_emb.iter()).map(|(a,b)| a*b).sum();

                // Alejar solo si están demasiado cerca
                let negative_factor = (-dot / (1.0 + (-dot).exp())).max(0.0);

                let diff = &neg_emb - &center_emb;
                center_emb = &center_emb - &(&diff * (learning_rate * 0.1 * negative_factor));

                // Clip para evitar explosión
                for val in center_emb.iter_mut() {
                    *val = val.clamp(-10.0, 10.0);
                }

                embeddings.update_embedding_by_id(center_id, &center_emb);
            }
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            let avg_loss = if samples > 0 { total_loss / samples as f32 } else { 0.0 };
            println!("   Época {}/{}: Loss promedio = {:.4}", epoch + 1, epochs, avg_loss);
        }
    }

    // Normalizar embeddings al final para que similitud coseno funcione mejor
    embeddings.normalize_embeddings();

    println!("✅ Entrenamiento completado!");
    embeddings
}

/// Carga texto de ejemplo para entrenamiento
fn get_example_corpus() -> String {
    // Corpus estructurado con categorías semánticas separadas
    r#"
    The king rules the kingdom. The queen wears the crown. The prince is royal. The princess is royal.
    The king and queen live in palace. Royal family has power. The king commands the army.
    The queen governs with wisdom. Prince will become king. Princess will become queen.

    The cat meows loudly. The dog barks constantly. The cat chases mouse. The dog runs fast.
    Cats are pets. Dogs are pets. Cat sleeps on bed. Dog plays in yard.
    The kitten is small cat. The puppy is small dog. Animals need food.

    The apple is red fruit. The banana is yellow fruit. The orange is citrus fruit.
    Apple grows on tree. Banana grows in tropics. Orange has vitamin.
    Fruit tastes sweet. People eat fruit. Fresh fruit is healthy.
    Apple juice is delicious. Banana smoothie is good. Orange juice is fresh.

    The car drives on road. The bus transports people. The truck carries cargo.
    Car has four wheels. Bus is large vehicle. Truck is heavy vehicle.
    Vehicle needs fuel. Car runs on gas. Bus stops frequently.
    The highway has cars. The street has buses. Traffic moves slowly.

    Happy people smile often. Sad people cry sometimes. Joy brings happiness.
    Good mood feels nice. Bad mood feels terrible. Happiness is positive emotion.
    Sadness is negative emotion. Joy and happiness are similar. Sad and unhappy are similar.

    The computer runs programs. The phone makes calls. The internet connects devices.
    Computer has keyboard. Phone has screen. Internet uses network.
    Technology advances rapidly. Devices become smarter. Digital world grows.
    "#.to_string()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "train" => {
            // Entrenar embeddings
            let corpus = get_example_corpus();
            let embeddings = train_skipgram(&corpus, 50, 100, 2);

            // Guardar embeddings entrenados
            match embeddings.save("embeddings.json") {
                Ok(_) => println!("💾 Embeddings guardados en 'embeddings.json'"),
                Err(e) => eprintln!("❌ Error al guardar: {}", e),
            }

            // Mostrar algunos ejemplos
            println!("\n🔍 Ejemplos de similitud:");
            demo_similarities(&embeddings);
        }

        "similar" => {
            if args.len() < 3 {
                eprintln!("❌ Uso: cargo run -- similar <palabra>");
                return;
            }

            let word = &args[2];
            let k = if args.len() > 3 {
                args[3].parse().unwrap_or(5)
            } else {
                5
            };

            // Cargar embeddings
            let embeddings = match EmbeddingLayer::load("embeddings.json") {
                Ok(e) => e,
                Err(_) => {
                    println!("⚠️  No se encontraron embeddings entrenados. Entrenando...");
                    let corpus = get_example_corpus();
                    train_skipgram(&corpus, 50, 100, 2)
                }
            };

            println!("\n🔍 Palabras similares a '{}':\n", word);
            let similar = find_similar_tokens(&embeddings, word, k);

            for (i, (token, sim)) in similar.iter().enumerate() {
                println!("  {}. {} (similitud: {:.4})", i + 1, token, sim);
            }
        }

        "analogy" => {
            if args.len() < 5 {
                eprintln!("❌ Uso: cargo run -- analogy <palabra1> <palabra2> <palabra3>");
                eprintln!("   Ejemplo: cargo run -- analogy king man woman");
                eprintln!("   (Encuentra: king - man + woman ≈ ?)");
                return;
            }

            let word1 = &args[2];
            let word2 = &args[3];
            let word3 = &args[4];
            let k = 5;

            // Cargar embeddings
            let embeddings = match EmbeddingLayer::load("embeddings.json") {
                Ok(e) => e,
                Err(_) => {
                    println!("⚠️  No se encontraron embeddings entrenados. Entrenando...");
                    let corpus = get_example_corpus();
                    train_skipgram(&corpus, 50, 100, 2)
                }
            };

            println!("\n🧮 Analogía: '{}' - '{}' + '{}' ≈ ?\n", word1, word2, word3);
            let results = word_analogy(&embeddings, word1, word2, word3, k);

            for (i, (token, sim)) in results.iter().enumerate() {
                println!("  {}. {} (similitud: {:.4})", i + 1, token, sim);
            }
        }

        "demo" => {
            println!("🎯 Demostración de Embeddings - Día 18\n");

            // Entrenar o cargar embeddings
            let embeddings = match EmbeddingLayer::load("embeddings.json") {
                Ok(e) => {
                    println!("✅ Embeddings cargados desde 'embeddings.json'");
                    e
                }
                Err(_) => {
                    let corpus = get_example_corpus();
                    let e = train_skipgram(&corpus, 50, 100, 2);
                    let _ = e.save("embeddings.json");
                    e
                }
            };

            demo_similarities(&embeddings);
            demo_analogies(&embeddings);
        }

        _ => {
            eprintln!("❌ Comando desconocido: {}", args[1]);
            print_usage();
        }
    }
}

fn print_usage() {
    println!("🎯 Word Embeddings - Día 18");
    println!("\nUso:");
    println!("  cargo run -- train              Entrena embeddings con Skip-gram");
    println!("  cargo run -- similar <palabra>  Encuentra palabras similares");
    println!("  cargo run -- analogy <w1> <w2> <w3>  Operación de analogía (w1 - w2 + w3)");
    println!("  cargo run -- demo               Ejecuta demostración completa");
    println!("\nEjemplos:");
    println!("  cargo run -- train");
    println!("  cargo run -- similar king");
    println!("  cargo run -- analogy king man woman");
}

fn demo_similarities(embeddings: &EmbeddingLayer) {
    println!("\n📊 Similitudes entre pares de palabras:");
    println!("═════════════════════════════════════════════════════════");

    println!("\n✅ Palabras SIMILARES (similitud alta, cerca de 1.0):");
    println!("─────────────────────────────────────────────────────");
    let similar_pairs = vec![
        ("king", "queen"),        // Realeza
        ("prince", "princess"),   // Realeza joven
        ("cat", "dog"),          // Mascotas
        ("apple", "banana"),     // Frutas
        ("car", "bus"),          // Vehículos
        ("happy", "joy"),        // Emociones positivas
    ];

    for (w1, w2) in similar_pairs {
        if embeddings.get_embedding(w1).is_some() && embeddings.get_embedding(w2).is_some() {
            print_similarity(embeddings, w1, w2);
        }
    }

    println!("\n❌ Palabras DIFERENTES (similitud baja, cerca de 0.0):");
    println!("──────────────────────────────────────────────────────");
    let different_pairs = vec![
        ("king", "apple"),       // Realeza vs Fruta
        ("cat", "car"),          // Animal vs Vehículo
        ("happy", "computer"),   // Emoción vs Tecnología
        ("dog", "banana"),       // Animal vs Fruta
        ("queen", "bus"),        // Realeza vs Vehículo
        ("fruit", "vehicle"),    // Categorías opuestas
    ];

    for (w1, w2) in different_pairs {
        if embeddings.get_embedding(w1).is_some() && embeddings.get_embedding(w2).is_some() {
            print_similarity(embeddings, w1, w2);
        }
    }

    println!("\n🤔 Palabras con RELACIÓN PARCIAL (similitud media):");
    println!("────────────────────────────────────────────────────");
    let partial_pairs = vec![
        ("good", "bad"),         // Antónimos (relacionados conceptualmente)
        ("happy", "sad"),        // Antónimos
        ("king", "royal"),       // Relacionados semánticamente
    ];

    for (w1, w2) in partial_pairs {
        if embeddings.get_embedding(w1).is_some() && embeddings.get_embedding(w2).is_some() {
            print_similarity(embeddings, w1, w2);
        }
    }
}

fn demo_analogies(embeddings: &EmbeddingLayer) {
    println!("\n🧮 Ejemplos de Analogías:");
    println!("───────────────────────────");

    let analogies = vec![
        ("king", "prince", "princess", "¿Qué es a 'princess' como 'king' es a 'prince'?"),
        ("cat", "kitten", "puppy", "¿Qué es a 'puppy' como 'cat' es a 'kitten'?"),
        ("apple", "fruit", "vehicle", "¿Qué es a 'vehicle' como 'apple' es a 'fruit'?"),
    ];

    for (w1, w2, w3, description) in analogies {
        if embeddings.get_embedding(w1).is_some()
            && embeddings.get_embedding(w2).is_some()
            && embeddings.get_embedding(w3).is_some() {

            println!("\n{}", description);
            println!("'{}' - '{}' + '{}' ≈", w1, w2, w3);

            let results = word_analogy(embeddings, w1, w2, w3, 3);
            for (i, (token, sim)) in results.iter().enumerate() {
                println!("  {}. {} (similitud: {:.4})", i + 1, token, sim);
            }
        }
    }
}
