// Modelos N-gram desde Cero
// Día 15-16: Fundamentos de Modelos de Lenguaje

mod dataset;
mod ngram;
mod sampling;

use dataset::{load_africa_galore, tokenize_corpus};
use ngram::{
    calculate_perplexity_bigram, calculate_perplexity_trigram, calculate_perplexity_unigram,
    BigramModel, TrigramModel, UnigramModel,
};
use rand::thread_rng;
use sampling::{generate_bigram, generate_trigram, generate_unigram, random_start_bigram};

fn main() {
    println!("🚀 Modelos N-gram - Día 15");
    println!("═══════════════════════════════════════════════════");
    println!();

    // 1. Cargar dataset Africa Galore
    println!("📦 Cargando dataset Africa Galore...");
    let dataset_path = "../../datasets/africa_galore.json";
    let texts = match load_africa_galore(dataset_path) {
        Ok(texts) => {
            println!("✅ Dataset cargado: {} textos", texts.len());
            texts
        }
        Err(e) => {
            eprintln!("❌ Error cargando dataset: {}", e);
            eprintln!("💡 Asegúrate de ejecutar desde: days_15_21_llm/day15_ngram_models/");
            eprintln!("💡 Y que el dataset esté en: datasets/africa_galore.json");
            return;
        }
    };

    // 2. Tokenizar corpus
    println!("\n🔤 Tokenizando corpus...");
    let all_tokens = tokenize_corpus(&texts);
    println!("✅ Total de tokens: {}", all_tokens.len());

    // Calcular vocabulario único
    let vocab_size: std::collections::HashSet<_> = all_tokens.iter().collect();
    println!("✅ Vocabulario único: {} palabras", vocab_size.len());

    // Dividir en entrenamiento (80%) y prueba (20%)
    let split_index = (all_tokens.len() as f64 * 0.8) as usize;
    let train_tokens = &all_tokens[..split_index];
    let test_tokens = &all_tokens[split_index..];
    println!("✅ Tokens de entrenamiento: {}", train_tokens.len());
    println!("✅ Tokens de prueba: {}", test_tokens.len());

    println!("\n═══════════════════════════════════════════════════");
    println!("📊 MODELO UNIGRAM");
    println!("═══════════════════════════════════════════════════");

    // 3. Entrenar modelo Unigram
    println!("\n🎓 Entrenando modelo Unigram...");
    let unigram_model = UnigramModel::new(train_tokens);
    println!("✅ Modelo Unigram entrenado");

    // Mostrar palabras más frecuentes
    let mut word_freq: Vec<_> = unigram_model.word_frequencies().iter().collect();
    word_freq.sort_by(|a, b| b.1.cmp(a.1));
    println!("\n📈 Top 10 palabras más frecuentes:");
    for (i, (word, count)) in word_freq.iter().take(10).enumerate() {
        let prob = unigram_model.probability(word);
        println!("   {}. '{}' - {} veces (P={:.4})", i + 1, word, count, prob);
    }

    // Generar texto con Unigram
    println!("\n✍️  Generación de texto (Unigram - 30 palabras):");
    let unigram_text = generate_unigram(&unigram_model, 30);
    println!("   {}", unigram_text.join(" "));

    // Calcular perplexity
    let unigram_perplexity = calculate_perplexity_unigram(&unigram_model, test_tokens);
    println!("\n📉 Perplexity (Unigram): {:.2}", unigram_perplexity);

    println!("\n═══════════════════════════════════════════════════");
    println!("📊 MODELO BIGRAM");
    println!("═══════════════════════════════════════════════════");

    // 4. Entrenar modelo Bigram
    println!("\n🎓 Entrenando modelo Bigram...");
    let bigram_model = BigramModel::new(train_tokens);
    println!("✅ Modelo Bigram entrenado");

    // Ejemplo de probabilidades condicionales
    println!("\n🔍 Ejemplos de probabilidades P(w2|w1):");
    let example_pairs = [("the", "music"), ("in", "the"), ("was", "a")];
    for (w1, w2) in &example_pairs {
        let prob = bigram_model.probability(w1, w2);
        println!("   P('{}' | '{}') = {:.4}", w2, w1, prob);
    }

    // Generar texto con Bigram
    println!("\n✍️  Generación de texto (Bigram - 30 palabras):");
    let bigram_text = generate_bigram(&bigram_model, "the", 29);
    println!("   {}", bigram_text.join(" "));

    // Calcular perplexity
    let bigram_perplexity = calculate_perplexity_bigram(&bigram_model, test_tokens);
    println!("\n📉 Perplexity (Bigram): {:.2}", bigram_perplexity);

    println!("\n═══════════════════════════════════════════════════");
    println!("📊 MODELO TRIGRAM");
    println!("═══════════════════════════════════════════════════");

    // 5. Entrenar modelo Trigram
    println!("\n🎓 Entrenando modelo Trigram...");
    let trigram_model = TrigramModel::new(train_tokens);
    println!("✅ Modelo Trigram entrenado");

    // Ejemplo de probabilidades condicionales
    println!("\n🔍 Ejemplos de probabilidades P(w3|w1,w2):");
    let example_triples = [("the", "music", "was"), ("in", "the", "club"), ("of", "the", "music")];
    for (w1, w2, w3) in &example_triples {
        let prob = trigram_model.probability(w1, w2, w3);
        println!("   P('{}' | '{}', '{}') = {:.4}", w3, w1, w2, prob);
    }

    // Generar texto con Trigram
    println!("\n✍️  Generación de texto (Trigram - 30 palabras):");
    let mut rng = thread_rng();
    let vocab = bigram_model.vocabulary();
    let start_pair = random_start_bigram(&vocab, &mut rng);
    let trigram_text = generate_trigram(&trigram_model, (&start_pair.0, &start_pair.1), 28);
    println!("   {}", trigram_text.join(" "));

    // Calcular perplexity
    let trigram_perplexity = calculate_perplexity_trigram(&trigram_model, test_tokens);
    println!("\n📉 Perplexity (Trigram): {:.2}", trigram_perplexity);

    println!("\n═══════════════════════════════════════════════════");
    println!("🏆 COMPARACIÓN DE MODELOS");
    println!("═══════════════════════════════════════════════════");

    println!("\n📊 Resumen de Perplexity (menor es mejor):");
    println!("   Unigram:  {:.2}", unigram_perplexity);
    println!("   Bigram:   {:.2}", bigram_perplexity);
    println!("   Trigram:  {:.2}", trigram_perplexity);

    println!("\n💡 Interpretación:");
    println!("   - Perplexity mide qué tan 'sorprendido' está el modelo");
    println!("   - Menor perplexity = mejor predicción");
    println!("   - Modelos de mayor orden (trigram) suelen tener menor perplexity");
    println!("   - Pero requieren más datos y pueden sufrir de overfitting");

    println!("\n✅ Análisis completo de modelos N-gram finalizado!");
    println!("═══════════════════════════════════════════════════");
}
