// Métricas de Evaluación: Perplexity
// Día 16: Evaluación de Modelos de Lenguaje

mod dataset;
mod ngram;
mod visualization;

use dataset::{load_africa_galore, tokenize_corpus};
use ngram::{
    calculate_perplexity_bigram, calculate_perplexity_trigram, calculate_perplexity_unigram,
    BigramModel, TrigramModel, UnigramModel,
};
use visualization::plot_perplexity_comparison;

fn main() {
    println!("📈 Perplexity - Día 16");
    println!("═══════════════════════════════════════════════════");
    println!("Métricas de Evaluación de Modelos de Lenguaje");
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
            eprintln!("💡 Asegúrate de ejecutar desde: days_15_21_llm/day16_perplexity/");
            eprintln!("💡 Y que el dataset esté en: datasets/africa_galore.json");
            return;
        }
    };

    // 2. Tokenizar corpus
    println!("\n🔤 Tokenizando corpus...");
    let all_tokens = tokenize_corpus(&texts);
    println!("✅ Total de tokens: {}", all_tokens.len());

    // Dividir en entrenamiento (80%) y prueba (20%)
    let split_index = (all_tokens.len() as f64 * 0.8) as usize;
    let train_tokens = &all_tokens[..split_index];
    let test_tokens = &all_tokens[split_index..];
    println!("✅ Tokens de entrenamiento: {}", train_tokens.len());
    println!("✅ Tokens de prueba: {}", test_tokens.len());

    println!("\n═══════════════════════════════════════════════════");
    println!("🎯 ¿QUÉ ES PERPLEXITY?");
    println!("═══════════════════════════════════════════════════");
    println!("\n📖 Perplexity mide qué tan 'sorprendido' está un modelo");
    println!("   ante nuevos datos. Es una medida de incertidumbre.");
    println!();
    println!("   Fórmula: PPL = exp(-1/N * Σ log P(palabra_i | contexto))");
    println!();
    println!("   🎯 Menor perplexity = Mejor modelo");
    println!("   - PPL = 1: Predicción perfecta (imposible en la práctica)");
    println!("   - PPL = 100: El modelo tiene ~100 opciones equiprobables");
    println!("   - PPL = 1000+: Modelo muy confundido");
    println!();
    println!("   💡 Interpretación intuitiva:");
    println!("      Si PPL = 50, el modelo está tan sorprendido como si");
    println!("      tuviera que elegir entre 50 palabras igualmente probables");

    println!("\n═══════════════════════════════════════════════════");
    println!("🔬 ENTRENANDO Y EVALUANDO MODELOS");
    println!("═══════════════════════════════════════════════════");

    // 3. Entrenar modelos
    println!("\n🎓 Entrenando modelo Unigram...");
    let unigram_model = UnigramModel::new(train_tokens);
    println!("✅ Unigram entrenado");

    println!("🎓 Entrenando modelo Bigram...");
    let bigram_model = BigramModel::new(train_tokens);
    println!("✅ Bigram entrenado");

    println!("🎓 Entrenando modelo Trigram...");
    let trigram_model = TrigramModel::new(train_tokens);
    println!("✅ Trigram entrenado");

    // 4. Calcular perplexity en conjunto de prueba
    println!("\n📊 Calculando perplexity en conjunto de prueba...");
    let unigram_ppl = calculate_perplexity_unigram(&unigram_model, test_tokens);
    let bigram_ppl = calculate_perplexity_bigram(&bigram_model, test_tokens);
    let trigram_ppl = calculate_perplexity_trigram(&trigram_model, test_tokens);

    println!("\n═══════════════════════════════════════════════════");
    println!("📊 RESULTADOS: PERPLEXITY EN CONJUNTO DE PRUEBA");
    println!("═══════════════════════════════════════════════════");
    println!("\n   Modelo      | Perplexity | Interpretación");
    println!("   ------------|------------|---------------");
    println!("   Unigram     | {:>10.2} | {}", unigram_ppl, interpret_perplexity(unigram_ppl));
    println!("   Bigram      | {:>10.2} | {}", bigram_ppl, interpret_perplexity(bigram_ppl));
    println!("   Trigram     | {:>10.2} | {}", trigram_ppl, interpret_perplexity(trigram_ppl));

    // 5. Análisis de mejora
    println!("\n📈 Análisis de mejora:");
    let bigram_improvement = ((unigram_ppl - bigram_ppl) / unigram_ppl) * 100.0;
    let trigram_improvement = ((bigram_ppl - trigram_ppl) / bigram_ppl) * 100.0;
    println!("   • Bigram vs Unigram: {:.1}% de mejora", bigram_improvement);
    println!("   • Trigram vs Bigram: {:.1}% de mejora", trigram_improvement);

    println!("\n═══════════════════════════════════════════════════");
    println!("📉 ANÁLISIS: PERPLEXITY VS CANTIDAD DE DATOS");
    println!("═══════════════════════════════════════════════════");

    // 6. Evaluar con diferentes cantidades de datos de entrenamiento
    let training_fractions = vec![0.2, 0.4, 0.6, 0.8, 1.0];
    let mut learning_curves = vec![
        ("Unigram", Vec::new()),
        ("Bigram", Vec::new()),
        ("Trigram", Vec::new()),
    ];

    println!("\n🔬 Entrenando modelos con diferentes cantidades de datos...");
    for &fraction in &training_fractions {
        let train_size = (train_tokens.len() as f64 * fraction) as usize;
        let partial_train = &train_tokens[..train_size];

        let unigram = UnigramModel::new(partial_train);
        let bigram = BigramModel::new(partial_train);
        let trigram = TrigramModel::new(partial_train);

        let unigram_ppl = calculate_perplexity_unigram(&unigram, test_tokens);
        let bigram_ppl = calculate_perplexity_bigram(&bigram, test_tokens);
        let trigram_ppl = calculate_perplexity_trigram(&trigram, test_tokens);

        learning_curves[0].1.push((fraction, unigram_ppl));
        learning_curves[1].1.push((fraction, bigram_ppl));
        learning_curves[2].1.push((fraction, trigram_ppl));

        println!("   {}% datos: Unigram={:.1}, Bigram={:.1}, Trigram={:.1}",
                 (fraction * 100.0) as u32, unigram_ppl, bigram_ppl, trigram_ppl);
    }

    println!("\n💡 Observaciones:");
    println!("   • Más datos de entrenamiento generalmente reducen perplexity");
    println!("   • Modelos de mayor orden (trigram) se benefician más de datos adicionales");
    println!("   • Hay rendimientos decrecientes: duplicar datos no duplica la mejora");

    println!("\n═══════════════════════════════════════════════════");
    println!("📊 GENERANDO VISUALIZACIONES");
    println!("═══════════════════════════════════════════════════");

    // 7. Crear visualización de comparación
    let model_names = vec!["Unigram", "Bigram", "Trigram"];
    let perplexities = vec![unigram_ppl, bigram_ppl, trigram_ppl];

    match plot_perplexity_comparison(&model_names, &perplexities, "perplexity_comparison.png") {
        Ok(_) => println!("\n✅ Gráfica guardada: perplexity_comparison.png"),
        Err(e) => eprintln!("\n❌ Error creando gráfica: {}", e),
    }

    // 8. Crear visualización de curvas de aprendizaje
    match visualization::plot_learning_curves(&learning_curves, "learning_curves.png") {
        Ok(_) => println!("✅ Curvas de aprendizaje guardadas: learning_curves.png"),
        Err(e) => eprintln!("❌ Error creando curvas: {}", e),
    }

    println!("\n═══════════════════════════════════════════════════");
    println!("🎓 CONCLUSIONES");
    println!("═══════════════════════════════════════════════════");
    println!("\n✅ Lecciones aprendidas sobre Perplexity:");
    println!();
    println!("   1. 📏 Métrica estándar para evaluar modelos de lenguaje");
    println!("      - Permite comparar diferentes arquitecturas");
    println!("      - Independiente del tamaño del vocabulario (normalizada)");
    println!();
    println!("   2. 🎯 Modelos de mayor contexto (trigram) tienen menor perplexity");
    println!("      - Capturan más patrones del lenguaje");
    println!("      - Pero requieren más datos para entrenar bien");
    println!();
    println!("   3. 📊 Más datos = Mejor perplexity (hasta cierto punto)");
    println!("      - Importante tener suficientes datos de entrenamiento");
    println!("      - Los rendimientos decrecen con más datos");
    println!();
    println!("   4. ⚠️  Limitaciones de perplexity:");
    println!("      - No mide coherencia semántica directamente");
    println!("      - Un modelo puede tener baja perplexity y aún generar");
    println!("        texto sin sentido en contextos específicos");
    println!("      - Debe complementarse con evaluación humana");
    println!();
    println!("   5. 🚀 Próximos pasos:");
    println!("      - Tokenización avanzada (BPE) - Día 17");
    println!("      - Embeddings para capturar significado - Día 18");
    println!("      - Modelos neuronales (Transformers) - Días 19-21");

    println!("\n✅ Análisis de perplexity completado!");
    println!("═══════════════════════════════════════════════════");
}

/// Interpreta un valor de perplexity y retorna una descripción
fn interpret_perplexity(ppl: f64) -> &'static str {
    if ppl < 10.0 {
        "Excelente"
    } else if ppl < 50.0 {
        "Muy bueno"
    } else if ppl < 100.0 {
        "Bueno"
    } else if ppl < 200.0 {
        "Aceptable"
    } else if ppl < 500.0 {
        "Regular"
    } else {
        "Pobre"
    }
}
