// MLP para Clasificación de Texto
// Día 19: Redes Neuronales para NLP

mod mlp;
mod text_classifier;
mod batch;

use batch::{DataLoader, SimpleTokenizer, Batch};
use text_classifier::TextClassifier;
use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() {
    println!("🧠 MLP Text - Día 19: Clasificación de Texto con MLP");
    println!("================================================\n");

    // 1. Crear dataset sintético de sentimientos
    println!("📊 Creando dataset de sentimientos...");
    let (train_data, test_data, class_names) = create_sentiment_dataset();

    println!("   - {} ejemplos de entrenamiento", train_data.len());
    println!("   - {} ejemplos de prueba", test_data.len());
    println!("   - Clases: {:?}\n", class_names);

    // 2. Construir vocabulario y tokenizar
    println!("📝 Construyendo vocabulario...");
    let texts: Vec<String> = train_data.iter().map(|(text, _)| text.clone()).collect();
    let tokenizer = SimpleTokenizer::from_corpus(&texts);
    println!("   - Vocabulario: {} palabras\n", tokenizer.vocab_size());

    // 3. Tokenizar datos
    let mut train_tokenized: Vec<(Vec<usize>, usize)> = train_data
        .iter()
        .map(|(text, label)| (tokenizer.encode(text), *label))
        .collect();

    let test_tokenized: Vec<(Vec<usize>, usize)> = test_data
        .iter()
        .map(|(text, label)| (tokenizer.encode(text), *label))
        .collect();

    // 4. Crear modelo
    println!("🏗️  Creando modelo...");
    let vocab_size = tokenizer.vocab_size();
    let embedding_dim = 16;  // Reducir dimensión
    let hidden_dims = vec![32];  // Una sola capa oculta
    let num_classes = class_names.len();

    let mut classifier = TextClassifier::new(
        vocab_size,
        embedding_dim,
        hidden_dims.clone(),
        num_classes,
    );

    println!("   - Embedding dim: {}", embedding_dim);
    println!("   - Hidden layers: {:?}", &hidden_dims);
    println!("   - Clases: {}\n", num_classes);

    // 5. Entrenar modelo
    println!("🎓 Entrenando modelo...\n");
    let epochs = 300;
    let batch_size = 16;
    let learning_rate = 0.01;  // Learning rate reducido para estabilidad

    let mut rng = thread_rng();

    for epoch in 0..epochs {
        // Mezclar datos antes de crear batches para asegurar que cada batch
        // tenga una mezcla de clases
        train_tokenized.shuffle(&mut rng);
        
        let mut data_loader = DataLoader::new(train_tokenized.clone(), batch_size);
        data_loader.shuffle();

        let mut total_loss = 0.0;
        let mut num_batches = 0;

        while let Some(batch) = data_loader.next_batch() {
            let loss = classifier.train(batch, learning_rate);
            total_loss += loss;
            num_batches += 1;
        }

        let avg_loss = total_loss / num_batches as f32;

        // Calcular accuracy cada 10 epochs
        if epoch % 10 == 0 || epoch == epochs - 1 {
            let train_acc = classifier.accuracy(&train_tokenized);
            let test_acc = classifier.accuracy(&test_tokenized);

            println!(
                "Epoch {:3} | Loss: {:.4} | Train Acc: {:.2}% | Test Acc: {:.2}%",
                epoch,
                avg_loss,
                train_acc * 100.0,
                test_acc * 100.0
            );
        }
    }

    println!("\n✅ Entrenamiento completado!\n");

    // 6. Ejemplos de predicción
    println!("🔮 Probando predicciones:\n");

    let test_examples = vec![
        "me encanta este producto es increíble",
        "muy malo no lo recomiendo",
        "está bien nada especial",
        "excelente calidad lo amo",
        "terrible experiencia muy decepcionante",
        "bastante mediocre pero funciona",
    ];

    for example in test_examples {
        let tokens = tokenizer.encode(example);
        let predicted_class = classifier.predict(&tokens);
        let class_name = &class_names[predicted_class];

        println!("   \"{}\"", example);
        println!("   → Clase predicha: {} ({})\n", predicted_class, class_name);
    }

    println!("================================================");
    println!("✨ Proyecto completado exitosamente!");
}

/// Crea un dataset sintético de análisis de sentimientos
///
/// # Retorna
/// (datos_entrenamiento, datos_prueba, nombres_clases)
fn create_sentiment_dataset() -> (Vec<(String, usize)>, Vec<(String, usize)>, Vec<String>) {
    let class_names = vec![
        "positivo".to_string(),
        "negativo".to_string(),
        "neutral".to_string(),
    ];

    // Dataset de entrenamiento expandido (0: positivo, 1: negativo, 2: neutral)
    let train_data = vec![
        // Positivos (40 ejemplos)
        ("me encanta este producto es increíble".to_string(), 0),
        ("excelente calidad muy satisfecho".to_string(), 0),
        ("maravilloso lo recomiendo totalmente".to_string(), 0),
        ("fantástico superó mis expectativas".to_string(), 0),
        ("perfecto justo lo que necesitaba".to_string(), 0),
        ("estoy muy contento con la compra".to_string(), 0),
        ("lo amo es exactamente lo que buscaba".to_string(), 0),
        ("increíble producto de alta calidad".to_string(), 0),
        ("genial muy buena experiencia".to_string(), 0),
        ("espectacular vale la pena".to_string(), 0),
        ("buenísimo lo volvería a comprar".to_string(), 0),
        ("me gusta mucho funciona perfecto".to_string(), 0),
        ("excelente servicio muy recomendable".to_string(), 0),
        ("magnífico superó todas mis expectativas".to_string(), 0),
        ("brillante producto de primera calidad".to_string(), 0),
        ("extraordinario estoy encantado".to_string(), 0),
        ("sobresaliente vale cada peso".to_string(), 0),
        ("impresionante lo mejor que he comprado".to_string(), 0),
        ("maravillosa experiencia de compra".to_string(), 0),
        ("fantástica calidad recomendado cien por ciento".to_string(), 0),
        ("hermoso producto me fascina".to_string(), 0),
        ("bellísimo justo como lo quería".to_string(), 0),
        ("divino estoy super feliz".to_string(), 0),
        ("adorable lo amo completamente".to_string(), 0),
        ("sensacional mejor de lo esperado".to_string(), 0),
        ("asombroso producto excelente".to_string(), 0),
        ("fenomenal muy satisfecho con todo".to_string(), 0),
        ("óptimo cumple perfectamente".to_string(), 0),
        ("superior calidad inmejorable".to_string(), 0),
        ("ideal para lo que necesito".to_string(), 0),
        ("perfección pura lo recomiendo".to_string(), 0),
        ("cinco estrellas sin duda".to_string(), 0),
        ("totalmente satisfecho con mi compra".to_string(), 0),
        ("producto top lo mejor".to_string(), 0),
        ("calidad premium vale la pena".to_string(), 0),
        ("super contento con todo".to_string(), 0),
        ("muy feliz con mi adquisición".to_string(), 0),
        ("excepcional no tengo quejas".to_string(), 0),
        ("notable cumple y supera".to_string(), 0),
        ("destacado producto premium".to_string(), 0),

        // Negativos (40 ejemplos)
        ("muy malo no lo recomiendo".to_string(), 1),
        ("terrible experiencia muy decepcionante".to_string(), 1),
        ("pésima calidad no funciona".to_string(), 1),
        ("horrible perdí mi dinero".to_string(), 1),
        ("no me gustó para nada".to_string(), 1),
        ("mala compra no lo vuelvo a comprar".to_string(), 1),
        ("decepcionante no cumplió expectativas".to_string(), 1),
        ("desastre total no sirve".to_string(), 1),
        ("muy insatisfecho mala calidad".to_string(), 1),
        ("no funciona es una estafa".to_string(), 1),
        ("pésimo producto defectuoso".to_string(), 1),
        ("mal servicio muy malo".to_string(), 1),
        ("espantoso no lo compren".to_string(), 1),
        ("malísimo peor compra de mi vida".to_string(), 1),
        ("fatal producto de mala calidad".to_string(), 1),
        ("lamentable no sirve para nada".to_string(), 1),
        ("desastroso completamente inútil".to_string(), 1),
        ("pobre calidad muy decepcionado".to_string(), 1),
        ("deficiente no lo recomiendo jamás".to_string(), 1),
        ("deplorable perdida de dinero".to_string(), 1),
        ("penoso producto muy malo".to_string(), 1),
        ("vergonzoso no funciona bien".to_string(), 1),
        ("mediocre calidad inferior".to_string(), 1),
        ("patético no vale la pena".to_string(), 1),
        ("miserable producto fallado".to_string(), 1),
        ("terrible no lo quiero más".to_string(), 1),
        ("nefasto experiencia horrible".to_string(), 1),
        ("espantosa mala inversión".to_string(), 1),
        ("horroroso totalmente defectuoso".to_string(), 1),
        ("abominable no funciona nada".to_string(), 1),
        ("detestable producto basura".to_string(), 1),
        ("repugnante mala calidad".to_string(), 1),
        ("insoportable no sirve".to_string(), 1),
        ("intolerable producto malo".to_string(), 1),
        ("inaceptable muy decepcionante".to_string(), 1),
        ("inadmisible pérdida total".to_string(), 1),
        ("inservible completamente roto".to_string(), 1),
        ("inútil no hace nada".to_string(), 1),
        ("fracaso total de producto".to_string(), 1),
        ("fiasco completo mala compra".to_string(), 1),

        // Neutrales (40 ejemplos)
        ("está bien nada especial".to_string(), 2),
        ("es normal ni bueno ni malo".to_string(), 2),
        ("cumple su función básica".to_string(), 2),
        ("aceptable para el precio".to_string(), 2),
        ("producto estándar sin sorpresas".to_string(), 2),
        ("es lo que esperaba normal".to_string(), 2),
        ("ni chicha ni limonada regular".to_string(), 2),
        ("podría ser mejor pero funciona".to_string(), 2),
        ("nada del otro mundo aceptable".to_string(), 2),
        ("común y corriente sin más".to_string(), 2),
        ("sirve para lo básico".to_string(), 2),
        ("producto promedio estándar".to_string(), 2),
        ("regular cumple lo mínimo".to_string(), 2),
        ("pasable nada extraordinario".to_string(), 2),
        ("suficiente para el uso diario".to_string(), 2),
        ("adecuado sin destacar".to_string(), 2),
        ("correcto pero mejorable".to_string(), 2),
        ("decente sin más".to_string(), 2),
        ("razonable cumple su función".to_string(), 2),
        ("tolerable producto normal".to_string(), 2),
        ("medio pelo ni fu ni fa".to_string(), 2),
        ("así así nada especial".to_string(), 2),
        ("tirando a normal".to_string(), 2),
        ("normalito sin pena ni gloria".to_string(), 2),
        ("ordinario producto común".to_string(), 2),
        ("simple cumple lo básico".to_string(), 2),
        ("básico sin extras".to_string(), 2),
        ("estándar como cualquier otro".to_string(), 2),
        ("convencional nada nuevo".to_string(), 2),
        ("típico producto promedio".to_string(), 2),
        ("habitual sin sorpresas".to_string(), 2),
        ("usual producto normal".to_string(), 2),
        ("rutinario cumple mínimo".to_string(), 2),
        ("corriente sin destacar".to_string(), 2),
        ("medio regular tirando".to_string(), 2),
        ("moderado ni muy bien ni muy mal".to_string(), 2),
        ("intermedio calidad media".to_string(), 2),
        ("mediano producto aceptable".to_string(), 2),
        ("equilibrado sin extremos".to_string(), 2),
        ("balanceado producto estándar".to_string(), 2),
    ];

    // Dataset de prueba
    let test_data = vec![
        ("me fascina es excelente".to_string(), 0),
        ("muy bueno lo recomiendo".to_string(), 0),
        ("espantoso muy malo".to_string(), 1),
        ("no sirve es terrible".to_string(), 1),
        ("normal nada extraordinario".to_string(), 2),
        ("regular sin más".to_string(), 2),
    ];

    (train_data, test_data, class_names)
}
