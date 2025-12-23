// Tokenizadores desde Cero
// Día 17: Tokenización y Preprocesamiento

mod bpe_tokenizer;
mod char_tokenizer;
mod vocab;
mod word_tokenizer;

use bpe_tokenizer::BPETokenizer;
use char_tokenizer::CharTokenizer;
use word_tokenizer::WordTokenizer;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    match args[1].as_str() {
        "train" => {
            if args.len() < 4 {
                println!("Uso: cargo run -- train <corpus.txt> <vocab.json> [num_merges]");
                return;
            }
            let corpus_path = &args[2];
            let vocab_path = &args[3];
            let num_merges = if args.len() > 4 {
                args[4].parse().unwrap_or(500)
            } else {
                500
            };

            train_bpe(corpus_path, vocab_path, num_merges);
        }
        "encode" => {
            if args.len() < 4 {
                println!("Uso: cargo run -- encode <vocab.json> \"texto a codificar\"");
                return;
            }
            let vocab_path = &args[2];
            let text = &args[3];

            encode_text(vocab_path, text);
        }
        "demo" => {
            run_demo();
        }
        "compare" => {
            compare_tokenizers();
        }
        _ => {
            print_help();
        }
    }
}

fn print_help() {
    println!("🔤 Tokenizadores - Día 17");
    println!("\nComandos disponibles:");
    println!("  train <corpus.txt> <vocab.json> [num_merges]  - Entrena vocabulario BPE");
    println!("  encode <vocab.json> \"texto\"                    - Codifica texto con BPE");
    println!("  demo                                           - Ejecuta demostración");
    println!("  compare                                        - Compara tokenizadores");
    println!("\nEjemplos:");
    println!("  cargo run -- train corpus.txt vocab.json 1000");
    println!("  cargo run -- encode vocab.json \"Hello world\"");
    println!("  cargo run -- demo");
}

fn train_bpe(corpus_path: &str, vocab_path: &str, num_merges: usize) {
    println!("🔤 Entrenando tokenizador BPE");
    println!("📖 Cargando corpus desde: {}", corpus_path);

    // Cargar corpus
    let corpus_text = fs::read_to_string(corpus_path).expect("Error al leer corpus");

    // Dividir en líneas/párrafos
    let corpus: Vec<String> = corpus_text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|s| s.to_string())
        .collect();

    println!("📊 Corpus: {} líneas", corpus.len());
    println!("🔄 Entrenando con {} merges...\n", num_merges);

    // Entrenar tokenizador BPE
    let tokenizer = BPETokenizer::train(&corpus, num_merges);

    println!("\n✅ Entrenamiento completado!");
    println!("📝 Vocabulario: {} tokens", tokenizer.vocab.size());
    println!("🔗 Merges realizados: {}", tokenizer.merges.len());

    // Guardar vocabulario
    tokenizer
        .save_vocab(vocab_path)
        .expect("Error al guardar vocabulario");
    println!("💾 Vocabulario guardado en: {}", vocab_path);

    // Mostrar algunos ejemplos
    println!("\n📋 Primeros 20 tokens del vocabulario:");
    for (i, token) in tokenizer.vocab.id_to_token.iter().take(20).enumerate() {
        println!("  {}: '{}'", i, token);
    }

    if !tokenizer.merges.is_empty() {
        println!("\n🔗 Últimos 5 merges:");
        for (i, merge) in tokenizer.merges.iter().rev().take(5).enumerate() {
            println!("  {}: ('{}', '{}') -> '{}{}'",
                tokenizer.merges.len() - i, merge.0, merge.1, merge.0, merge.1);
        }
    }
}

fn encode_text(vocab_path: &str, text: &str) {
    println!("🔤 Codificando texto con BPE");
    println!("📖 Cargando vocabulario desde: {}", vocab_path);

    // Cargar tokenizador
    let tokenizer = BPETokenizer::load_vocab(vocab_path).expect("Error al cargar vocabulario");

    println!("📝 Vocabulario: {} tokens", tokenizer.vocab.size());
    println!("\n📝 Texto original:");
    println!("  \"{}\"", text);

    // Tokenizar
    let tokens = tokenizer.tokenize(text);
    println!("\n🔍 Tokens ({} tokens):", tokens.len());
    println!("  {:?}", tokens);

    // Codificar
    let ids = tokenizer.encode(text);
    println!("\n🔢 IDs ({} ids):", ids.len());
    println!("  {:?}", ids);

    // Decodificar
    let decoded = tokenizer.decode(&ids);
    println!("\n🔄 Texto decodificado:");
    println!("  \"{}\"", decoded);

    // Verificar
    let original_normalized = text.to_lowercase().chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>();
    let decoded_normalized = decoded.to_lowercase();

    if original_normalized.trim() == decoded_normalized.trim() {
        println!("\n✅ Codificación/Decodificación exitosa!");
    } else {
        println!("\n⚠️  Advertencia: El texto decodificado difiere del original");
    }
}

fn run_demo() {
    println!("🔤 Demostración de Tokenizadores - Día 17\n");

    // Corpus de ejemplo
    let corpus = vec![
        "The Lagos air was thick with humidity".to_string(),
        "The band launched into a hypnotic Afrobeat groove".to_string(),
        "The music was more than just entertainment".to_string(),
        "A woman named Imani moved effortlessly to the music".to_string(),
        "All around her people were dancing singing and clapping".to_string(),
    ];

    let sample_text = "The music was electric";

    println!("📖 Corpus de ejemplo ({} frases)\n", corpus.len());

    // 1. Tokenizador de caracteres
    println!("{}", "=".repeat(60));
    println!("1️⃣  TOKENIZADOR DE CARACTERES");
    println!("{}", "=".repeat(60));

    let char_tokenizer = CharTokenizer::new(&corpus);
    let char_tokens = char_tokenizer.tokenize(sample_text);
    let char_ids = char_tokenizer.encode(sample_text);

    println!("Texto: \"{}\"", sample_text);
    println!("Tokens: {:?}", char_tokens);
    println!("Cantidad de tokens: {}", char_tokens.len());
    println!("Tamaño del vocabulario: {}", char_tokenizer.vocab.size());
    println!("IDs: {:?}", char_ids);

    // 2. Tokenizador de palabras
    println!("\n{}", "=".repeat(60));
    println!("2️⃣  TOKENIZADOR DE PALABRAS");
    println!("{}", "=".repeat(60));

    let word_tokenizer = WordTokenizer::new(&corpus);
    let word_tokens = word_tokenizer.tokenize(sample_text);
    let word_ids = word_tokenizer.encode(sample_text);

    println!("Texto: \"{}\"", sample_text);
    println!("Tokens: {:?}", word_tokens);
    println!("Cantidad de tokens: {}", word_tokens.len());
    println!("Tamaño del vocabulario: {}", word_tokenizer.vocab.size());
    println!("IDs: {:?}", word_ids);

    // 3. Tokenizador BPE
    println!("\n{}", "=".repeat(60));
    println!("3️⃣  TOKENIZADOR BPE (Byte Pair Encoding)");
    println!("{}", "=".repeat(60));

    println!("Entrenando BPE con 50 merges...\n");
    let bpe_tokenizer = BPETokenizer::train(&corpus, 50);
    let bpe_tokens = bpe_tokenizer.tokenize(sample_text);
    let bpe_ids = bpe_tokenizer.encode(sample_text);

    println!("Texto: \"{}\"", sample_text);
    println!("Tokens: {:?}", bpe_tokens);
    println!("Cantidad de tokens: {}", bpe_tokens.len());
    println!("Tamaño del vocabulario: {}", bpe_tokenizer.vocab.size());
    println!("Merges realizados: {}", bpe_tokenizer.merges.len());
    println!("IDs: {:?}", bpe_ids);

    // Comparación
    println!("\n{}", "=".repeat(60));
    println!("📊 COMPARACIÓN DE TOKENIZADORES");
    println!("{}", "=".repeat(60));

    println!("{:<20} | {:>10} | {:>15}", "Tokenizador", "Tokens", "Vocab Size");
    println!("{}", "-".repeat(60));
    println!(
        "{:<20} | {:>10} | {:>15}",
        "Caracteres",
        char_tokens.len(),
        char_tokenizer.vocab.size()
    );
    println!(
        "{:<20} | {:>10} | {:>15}",
        "Palabras",
        word_tokens.len(),
        word_tokenizer.vocab.size()
    );
    println!(
        "{:<20} | {:>10} | {:>15}",
        "BPE",
        bpe_tokens.len(),
        bpe_tokenizer.vocab.size()
    );

    println!("\n💡 Observaciones:");
    println!("  • Caracteres: Vocabulario pequeño, muchos tokens");
    println!("  • Palabras: Vocabulario grande, pocos tokens");
    println!("  • BPE: Balance entre vocabulario y cantidad de tokens");
}

fn compare_tokenizers() {
    println!("🔤 Comparación Detallada de Tokenizadores\n");

    let test_cases = vec![
        "hello",
        "hello world",
        "The quick brown fox jumps",
        "tokenization",
        "untokenizable",
    ];

    let corpus = vec![
        "hello world".to_string(),
        "the quick brown fox".to_string(),
        "jumps over the lazy dog".to_string(),
    ];

    let char_tok = CharTokenizer::new(&corpus);
    let word_tok = WordTokenizer::new(&corpus);
    let bpe_tok = BPETokenizer::train(&corpus, 20);

    for text in test_cases {
        println!("{}", "=".repeat(60));
        println!("Texto: \"{}\"", text);
        println!("{}", "-".repeat(60));

        let char_tokens = char_tok.tokenize(text);
        let word_tokens = word_tok.tokenize(text);
        let bpe_tokens = bpe_tok.tokenize(text);

        println!("Caracteres ({:2} tokens): {:?}", char_tokens.len(), char_tokens);
        println!("Palabras    ({:2} tokens): {:?}", word_tokens.len(), word_tokens);
        println!("BPE         ({:2} tokens): {:?}", bpe_tokens.len(), bpe_tokens);
        println!();
    }

    println!("\n📊 Resumen:");
    println!("  Vocabulario Caracteres: {} tokens", char_tok.vocab.size());
    println!("  Vocabulario Palabras:   {} tokens", word_tok.vocab.size());
    println!("  Vocabulario BPE:        {} tokens", bpe_tok.vocab.size());
}
