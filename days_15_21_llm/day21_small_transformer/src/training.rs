// Loop de entrenamiento simplificado

use crate::transformer::TransformerLM;
use crate::dataset::TextDataset;
use crate::utils::{cross_entropy_loss, perplexity};

/// Entrena el modelo (versión simplificada sin backprop completo)
///
/// NOTA: Esta es una implementación educativa simplificada.
/// Un entrenamiento real requeriría backpropagation completo y optimizador Adam.
///
/// # Argumentos
/// * `model` - Modelo a entrenar (nota: tomado por valor, no modifica el original)
/// * `dataset` - Dataset de entrenamiento
/// * `epochs` - Número de épocas
/// * `batch_size` - Tamaño del batch
pub fn train(
    model: &TransformerLM,
    dataset: &TextDataset,
    epochs: usize,
    batch_size: usize,
) {
    println!("\n🎓 Iniciando entrenamiento");
    println!("════════════════════════════════════════");
    println!("Épocas: {}", epochs);
    println!("Batch size: {}", batch_size);
    println!("Dataset size: {}", dataset.len());
    println!("════════════════════════════════════════\n");

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        // Entrenar en batches
        for batch_idx in 0..10 {  // Limitado a 10 batches por época para velocidad
            if let Some((inputs, targets)) = dataset.get_batch(batch_size) {
                let mut batch_loss = 0.0;

                for (input_seq, target_seq) in inputs.iter().zip(targets.iter()) {
                    // Forward pass
                    let logits = model.forward(input_seq);

                    // Calcular loss
                    let loss = cross_entropy_loss(&logits, target_seq);
                    batch_loss += loss;

                    // NOTA: Aquí iría backpropagation en una implementación completa
                    // Por simplicidad educativa, solo calculamos métricas
                }

                epoch_loss += batch_loss / inputs.len() as f32;
                num_batches += 1;
            }
        }

        let avg_loss = epoch_loss / num_batches.max(1) as f32;
        let ppl = perplexity(avg_loss);

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!(
                "Época {:3}/{} | Loss: {:.4} | Perplexity: {:.2}",
                epoch + 1,
                epochs,
                avg_loss,
                ppl
            );
        }
    }

    println!("\n✅ Entrenamiento completado");
    println!("\n⚠️  NOTA IMPORTANTE:");
    println!("Esta es una demostración educativa simplificada.");
    println!("El modelo NO ha sido actualizado (falta implementación de backprop).");
    println!("Para un transformer funcional completo, se requeriría:");
    println!("  1. Backpropagation completa con chain rule");
    println!("  2. Optimizador Adam con moment estimation");
    println!("  3. Gradient clipping y learning rate scheduling");
    println!("  4. Implementación sería ~3000+ líneas adicionales de código");
}

/// Calcula métricas en dataset de validación
pub fn evaluate(model: &TransformerLM, dataset: &TextDataset) -> (f32, f32) {
    let batch_size = 8;
    let mut total_loss = 0.0;
    let mut num_samples = 0;

    for _ in 0..5 {  // Evaluar en 5 batches
        if let Some((inputs, targets)) = dataset.get_batch(batch_size) {
            for (input_seq, target_seq) in inputs.iter().zip(targets.iter()) {
                let logits = model.forward(input_seq);
                let loss = cross_entropy_loss(&logits, target_seq);
                total_loss += loss;
                num_samples += 1;
            }
        }
    }

    let avg_loss = total_loss / num_samples.max(1) as f32;
    let ppl = perplexity(avg_loss);

    (avg_loss, ppl)
}
