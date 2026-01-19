// Generación de texto autoregressiva

use crate::transformer::TransformerLM;
use crate::tokenizer::SimpleTokenizer;
use rand::Rng;

pub fn generate(
    model: &TransformerLM,
    tokenizer: &SimpleTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> String {
    let mut tokens = tokenizer.encode(prompt, false);
    
    println!("Generando {} tokens desde: '{}'", max_tokens, prompt);
    println!("Tokens iniciales: {:?}", tokens);

    for i in 0..max_tokens {
        let context = if tokens.len() > model.max_seq_len {
            &tokens[tokens.len() - model.max_seq_len..]
        } else {
            &tokens
        };

        let next_token = if temperature > 0.0 {
            sample_with_temperature(model, context, temperature)
        } else {
            model.predict_next(context)
        };

        if next_token == tokenizer.eos_token_id {
            break;
        }

        tokens.push(next_token);

        if (i + 1) % 10 == 0 {
            println!("  Generados {} tokens...", i + 1);
        }
    }

    tokenizer.decode(&tokens)
}

fn sample_with_temperature(model: &TransformerLM, context: &[usize], temperature: f32) -> usize {
    let logits = model.forward(context);
    let last_logits = logits.row(logits.nrows() - 1);

    let scaled_logits: Vec<f32> = last_logits.iter().map(|&x| x / temperature).collect();

    let max_val = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled_logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();

    let mut rng = rand::thread_rng();
    let rand_val: f32 = rng.gen_range(0.0..1.0);
    let mut cumsum = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val < cumsum {
            return i;
        }
    }

    probs.len() - 1
}
