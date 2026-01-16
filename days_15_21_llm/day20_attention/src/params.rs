// Conteo de Parámetros Entrenables
//
// Funciones para calcular el número de parámetros entrenables
// en los diferentes componentes del mecanismo de atención

/// Calcula el número de parámetros en Scaled Dot-Product Attention
///
/// # Argumentos
/// * `d_model` - Dimensión del modelo
/// * `d_k` - Dimensión de keys/queries
/// * `d_v` - Dimensión de values
///
/// # Retorna
/// Número total de parámetros entrenables
///
/// # Componentes
/// - W_Q: d_model × d_k
/// - W_K: d_model × d_k
/// - W_V: d_model × d_v
/// - W_O: d_v × d_model
pub fn contar_params_attention(d_model: usize, d_k: usize, d_v: usize) -> usize {
    let w_q = d_model * d_k;
    let w_k = d_model * d_k;
    let w_v = d_model * d_v;
    let w_o = d_v * d_model;

    w_q + w_k + w_v + w_o
}

/// Calcula el número de parámetros en Multi-Head Attention
///
/// # Argumentos
/// * `d_model` - Dimensión del modelo
/// * `num_heads` - Número de cabezas de atención
///
/// # Retorna
/// Número total de parámetros entrenables
///
/// # Componentes
/// Para cada cabeza h:
/// - W_Q_h: d_model × (d_model/num_heads)
/// - W_K_h: d_model × (d_model/num_heads)
/// - W_V_h: d_model × (d_model/num_heads)
/// - W_O: d_model × d_model (compartido)
pub fn contar_params_multihead(d_model: usize, num_heads: usize) -> usize {
    assert!(d_model % num_heads == 0, "d_model debe ser divisible por num_heads");

    let d_k = d_model / num_heads;

    // Parámetros por cabeza: Q, K, V proyecciones
    let params_per_head = 3 * (d_model * d_k);

    // Total para todas las cabezas + proyección de salida
    let total_heads = num_heads * params_per_head;
    let w_o = d_model * d_model;

    total_heads + w_o
}

/// Calcula el número de parámetros en Positional Encoding
///
/// # Argumentos
/// * `encoding_type` - Tipo de codificación (Sinusoidal o Learnable)
/// * `max_len` - Longitud máxima de secuencia
/// * `d_model` - Dimensión del modelo
///
/// # Retorna
/// Número de parámetros entrenables
///
/// # Notas
/// - Sinusoidal: 0 parámetros (fijo, no entrenable)
/// - Learnable: max_len × d_model parámetros
pub fn contar_params_positional_encoding(
    encoding_type: &str,
    max_len: usize,
    d_model: usize,
) -> usize {
    match encoding_type {
        "sinusoidal" => 0, // No tiene parámetros entrenables
        "learnable" => max_len * d_model,
        _ => panic!("Tipo de encoding desconocido: {}", encoding_type),
    }
}

/// Estructura para almacenar el desglose de parámetros
#[derive(Debug)]
pub struct ParameterBreakdown {
    pub component: String,
    pub parameters: usize,
}

impl ParameterBreakdown {
    pub fn new(component: &str, parameters: usize) -> Self {
        Self {
            component: component.to_string(),
            parameters,
        }
    }
}

/// Genera un reporte detallado de parámetros para una configuración
///
/// # Argumentos
/// * `d_model` - Dimensión del modelo
/// * `num_heads` - Número de cabezas
/// * `max_len` - Longitud máxima de secuencia
/// * `use_learnable_pe` - Si se usa PE aprendible
pub fn generar_reporte(
    d_model: usize,
    num_heads: usize,
    max_len: usize,
    use_learnable_pe: bool,
) {
    println!("📊 Reporte de Parámetros Entrenables");
    println!("=====================================\n");

    println!("Configuración:");
    println!("  - d_model: {}", d_model);
    println!("  - num_heads: {}", num_heads);
    println!("  - d_k (por cabeza): {}", d_model / num_heads);
    println!("  - max_len: {}", max_len);
    println!("  - PE type: {}\n", if use_learnable_pe { "Learnable" } else { "Sinusoidal" });

    let mut breakdown = Vec::new();

    // Single-head attention
    let d_k = d_model / num_heads;
    let single_head = contar_params_attention(d_model, d_k, d_k);
    breakdown.push(ParameterBreakdown::new("Single-Head Attention", single_head));

    // Multi-head attention
    let multihead = contar_params_multihead(d_model, num_heads);
    breakdown.push(ParameterBreakdown::new("Multi-Head Attention", multihead));

    // Positional encoding
    let pe_type = if use_learnable_pe { "learnable" } else { "sinusoidal" };
    let pe_params = contar_params_positional_encoding(pe_type, max_len, d_model);
    breakdown.push(ParameterBreakdown::new("Positional Encoding", pe_params));

    // Total
    let total = multihead + pe_params;
    breakdown.push(ParameterBreakdown::new("TOTAL", total));

    println!("Desglose de parámetros:");
    println!("{:<30} {:>15}", "Componente", "Parámetros");
    println!("{}", "-".repeat(47));

    for item in &breakdown {
        println!("{:<30} {:>15}", item.component, format_number(item.parameters));
    }

    println!("\n💡 Notas:");
    println!("  - Sinusoidal PE: 0 parámetros (fijo, no entrenable)");
    println!("  - Learnable PE: {} parámetros (entrenables)", format_number(max_len * d_model));
    println!("  - Multi-head usa {} cabezas, cada una con d_k = {}", num_heads, d_k);
}

/// Formatea un número con separadores de miles
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(c);
        count += 1;
    }

    result.chars().rev().collect()
}

/// Compara diferentes configuraciones
pub fn comparar_configuraciones() {
    println!("\n🔍 Comparación de Configuraciones");
    println!("==================================\n");

    let configs = vec![
        ("Pequeño", 64, 4, 128),
        ("Mediano", 256, 8, 512),
        ("Grande", 512, 16, 1024),
        ("GPT-2 Small", 768, 12, 1024),
    ];

    println!("{:<20} {:>10} {:>10} {:>15}", "Configuración", "d_model", "heads", "Parámetros");
    println!("{}", "-".repeat(57));

    for (name, d_model, num_heads, max_len) in configs {
        let params = contar_params_multihead(d_model, num_heads);
        println!("{:<20} {:>10} {:>10} {:>15}",
            name,
            d_model,
            num_heads,
            format_number(params)
        );
    }

    println!("\n💡 Observaciones:");
    println!("  - Los parámetros crecen cuadráticamente con d_model");
    println!("  - El número de cabezas no afecta el total de parámetros");
    println!("  - Solo cambia cómo se distribuyen los cálculos");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contar_params_attention() {
        // d_model=64, d_k=16, d_v=16
        let params = contar_params_attention(64, 16, 16);
        // W_Q: 64*16 = 1024
        // W_K: 64*16 = 1024
        // W_V: 64*16 = 1024
        // W_O: 16*64 = 1024
        // Total: 4096
        assert_eq!(params, 4096);
    }

    #[test]
    fn test_contar_params_multihead() {
        // d_model=64, num_heads=4
        // d_k = 64/4 = 16 por cabeza
        let params = contar_params_multihead(64, 4);

        // Por cabeza: 64*16 * 3 (Q,K,V) = 3072
        // Total cabezas: 3072 * 4 = 12288
        // W_O: 64*64 = 4096
        // Total: 16384
        assert_eq!(params, 16384);
    }

    #[test]
    fn test_contar_params_pe_sinusoidal() {
        let params = contar_params_positional_encoding("sinusoidal", 128, 64);
        assert_eq!(params, 0); // Sinusoidal no tiene parámetros
    }

    #[test]
    fn test_contar_params_pe_learnable() {
        let params = contar_params_positional_encoding("learnable", 128, 64);
        assert_eq!(params, 128 * 64); // max_len * d_model
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(42), "42");
    }
}
