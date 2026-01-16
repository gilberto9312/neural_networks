// Visualización de pesos de atención
//
// Genera heatmaps para visualizar los patrones de atención

use ndarray::Array2;
use plotters::prelude::*;
use std::error::Error;

/// Visualiza los pesos de atención como un heatmap
///
/// # Argumentos
/// * `attention_weights` - Matriz de pesos de atención (seq_len, seq_len)
/// * `output_path` - Ruta donde guardar la imagen
/// * `title` - Título del gráfico
/// * `labels` - Etiquetas opcionales para los tokens
///
/// # Retorna
/// Result indicando éxito o error
pub fn plot_attention_heatmap(
    attention_weights: &Array2<f32>,
    output_path: &str,
    title: &str,
    labels: Option<&[String]>,
) -> Result<(), Box<dyn Error>> {
    let seq_len = attention_weights.shape()[0];

    // Crear área de dibujo
    let root = BitMapBackend::new(output_path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..seq_len as f32, 0f32..seq_len as f32)?;

    chart
        .configure_mesh()
        .x_desc("Key Position")
        .y_desc("Query Position")
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    // Encontrar min y max para normalización
    let min_val = attention_weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = attention_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Dibujar celdas del heatmap
    for i in 0..seq_len {
        for j in 0..seq_len {
            let value = attention_weights[[i, j]];

            // Normalizar entre 0 y 1
            let normalized = if max_val > min_val {
                (value - min_val) / (max_val - min_val)
            } else {
                0.5
            };

            // Mapear a color (azul oscuro -> rojo brillante)
            let color = RGBColor(
                (255.0 * normalized) as u8,
                (100.0 * (1.0 - normalized)) as u8,
                (200.0 * (1.0 - normalized)) as u8,
            );

            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (j as f32, i as f32),
                    ((j + 1) as f32, (i + 1) as f32),
                ],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

/// Visualiza múltiples cabezas de atención en un solo gráfico
///
/// # Argumentos
/// * `attention_heads` - Vector de matrices de atención (una por cabeza)
/// * `output_path` - Ruta donde guardar la imagen
/// * `title` - Título del gráfico
///
/// # Retorna
/// Result indicando éxito o error
pub fn plot_multihead_attention(
    attention_heads: &[Array2<f32>],
    output_path: &str,
    title: &str,
) -> Result<(), Box<dyn Error>> {
    let num_heads = attention_heads.len();
    let seq_len = attention_heads[0].shape()[0];

    // Calcular layout de grid (intentar hacer cuadrado)
    let cols = (num_heads as f32).sqrt().ceil() as usize;
    let rows = (num_heads + cols - 1) / cols;

    let cell_size = 250;
    let width = cols * cell_size + 100;
    let height = rows * cell_size + 100;

    let root = BitMapBackend::new(output_path, (width as u32, height as u32))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((rows, cols));

    for (head_idx, head_weights) in attention_heads.iter().enumerate() {
        if head_idx >= areas.len() {
            break;
        }

        let area = &areas[head_idx];

        let mut chart = ChartBuilder::on(area)
            .caption(
                format!("Head {}", head_idx),
                ("sans-serif", 20).into_font(),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f32..seq_len as f32, 0f32..seq_len as f32)?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()?;

        // Encontrar min y max para esta cabeza
        let min_val = head_weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = head_weights
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        // Dibujar heatmap
        for i in 0..seq_len {
            for j in 0..seq_len {
                let value = head_weights[[i, j]];

                let normalized = if max_val > min_val {
                    (value - min_val) / (max_val - min_val)
                } else {
                    0.5
                };

                let color = RGBColor(
                    (255.0 * normalized) as u8,
                    (100.0 * (1.0 - normalized)) as u8,
                    (200.0 * (1.0 - normalized)) as u8,
                );

                chart.draw_series(std::iter::once(Rectangle::new(
                    [(j as f32, i as f32), ((j + 1) as f32, (i + 1) as f32)],
                    color.filled(),
                )))?;
            }
        }
    }

    root.present()?;
    Ok(())
}

/// Visualiza la codificación posicional como un heatmap
///
/// # Argumentos
/// * `positional_encoding` - Matriz de codificación posicional (max_len, d_model)
/// * `output_path` - Ruta donde guardar la imagen
/// * `title` - Título del gráfico
///
/// # Retorna
/// Result indicando éxito o error
pub fn plot_positional_encoding(
    positional_encoding: &Array2<f32>,
    output_path: &str,
    title: &str,
) -> Result<(), Box<dyn Error>> {
    let max_len = positional_encoding.shape()[0];
    let d_model = positional_encoding.shape()[1];

    let root = BitMapBackend::new(output_path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..d_model as f32, 0f32..max_len as f32)?;

    chart
        .configure_mesh()
        .x_desc("Dimensión del Embedding")
        .y_desc("Posición en la Secuencia")
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    // Encontrar min y max
    let min_val = positional_encoding
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let max_val = positional_encoding
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Dibujar heatmap
    for pos in 0..max_len {
        for dim in 0..d_model {
            let value = positional_encoding[[pos, dim]];

            let normalized = if max_val > min_val {
                (value - min_val) / (max_val - min_val)
            } else {
                0.5
            };

            // Usar esquema de color divergente (azul-blanco-rojo)
            let color = if value >= 0.0 {
                RGBColor(
                    (255.0 * normalized) as u8,
                    (255.0 * (1.0 - normalized * 0.5)) as u8,
                    (255.0 * (1.0 - normalized)) as u8,
                )
            } else {
                RGBColor(
                    (255.0 * (1.0 - normalized)) as u8,
                    (255.0 * (1.0 - normalized * 0.5)) as u8,
                    (255.0 * normalized) as u8,
                )
            };

            chart.draw_series(std::iter::once(Rectangle::new(
                [(dim as f32, pos as f32), ((dim + 1) as f32, (pos + 1) as f32)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

/// Visualiza un patrón de atención específico con etiquetas de texto
///
/// Útil para ver cómo un modelo atiende a palabras específicas
///
/// # Argumentos
/// * `attention_weights` - Matriz de pesos de atención (seq_len, seq_len)
/// * `tokens` - Tokens de la secuencia
/// * `output_path` - Ruta donde guardar la imagen
/// * `title` - Título del gráfico
///
/// # Retorna
/// Result indicando éxito o error
pub fn plot_attention_with_tokens(
    attention_weights: &Array2<f32>,
    tokens: &[&str],
    output_path: &str,
    title: &str,
) -> Result<(), Box<dyn Error>> {
    let seq_len = attention_weights.shape()[0];
    assert_eq!(
        seq_len,
        tokens.len(),
        "Número de tokens debe coincidir con seq_len"
    );

    let root = BitMapBackend::new(output_path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d(0f32..seq_len as f32, 0f32..seq_len as f32)?;

    // Configurar mesh con etiquetas
    chart
        .configure_mesh()
        .x_desc("Keys (atendido)")
        .y_desc("Queries (atiende)")
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    // Encontrar min y max
    let min_val = attention_weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = attention_weights
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Dibujar heatmap
    for i in 0..seq_len {
        for j in 0..seq_len {
            let value = attention_weights[[i, j]];

            let normalized = if max_val > min_val {
                (value - min_val) / (max_val - min_val)
            } else {
                0.5
            };

            let color = RGBColor(
                (255.0 * normalized) as u8,
                (100.0 * (1.0 - normalized)) as u8,
                (200.0 * (1.0 - normalized)) as u8,
            );

            chart.draw_series(std::iter::once(Rectangle::new(
                [(j as f32, i as f32), ((j + 1) as f32, (i + 1) as f32)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    println!("Heatmap guardado en: {}", output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_plot_attention_heatmap() {
        let weights = array![
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8]
        ];

        // Solo verificar que no haya errores
        let result = plot_attention_heatmap(
            &weights,
            "test_attention.png",
            "Test Attention",
            None,
        );

        assert!(result.is_ok());
    }
}
