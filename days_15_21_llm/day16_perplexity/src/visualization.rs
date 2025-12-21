// Módulo de visualización con plotters
// Genera gráficas de comparación de perplexity

use plotters::prelude::*;

/// Grafica una comparación de perplexity entre diferentes modelos usando barras
///
/// # Argumentos
/// * `model_names` - Nombres de los modelos
/// * `perplexities` - Valores de perplexity correspondientes
/// * `output_path` - Ruta donde guardar la imagen
pub fn plot_perplexity_comparison(
    model_names: &[&str],
    perplexities: &[f64],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Crear área de dibujo
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Encontrar el máximo para escalar el eje Y
    let max_ppl = perplexities
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_max = (max_ppl * 1.2).ceil();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Comparación de Perplexity entre Modelos N-gram",
            ("sans-serif", 30).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f32..model_names.len() as f32,
            0f64..y_max,
        )?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("Perplexity (menor es mejor)")
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()?;

    // Dibujar barras
    let bar_width = 0.6;
    let colors = [&RED, &BLUE, &GREEN];

    for (i, (name, &ppl)) in model_names.iter().zip(perplexities.iter()).enumerate() {
        let color = colors[i % colors.len()];

        // Dibujar barra
        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (i as f32, 0.0),
                ((i as f32) + bar_width, ppl),
            ],
            color.filled(),
        )))?;

        // Etiqueta del modelo
        chart.draw_series(std::iter::once(Text::new(
            name.to_string(),
            (i as f32 + bar_width / 2.0, -y_max * 0.05),
            ("sans-serif", 15).into_font(),
        )))?;

        // Valor de perplexity sobre la barra
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.1}", ppl),
            (i as f32 + bar_width / 2.0, ppl + y_max * 0.02),
            ("sans-serif", 14).into_font(),
        )))?;
    }

    root.present()?;
    Ok(())
}

/// Grafica curvas de aprendizaje (perplexity vs cantidad de datos de entrenamiento)
///
/// # Argumentos
/// * `learning_curves` - Vector de tuplas (nombre_modelo, puntos) donde puntos es Vec<(fracción, perplexity)>
/// * `output_path` - Ruta donde guardar la imagen
pub fn plot_learning_curves(
    learning_curves: &[(&str, Vec<(f64, f64)>)],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Crear área de dibujo
    let root = BitMapBackend::new(output_path, (1000, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    // Encontrar rangos
    let max_ppl = learning_curves
        .iter()
        .flat_map(|(_, points)| points.iter().map(|(_, ppl)| ppl))
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let min_ppl = learning_curves
        .iter()
        .flat_map(|(_, points)| points.iter().map(|(_, ppl)| ppl))
        .copied()
        .fold(f64::INFINITY, f64::min);

    let y_max = (max_ppl * 1.1).ceil();
    let y_min = (min_ppl * 0.9).floor().max(0.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Curvas de Aprendizaje: Perplexity vs Cantidad de Datos",
            ("sans-serif", 35).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0f64..1.0f64, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Fracción de datos de entrenamiento")
        .y_desc("Perplexity (menor es mejor)")
        .x_label_formatter(&|x| format!("{:.0}%", x * 100.0))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()?;

    // Colores para cada modelo
    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];

    // Dibujar cada curva
    for (idx, (name, points)) in learning_curves.iter().enumerate() {
        let color = colors[idx % colors.len()];

        // Línea
        chart
            .draw_series(LineSeries::new(
                points.iter().map(|(x, y)| (*x, *y)),
                color.stroke_width(3),
            ))?
            .label(*name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3))
            });

        // Puntos
        chart.draw_series(points.iter().map(|(x, y)| {
            Circle::new((*x, *y), 5, color.filled())
        }))?;
    }

    // Leyenda
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 18))
        .draw()?;

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_perplexity_comparison() {
        let names = vec!["Unigram", "Bigram", "Trigram"];
        let ppls = vec![300.0, 150.0, 80.0];

        let result = plot_perplexity_comparison(&names, &ppls, "test_comparison.png");
        assert!(result.is_ok());

        // Limpiar
        std::fs::remove_file("test_comparison.png").ok();
    }

    #[test]
    fn test_plot_learning_curves() {
        let curves = vec![
            ("Model A", vec![(0.2, 500.0), (0.4, 300.0), (0.8, 150.0), (1.0, 100.0)]),
            ("Model B", vec![(0.2, 400.0), (0.4, 250.0), (0.8, 120.0), (1.0, 80.0)]),
        ];

        let result = plot_learning_curves(&curves, "test_learning.png");
        assert!(result.is_ok());

        // Limpiar
        std::fs::remove_file("test_learning.png").ok();
    }
}
