

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use plotters::prelude::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use image::{ImageReader, Luma};
mod gpu_backend; // declara el módulo (si gpu.rs está en la misma carpeta de src/)

use crate::gpu_backend::GpuBackend;

/// Red neuronal completa
pub struct NeuralNetwork {
    /// Capas de la red
    pub layers: Vec<Layer>,
    /// Optimizador
    pub optimizer: AdamOptimizer,
}

impl NeuralNetwork {
    /// Crea una nueva red neuronal con la arquitectura especificada
    pub fn new(input_size: usize) -> Self {
        let mut layers = Vec::new();
        
        // Capa oculta 1: 784 -> 128
        layers.push(Layer::new(input_size, 32));
        // Capa oculta 2: 128 -> 64
        layers.push(Layer::new(32, 10));
        
        let optimizer = AdamOptimizer::new(&layers, 0.001);

        Self { layers, optimizer }
    }

    pub async fn forward_gpu(&mut self, inputs: &[f64], backend: &GpuBackend) -> Vec<f64> {
        let mut current_input = inputs.to_vec();
        
        // Capas ocultas con ReLU
        for i in 0..self.layers.len() - 1 {
            let z = self.layers[i].forward_gpu(&current_input, backend).await;
            self.layers[i].z_values = z;
            self.layers[i].relu();
            current_input = self.layers[i].activations.clone();
        }
        
        // Capa de salida con Softmax
        let last_idx = self.layers.len() - 1;
        let z = self.layers[last_idx].forward_gpu(&current_input, backend).await;
        self.layers[last_idx].z_values = z;
        self.layers[last_idx].softmax();
        
        self.layers[last_idx].activations.clone()
    }
    /// Forward pass de toda la red
    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut current_input = inputs.to_vec();
        
        // Capas ocultas con ReLU
        for i in 0..self.layers.len() - 1 {
            let z = self.layers[i].forward(&current_input);
            self.layers[i].z_values = z;
            self.layers[i].relu();
            current_input = self.layers[i].activations.clone();
        }
        
        // Capa de salida con Softmax
        let last_idx = self.layers.len() - 1;
        let z = self.layers[last_idx].forward(&current_input);
        self.layers[last_idx].z_values = z;
        self.layers[last_idx].softmax();
        
        self.layers[last_idx].activations.clone()
    }

    /// Calcula la pérdida de entropía cruzada
    pub fn cross_entropy_loss(&self, predictions: &[f64], target: usize) -> f64 {
        let epsilon = 1e-15;
        let clamped_pred = predictions[target].max(epsilon).min(1.0 - epsilon);
        -clamped_pred.ln()
    }

    /// Backpropagation
    pub fn backward(&mut self, inputs: &[f64], target: usize) {
        let num_layers = self.layers.len();
        
        // Limpiar gradientes
        for layer in &mut self.layers {
            for i in 0..layer.weight_gradients.len() {
                for j in 0..layer.weight_gradients[i].len() {
                    layer.weight_gradients[i][j] = 0.0;
                }
            }
            for i in 0..layer.bias_gradients.len() {
                layer.bias_gradients[i] = 0.0;
            }
        }

        // Error de la capa de salida (derivada de softmax + cross entropy)
        let mut delta = vec![0.0; self.layers[num_layers - 1].activations.len()];
        for i in 0..delta.len() {
            delta[i] = self.layers[num_layers - 1].activations[i];
            if i == target {
                delta[i] -= 1.0;
            }
        }

        // Backpropagate desde la última capa
        for layer_idx in (0..num_layers).rev() {
            // Obtener entradas de la capa anterior
            let layer_inputs = if layer_idx == 0 {
                inputs.to_vec()
            } else {
                self.layers[layer_idx - 1].activations.clone()
            };

            // Calcular gradientes de pesos y sesgos
            for i in 0..self.layers[layer_idx].weights.len() {
                self.layers[layer_idx].bias_gradients[i] = delta[i];
                for j in 0..self.layers[layer_idx].weights[i].len() {
                    self.layers[layer_idx].weight_gradients[i][j] = delta[i] * layer_inputs[j];
                }
            }

            // Calcular delta para la capa anterior (si no es la primera)
            if layer_idx > 0 {
                let mut next_delta = vec![0.0; layer_inputs.len()];
                for j in 0..next_delta.len() {
                    for i in 0..self.layers[layer_idx].weights.len() {
                        next_delta[j] += delta[i] * self.layers[layer_idx].weights[i][j];
                    }
                    // Aplicar derivada de ReLU
                    if self.layers[layer_idx - 1].z_values[j] <= 0.0 {
                        next_delta[j] = 0.0;
                    }
                }
                delta = next_delta;
            }
        }
    }

    /// Entrena la red neuronal
    pub async fn train(&mut self, train_data: &[(Vec<f64>, usize)], val_data: &[(Vec<f64>, usize)], epochs: usize, batch_size: usize, backend: &GpuBackend) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut train_losses = Vec::new();
        let mut val_losses = Vec::new();
        let mut val_accuracies = Vec::new();

        for epoch in 0..epochs {
            // Mezclar datos de entrenamiento
            let mut shuffled_data = train_data.to_vec();
            shuffled_data.shuffle(&mut thread_rng());

            let mut epoch_loss = 0.0;
            let mut processed_samples = 0;

            // Entrenamiento por batches
            for batch in shuffled_data.chunks(batch_size) {
                let mut batch_loss = 0.0;
                
                // Acumular gradientes del batch
                for (sample_input, target) in batch {
                    let prediction = self.forward_gpu(sample_input, backend).await;
                    let loss = self.cross_entropy_loss(&prediction, *target);
                    batch_loss += loss;
                    
                    self.backward(sample_input, *target);
                }
                
                // Actualizar pesos
                self.optimizer.update(&mut self.layers);
                
                epoch_loss += batch_loss;
                processed_samples += batch.len();
            }

            let avg_train_loss = epoch_loss / processed_samples as f64;
            train_losses.push(avg_train_loss);

            // Validación
            let (val_loss, val_accuracy) = self.validate(val_data).await;
            val_losses.push(val_loss);
            val_accuracies.push(val_accuracy);

            println!("Época {}/{}: Train Loss: {:.4}, Val Loss: {:.4}, Val Accuracy: {:.2}%", 
                     epoch + 1, epochs, avg_train_loss, val_loss, val_accuracy * 100.0);
        }

        (train_losses, val_losses, val_accuracies)
    }

    /// Valida el modelo
    pub async fn validate(&mut self, val_data: &[(Vec<f64>, usize)]) -> (f64, f64) {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (input, target) in val_data {
            let prediction = self.forward(input);
            let loss = self.cross_entropy_loss(&prediction, *target);
            total_loss += loss;

            let predicted_class = prediction
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;

            if predicted_class == *target {
                correct += 1;
            }
        }

        let avg_loss = total_loss / val_data.len() as f64;
        let accuracy = correct as f64 / val_data.len() as f64;

        (avg_loss, accuracy)
    }

    pub fn new_custom(input_size: usize, hidden_sizes: &[usize], learning_rate: f64) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = input_size;
        
        // Crear capas ocultas
        for &hidden_size in hidden_sizes {
            layers.push(Layer::new(prev_size, hidden_size));
            prev_size = hidden_size;
        }
        
        // Capa de salida (10 clases)
        layers.push(Layer::new(prev_size, 10));
        
        let optimizer = AdamOptimizer::new(&layers, learning_rate);

        Self { layers, optimizer }
    }
}
#[derive(Clone, Debug)]
pub struct Layer {
    /// Matriz de pesos (filas = neuronas salida, columnas = neuronas entrada)
    pub weights: Vec<Vec<f64>>,
    /// Vector de sesgos
    pub biases: Vec<f64>,
    /// Activaciones de la capa (salida después de función de activación)
    pub activations: Vec<f64>,
    /// Valores pre-activación (entrada a función de activación)
    pub z_values: Vec<f64>,
    /// Gradientes para pesos
    pub weight_gradients: Vec<Vec<f64>>,
    /// Gradientes para sesgos
    pub bias_gradients: Vec<f64>,
}

impl Layer {
    /// Crea una nueva capa con inicialización Xavier/Glorot
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = thread_rng();
        let xavier_std = (2.0 / (input_size + output_size) as f64).sqrt();
        
        let weights = (0..output_size)
            .map(|_|
                (0..input_size)
                    .map(|_| rng.r#gen::<f64>() * xavier_std - xavier_std / 2.0)
                    .collect()
            )
            .collect();
        
        let biases = vec![0.0; output_size];
        let activations = vec![0.0; output_size];
        let z_values = vec![0.0; output_size];
        let weight_gradients = vec![vec![0.0; input_size]; output_size];
        let bias_gradients = vec![0.0; output_size];

        Self {
            weights,
            biases,
            activations,
            z_values,
            weight_gradients,
            bias_gradients,
        }
    }

    /// Forward pass de la capa
    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        for i in 0..self.weights.len() {
            let mut sum = self.biases[i];
            for j in 0..inputs.len() {
                sum += self.weights[i][j] * inputs[j];
            }
            self.z_values[i] = sum;
        }
        self.z_values.clone()
    }

    pub async fn forward_gpu(&mut self, inputs: &[f64], backend: &GpuBackend) -> Vec<f64> {
        let m = self.weights.len() as u32;
        let k = inputs.len() as u32;
        let n = 1;

        let a: Vec<f32> = self.weights.iter().flat_map(|row| row.iter().map(|&x| x as f32)).collect();
        let b: Vec<f32> = inputs.iter().map(|&x| x as f32).collect();

        let result = backend.matmul(&a, &b, m, n, k).await;

        for i in 0..m as usize {
            self.z_values[i] = result[i] as f64 + self.biases[i];
        }

        self.z_values.clone()
    }

    /// Aplica función de activación ReLU
    pub fn relu(&mut self) {
        for i in 0..self.z_values.len() {
            self.activations[i] = if self.z_values[i] > 0.0 {
                self.z_values[i]
            } else {
                0.0
            };
        }
    }

    /// Aplica función de activación Softmax
    pub fn softmax(&mut self) {
        let max_val = self.z_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = self.z_values.iter().map(|&x| (x - max_val).exp()).sum();
        
        for i in 0..self.z_values.len() {
            self.activations[i] = (self.z_values[i] - max_val).exp() / exp_sum;
        }
    }
}

pub struct AdamOptimizer {
    /// Tasa de aprendizaje
    pub learning_rate: f64,
    /// Beta1 para momentum
    pub beta1: f64,
    /// Beta2 para momentum de segundo orden
    pub beta2: f64,
    /// Epsilon para estabilidad numérica
    pub epsilon: f64,
    /// Momento de primer orden para pesos
    pub m_weights: Vec<Vec<Vec<f64>>>,
    /// Momento de segundo orden para pesos
    pub v_weights: Vec<Vec<Vec<f64>>>,
    /// Momento de primer orden para sesgos
    pub m_biases: Vec<Vec<f64>>,
    /// Momento de segundo orden para sesgos
    pub v_biases: Vec<Vec<f64>>,
    /// Contador de pasos
    pub t: i32,
}

impl AdamOptimizer {
    /// Crea un nuevo optimizador Adam
    pub fn new(layers: &[Layer], learning_rate: f64) -> Self {
        let m_weights: Vec<Vec<Vec<f64>>> = layers
            .iter()
            .map(|layer| vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()])
            .collect();
        
        let v_weights: Vec<Vec<Vec<f64>>> = layers
            .iter()
            .map(|layer| vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()])
            .collect();
        
        let m_biases: Vec<Vec<f64>> = layers
            .iter()
            .map(|layer| vec![0.0; layer.biases.len()])
            .collect();
        
        let v_biases: Vec<Vec<f64>> = layers
            .iter()
            .map(|layer| vec![0.0; layer.biases.len()])
            .collect();

        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
            t: 0,
        }
    }

    /// Actualiza los parámetros usando Adam
    pub fn update(&mut self, layers: &mut [Layer]) {
        self.t += 1;
        let lr_t = self.learning_rate * (1.0 - self.beta2.powi(self.t)).sqrt() / (1.0 - self.beta1.powi(self.t));

        for (layer_idx, layer) in layers.iter_mut().enumerate() {
            // Actualizar pesos
            for i in 0..layer.weights.len() {
                for j in 0..layer.weights[i].len() {
                    let grad = layer.weight_gradients[i][j];
                    
                    self.m_weights[layer_idx][i][j] = self.beta1 * self.m_weights[layer_idx][i][j] + (1.0 - self.beta1) * grad;
                    self.v_weights[layer_idx][i][j] = self.beta2 * self.v_weights[layer_idx][i][j] + (1.0 - self.beta2) * grad * grad;
                    
                    layer.weights[i][j] -= lr_t * self.m_weights[layer_idx][i][j] / (self.v_weights[layer_idx][i][j].sqrt() + self.epsilon);
                }
            }
            
            // Actualizar sesgos
            for i in 0..layer.biases.len() {
                let grad = layer.bias_gradients[i];
                
                self.m_biases[layer_idx][i] = self.beta1 * self.m_biases[layer_idx][i] + (1.0 - self.beta1) * grad;
                self.v_biases[layer_idx][i] = self.beta2 * self.v_biases[layer_idx][i] + (1.0 - self.beta2) * grad * grad;
                
                layer.biases[i] -= lr_t * self.m_biases[layer_idx][i] / (self.v_biases[layer_idx][i].sqrt() + self.epsilon);
            }
        }
    }
}

/// Función para graficar curvas de entrenamiento y métricas
fn plot_training_metrics(train_losses: &[f64], val_losses: &[f64], val_accuracies: &[f64], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let areas = root.split_evenly((2, 1));
    let upper = &areas[0];
    let lower = &areas[1];
    
    // Gráfico superior: Pérdidas
    {
        let max_epoch = train_losses.len();
        let max_loss = train_losses
            .iter()
            .chain(val_losses.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(upper)
            .caption("Curvas de Pérdida", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0..max_epoch, 0f64..max_loss)?;

        chart.configure_mesh()
            .x_desc("Épocas")
            .y_desc("Pérdida")
            .draw()?;

        chart.draw_series(LineSeries::new(
            train_losses.iter().enumerate().map(|(i, &loss)| (i, loss)),
            &RED,
        ))?.label("Entrenamiento").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.draw_series(LineSeries::new(
            val_losses.iter().enumerate().map(|(i, &loss)| (i, loss)),
            &BLUE,
        ))?.label("Validación").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart.configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE)
            .draw()?;
    }
    
    // Gráfico inferior: Accuracy
    {
        let max_epoch = val_accuracies.len();

        let mut chart = ChartBuilder::on(lower)
            .caption("Accuracy de Validación", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0..max_epoch, 0f64..1f64)?;

        chart.configure_mesh()
            .x_desc("Épocas")
            .y_desc("Accuracy")
            .draw()?;

        chart.draw_series(LineSeries::new(
            val_accuracies.iter().enumerate().map(|(i, &acc)| (i, acc)),
            &GREEN,
        ))?.label("Accuracy").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        chart.configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE)
            .draw()?;
    }

    println!("Gráfica guardada en: {}", filename);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 === Red Neuronal MNIST Optimizada para Datasets Pequeños ===\n");

    // Initialize GPU backend
    println!("⚙️  Initializing GPU backend...");
    let gpu_backend = GpuBackend::new().await.ok_or("Failed to initialize GPU backend")?;
    println!("✅ GPU backend initialized successfully.");

    // Cargar datos con normalización mejorada
    let (train_data, val_data) = load_mnist_from_png_improved()?;
    
    // Si solo tienes 4 imágenes, mostrar visualización
    if train_data.len() + val_data.len() <= 10 {
        println!("🖼️  Visualizando dataset pequeño:");
        
        for (i, (pixels, label)) in train_data.iter().enumerate() {
            println!("📍 Imagen de entrenamiento {} (etiqueta: {})", i + 1, label);
            visualize_processed_image(pixels, *label, &NormalizationMethod::Centered);
        }
        
        for (i, (pixels, label)) in val_data.iter().enumerate() {
            println!("📍 Imagen de validación {} (etiqueta: {})", i + 1, label);
            visualize_processed_image(pixels, *label, &NormalizationMethod::Centered);
        }
    }
    
    // Configuración optimizada para datasets pequeños
    let config = SmallDatasetConfig::new(train_data.len(), val_data.len());
    config.print_config();
    
    // Crear red neuronal con configuración adaptada
    println!("🧠 Creando red neuronal...");
    let mut network = NeuralNetwork::new_custom(
        784, 
        &config.hidden_layers, 
        config.learning_rate
    );
    
    println!("🎯 Iniciando entrenamiento optimizado...");
    
    // Entrenar con configuración adaptada
    let (train_losses, val_losses, val_accuracies) = network.train(
        &train_data, 
        &val_data, 
        config.epochs, 
        config.batch_size,
        &gpu_backend
    ).await;
    
    // Evaluación final detallada
    println!("\n🏆 === Evaluación Final ===");
    evaluate_small_dataset(&mut network, &train_data, &val_data);
    
    // Análisis de convergencia
    analyze_training_convergence(&train_losses, &val_losses, &val_accuracies);
    
    // Generar gráfica si hay suficientes datos
    if train_losses.len() > 1 {
        match plot_training_metrics(&train_losses, &val_losses, &val_accuracies, "mnist_small_dataset_metrics.png") {
            Ok(_) => println!("📈 Gráfica guardada exitosamente"),
            Err(e) => println!("⚠️ Error generando gráfica: {}", e),
        }
    }
    
    // Prueba de predicción individual
    test_individual_predictions(&mut network, &val_data);
    
    Ok(())
}

struct SmallDatasetConfig {
    pub hidden_layers: Vec<usize>,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub dataset_size: usize,
}

impl SmallDatasetConfig {
    fn new(train_size: usize, val_size: usize) -> Self {
        let total_size = train_size + val_size;
        
        let (hidden_layers, learning_rate, epochs, batch_size) = if total_size <= 5 {
            // Dataset muy pequeño (≤5 muestras)
            (vec![16, 8], 0.001, 5000, 1)
        } else if total_size <= 20 {
            // Dataset pequeño (≤20 muestras)
            (vec![32, 16], 0.005, 3000, 1)
        } else if total_size <= 100 {
            // Dataset mediano-pequeño
            (vec![64, 32], 0.001, 1000, 2)
        } else {
            // Dataset normal
            (vec![128, 64], 0.001, 500, 4)
        };

        Self {
            hidden_layers,
            learning_rate,
            epochs,
            batch_size,
            dataset_size: total_size,
        }
    }
    
    fn print_config(&self) {
        println!("⚙️  Configuración para dataset de {} muestras:", self.dataset_size);
        println!("  🏗️  Arquitectura: 784 -> {} -> 10", 
                 self.hidden_layers.iter()
                     .map(|x| x.to_string())
                     .collect::<Vec<_>>()
                     .join(" -> "));
        println!("  📚 Tasa de aprendizaje: {}", self.learning_rate);
        println!("  🔄 Épocas: {}", self.epochs);
        println!("  📦 Tamaño de batch: {}", self.batch_size);
        println!();
    }
}

/// Evaluación detallada para datasets pequeños
fn evaluate_small_dataset(
    network: &mut NeuralNetwork, 
    train_data: &[(Vec<f64>, usize)], 
    val_data: &[(Vec<f64>, usize)]
) {
    println!("🔍 Evaluación detallada del dataset pequeño:\n");
    
    // Evaluar en datos de entrenamiento
    println!("📊 Rendimiento en entrenamiento:");
    let train_metrics = calculate_detailed_metrics(network, train_data);
    print_detailed_metrics(&train_metrics, "Entrenamiento");
    
    // Evaluar en datos de validación si existen
    if !val_data.is_empty() {
        println!("\n📊 Rendimiento en validación:");
        let val_metrics = calculate_detailed_metrics(network, val_data);
        print_detailed_metrics(&val_metrics, "Validación");
    } else {
        println!("⚠️ No hay datos de validación separados");
    }
}

/// Métricas detalladas para datasets pequeños
#[derive(Debug)]
struct DetailedMetrics {
    accuracy: f64,
    per_class_accuracy: HashMap<usize, f64>,
    confusion_matrix: Vec<Vec<usize>>,
    individual_predictions: Vec<(usize, usize, Vec<f64>)>, // (actual, predicted, probabilities)
}

fn calculate_detailed_metrics(
    network: &mut NeuralNetwork, 
    data: &[(Vec<f64>, usize)]
) -> DetailedMetrics {
    let mut confusion_matrix = vec![vec![0; 10]; 10];
    let mut individual_predictions = Vec::new();
    let mut correct_total = 0;
    
    for (input, actual) in data {
        let probabilities = network.forward(input);
        let predicted = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        
        confusion_matrix[*actual][predicted] += 1;
        individual_predictions.push((*actual, predicted, probabilities));
        
        if *actual == predicted {
            correct_total += 1;
        }
    }
    
    let accuracy = correct_total as f64 / data.len() as f64;
    
    // Calcular accuracy por clase
    let mut per_class_accuracy = HashMap::new();
    for class in 0..10 {
        let class_total: usize = confusion_matrix[class].iter().sum();
        if class_total > 0 {
            let class_correct = confusion_matrix[class][class];
            per_class_accuracy.insert(class, class_correct as f64 / class_total as f64);
        }
    }
    
    DetailedMetrics {
        accuracy,
        per_class_accuracy,
        confusion_matrix,
        individual_predictions,
    }
}

fn print_detailed_metrics(metrics: &DetailedMetrics, dataset_name: &str) {
    println!("  📈 Accuracy total de {}: {:.2}%", dataset_name, metrics.accuracy * 100.0);
    
    // Mostrar predicciones individuales
    println!("  🎯 Predicciones individuales:");
    for (i, (actual, predicted, probabilities)) in metrics.individual_predictions.iter().enumerate() {
        let confidence = probabilities[*predicted] * 100.0;
        let status = if actual == predicted { "✅" } else { "❌" };
        
        println!("    Muestra {}: {} Actual={}, Predicho={} (confianza: {:.1}%)", 
                 i + 1, status, actual, predicted, confidence);
        
        // Mostrar top 3 predicciones si hay error
        if actual != predicted {
            let mut sorted_probs: Vec<_> = probabilities.iter().enumerate().collect();
            sorted_probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            
            println!("      Top 3: ");
            for (rank, &(class, &prob)) in sorted_probs.iter().take(3).enumerate() {
                println!("        {}. Clase {}: {:.1}%", rank + 1, class, prob * 100.0);
            }
        }
    }
    
    // Mostrar accuracy por clase si hay múltiples clases
    if metrics.per_class_accuracy.len() > 1 {
        println!("  📊 Accuracy por clase:");
        for (class, &acc) in &metrics.per_class_accuracy {
            println!("    Clase {}: {:.2}%", class, acc * 100.0);
        }
    }
}

/// Análisis de convergencia del entrenamiento
fn analyze_training_convergence(
    train_losses: &[f64], 
    val_losses: &[f64], 
    val_accuracies: &[f64]
) {
    println!("\n📈 Análisis de Convergencia:");
    
    if train_losses.len() < 2 {
        println!("⚠️ Datos insuficientes para análisis de convergencia");
        return;
    }
    
    // Analizar tendencia de loss de entrenamiento
    let initial_train_loss = train_losses[0];
    let final_train_loss = *train_losses.last().unwrap();
    let train_improvement = (initial_train_loss - final_train_loss) / initial_train_loss * 100.0;
    
    println!("  🏋️  Loss de entrenamiento: {:.4} -> {:.4} (mejora: {:.1}%)", 
             initial_train_loss, final_train_loss, train_improvement);
    
    if !val_losses.is_empty() {
        let initial_val_loss = val_losses[0];
        let final_val_loss = *val_losses.last().unwrap();
        let val_improvement = (initial_val_loss - final_val_loss) / initial_val_loss * 100.0;
        
        println!("  🧪 Loss de validación: {:.4} -> {:.4} (mejora: {:.1}%)", 
                 initial_val_loss, final_val_loss, val_improvement);
    }
    
    if !val_accuracies.is_empty() {
        let initial_acc = val_accuracies[0] * 100.0;
        let final_acc = val_accuracies.last().unwrap() * 100.0;
        let best_acc = val_accuracies.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 100.0;
        
        println!("  🎯 Accuracy: {:.1}% -> {:.1}% (mejor: {:.1}%)", 
                 initial_acc, final_acc, best_acc);
    }
    
    // Detectar overfitting
    if !val_losses.is_empty() && train_losses.len() > 10 {
        let recent_train = &train_losses[train_losses.len()-5..];
        let recent_val = &val_losses[val_losses.len()-5..];
        
        let train_trend = recent_train.last().unwrap() - recent_train[0];
        let val_trend = recent_val.last().unwrap() - recent_val[0];
        
        if train_trend < -0.01 && val_trend > 0.01 {
            println!("  ⚠️  Posible overfitting detectado");
        } else if train_improvement < 5.0 {
            println!("  ⚠️  Convergencia lenta - considera ajustar hiperparámetros");
        }
    }
}

/// Prueba predicciones individuales detalladas
fn test_individual_predictions(network: &mut NeuralNetwork, data: &[(Vec<f64>, usize)]) {
    if data.is_empty() {
        return;
    }
    
    println!("\n🔬 Análisis Detallado de Predicciones:");
    
    for (i, (input, actual)) in data.iter().enumerate() {
        let probabilities = network.forward(input);
        let predicted = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        
        let confidence = probabilities[predicted];
        let status = if *actual == predicted { "✅ CORRECTO" } else { "❌ ERROR" };
        
        println!("🔍 Imagen {}: {} (Esperado: {}, Predicho: {}, Confianza: {:.1}%)",
                 i + 1, status, actual, predicted, confidence * 100.0);
        
        // Mostrar distribución de probabilidades
        println!("  📊 Distribución de probabilidades:");
        for (class, &prob) in probabilities.iter().enumerate() {
            let bar_length = (prob * 20.0) as usize;
            let bar = "█".repeat(bar_length);
            println!("    Clase {}: {:.3} {}", class, prob, bar);
        }
        println!();
    }
}


/// Estructura mejorada para manejar la carga y normalización de imágenes
pub struct ImageProcessor {
    /// Parámetros de normalización
    pub normalize_method: NormalizationMethod,
    pub target_mean: f64,
    pub target_std: f64,
}

#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    /// Simple: (pixel / 255.0) - para valores entre 0 y 1
    Simple,
    /// Standardization: (pixel - mean) / std - centrado en 0
    Standardization,
    /// MinMax con rango personalizado
    MinMax(f64, f64),
    /// Centrado en 0.5 (útil para redes neuronales)
    Centered,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self {
            normalize_method: NormalizationMethod::Centered,
            target_mean: 0.0,
            target_std: 1.0,
        }
    }
}

impl ImageProcessor {
    pub fn new(method: NormalizationMethod) -> Self {
        Self {
            normalize_method: method,
            target_mean: 0.0,
            target_std: 1.0,
        }
    }

    /// Procesa una imagen individual con normalización mejorada
    pub fn process_image(&self, path: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        println!("  📸 Procesando: {}", path);
        
        // Cargar imagen
        let img = ImageReader::open(path)?
            .decode()?;
        let img = img.into_luma8();
        
        let (width, height) = img.dimensions();
        
        if width != 28 || height != 28 {
            return Err(format!("❌ Imagen {} tiene dimensiones {}x{}, esperadas 28x28", 
                             path, width, height).into());
        }
        
        // Extraer píxeles raw
        let raw_pixels: Vec<u8> = img.pixels().map(|pixel| pixel[0]).collect();
        
        // Aplicar normalización específica
        let normalized_pixels = match &self.normalize_method {
            NormalizationMethod::Simple => self.normalize_simple(&raw_pixels),
            NormalizationMethod::Standardization => self.normalize_standardization(&raw_pixels),
            NormalizationMethod::MinMax(min, max) => self.normalize_minmax(&raw_pixels, *min, *max),
            NormalizationMethod::Centered => self.normalize_centered(&raw_pixels),
        };
        
        // Mostrar estadísticas de la imagen
        self.print_image_stats(&normalized_pixels, path);
        
        Ok(normalized_pixels)
    }

    /// Normalización simple: 0-255 -> 0-1, invirtiendo colores para MNIST
    fn normalize_simple(&self, pixels: &[u8]) -> Vec<f64> {
        pixels.iter()
            .map(|&pixel| {
                // Invertir colores: PNG negro=0,blanco=255 -> MNIST blanco=0,negro=1
                let inverted = 255 - pixel;
                inverted as f64 / 255.0
            })
            .collect()
    }

    /// Normalización centrada: 0-255 -> -0.5 a 0.5, invirtiendo colores
    fn normalize_centered(&self, pixels: &[u8]) -> Vec<f64> {
        pixels.iter()
            .map(|&pixel| {
                let inverted = 255 - pixel;
                (inverted as f64 / 255.0) - 0.5
            })
            .collect()
    }

    /// Normalización con estandarización: media=0, std=1
    fn normalize_standardization(&self, pixels: &[u8]) -> Vec<f64> {
        // Primero invertir y convertir a f64
        let inverted: Vec<f64> = pixels.iter()
            .map(|&pixel| (255 - pixel) as f64)
            .collect();
        
        // Calcular media y desviación estándar
        let mean = inverted.iter().sum::<f64>() / inverted.len() as f64;
        let variance = inverted.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / inverted.len() as f64;
        let std = variance.sqrt();
        
        // Evitar división por cero
        let std = if std < 1e-8 { 1.0 } else { std };
        
        // Aplicar estandarización
        inverted.iter()
            .map(|&x| (x - mean) / std)
            .collect()
    }

    /// Normalización Min-Max con rango personalizado
    fn normalize_minmax(&self, pixels: &[u8], target_min: f64, target_max: f64) -> Vec<f64> {
        let inverted: Vec<f64> = pixels.iter()
            .map(|&pixel| (255 - pixel) as f64)
            .collect();
        
        let min_val = inverted.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = inverted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let range = if (max_val - min_val).abs() < 1e-8 { 1.0 } else { max_val - min_val };
        
        inverted.iter()
            .map(|&x| target_min + (x - min_val) / range * (target_max - target_min))
            .collect()
    }

    /// Muestra estadísticas de la imagen procesada
    fn print_image_stats(&self, pixels: &[f64], path: &str) {
        let mean = pixels.iter().sum::<f64>() / pixels.len() as f64;
        let variance = pixels.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / pixels.len() as f64;
        let std = variance.sqrt();
        let min_val = pixels.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = pixels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Contar píxeles "activos" (> umbral)
        let threshold = match &self.normalize_method {
            NormalizationMethod::Simple => 0.1,
            NormalizationMethod::Centered => -0.4,
            NormalizationMethod::Standardization => mean,
            NormalizationMethod::MinMax(_, _) => 0.1,
        };
        
        let active_pixels = pixels.iter().filter(|&&x| x > threshold).count();
        
        println!("    📊 Estadísticas de {}: Media={:.3}, Std={:.3}, Min={:.3}, Max={:.3}, Píxeles activos={}/784", 
                 path.split('/').last().unwrap_or(path), mean, std, min_val, max_val, active_pixels);
    }
}

/// Función mejorada para cargar MNIST desde PNG
pub fn load_mnist_from_png_improved() -> Result<(Vec<(Vec<f64>, usize)>, Vec<(Vec<f64>, usize)>), Box<dyn std::error::Error>> {
    println!("🚀 === Carga Mejorada de MNIST desde PNG === 🚀\n");

    // Crear procesador de imágenes con normalización centrada (mejor para redes neuronales)
    let processor = ImageProcessor::new(NormalizationMethod::Centered);
    
    // Leer content.json
    println!("📄 Leyendo content.json...");
    let content_file = File::open("content.json")?;
    let reader = BufReader::new(content_file);
    let content: ContentJson = serde_json::from_reader(reader)?;
    
    println!("📁 Encontradas {} imágenes en content.json\n", content.images.len());

    let mut all_data = Vec::new();
    
    // Procesar cada imagen con el procesador mejorado
    for image_info in &content.images {
        println!("🔄 Procesando: {} ({})", image_info.name, image_info.size);
        
        // Mapear nombre a número
        let label = digit_name_to_number(&image_info.name)
            .ok_or_else(|| format!("❌ No se pudo mapear '{}' a un dígito", image_info.name))?;
        
        // Construir ruta
        let image_path = format!("public/train/{}", image_info.name);
        
        // Verificar existencia
        if !std::path::Path::new(&image_path).exists() {
            return Err(format!("❌ Archivo no encontrado: {}", image_path).into());
        }
        
        // Procesar imagen con normalización mejorada
        let pixels = processor.process_image(&image_path)?;
        
        // Validar dimensiones
        if pixels.len() != 784 {
            return Err(format!("❌ Imagen {} tiene {} píxeles, esperados 784", 
                             image_info.name, pixels.len()).into());
        }
        
        all_data.push((pixels, label));
        println!("    ✅ Cargada como dígito {}\n", label);
    }
    
    println!("🎉 === Datos Cargados Exitosamente ===");
    println!("📈 Total de muestras: {}", all_data.len());
    
    // Análisis estadístico del dataset completo
    analyze_dataset_statistics(&all_data);
    
    // Mezclar y dividir datos
    let mut shuffled_data = all_data.clone();
    shuffled_data.shuffle(&mut thread_rng());
    
    // División 80/20 con mínimo de 1 muestra para validación
    let split_point = std::cmp::max(1, (shuffled_data.len() as f64 * 0.8) as usize);
    let split_point = std::cmp::min(split_point, shuffled_data.len() - 1);
    
    let train_data = shuffled_data[..split_point].to_vec();
    let val_data = shuffled_data[split_point..].to_vec();
    
    println!("\n📊 División Final:");
    println!("  🏋️  Entrenamiento: {} muestras ({:.1}%)", 
             train_data.len(), 
             train_data.len() as f64 / shuffled_data.len() as f64 * 100.0);
    println!("  🧪 Validación: {} muestras ({:.1}%)", 
             val_data.len(), 
             val_data.len() as f64 / shuffled_data.len() as f64 * 100.0);
    
    Ok((train_data, val_data))
}

/// Analiza estadísticas del dataset completo
fn analyze_dataset_statistics(data: &[(Vec<f64>, usize)]) {
    println!("\n📊 Análisis Estadístico del Dataset:");
    
    // Contar por clase
    let mut class_counts: HashMap<usize, usize> = HashMap::new();
    let mut all_pixels: Vec<f64> = Vec::new();
    
    for (pixels, label) in data {
        *class_counts.entry(*label).or_insert(0) += 1;
        all_pixels.extend(pixels.iter());
    }
    
    println!("  📋 Distribución por clase:");
    for i in 0..10 {
        let count = class_counts.get(&i).unwrap_or(&0);
        println!("    Dígito {}: {} muestras", i, count);
    }
    
    // Estadísticas globales de píxeles
    let mean = all_pixels.iter().sum::<f64>() / all_pixels.len() as f64;
    let variance = all_pixels.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all_pixels.len() as f64;
    let std = variance.sqrt();
    
    println!("  📈 Estadísticas globales de píxeles:");
    println!("    Media: {:.4}", mean);
    println!("    Desviación estándar: {:.4}", std);
    println!("    Rango: [{:.4}, {:.4}]", 
             all_pixels.iter().cloned().fold(f64::INFINITY, f64::min),
             all_pixels.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
}

/// Función para visualizar una imagen procesada (debugging)
pub fn visualize_processed_image(pixels: &[f64], label: usize, method: &NormalizationMethod) {
    println!("🖼️  Visualización - Etiqueta: {} | Método: {:?}", label, method);
    
    // Determinar umbrales según el método de normalización
    let (low_thresh, mid_thresh, high_thresh) = match method {
        NormalizationMethod::Simple => (0.1, 0.3, 0.6),
        NormalizationMethod::Centered => (-0.4, -0.2, 0.0),
        NormalizationMethod::Standardization => (-0.5, 0.0, 0.5),
        NormalizationMethod::MinMax(min, max) => {
            let range = max - min;
            (min + 0.1 * range, min + 0.3 * range, min + 0.6 * range)
        },
    };
    
    for row in 0..28 {
        print!("  ");
        for col in 0..28 {
            let pixel = pixels[row * 28 + col];
            if pixel > high_thresh {
                print!("██");
            } else if pixel > mid_thresh {
                print!("▓▓");
            } else if pixel > low_thresh {
                print!("░░");
            } else {
                print!("  ");
            }
        }
        println!();
    }
    println!();
}


#[derive(Debug, Deserialize, Serialize)]
struct ContentJson {
    images: Vec<ImageInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ImageInfo {
    name: String,
    size: String,
}

fn digit_name_to_number(name: &str) -> Option<usize> {

    let firs_part = name.split('/').next().unwrap_or(name);
    match firs_part.to_lowercase().as_str() {
        "zero" | "0" => Some(0),
        "one" | "1" => Some(1),
        "two" | "2" => Some(2),
        "three" | "3" => Some(3),
        "four" | "4" => Some(4),
        "five" | "5" => Some(5),
        "six" | "6" => Some(6),
        "seven" | "7" => Some(7),
        "eight" | "8" => Some(8),
        "nine" | "9" => Some(9),
        _ => None,
    }
}


