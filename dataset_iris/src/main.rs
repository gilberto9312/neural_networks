use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use plotters::prelude::*;

/// Estructura para manejar el dataset de Iris
#[derive(Debug)]
 struct IrisDataset {
     train_features: Vec<Vec<f64>>,
     train_labels: Vec<usize>,
     test_features: Vec<Vec<f64>>,
     test_labels: Vec<usize>,
}

/// Carga y procesa el dataset de Iris desde un archivo CSV
/// 
/// # Argumentos
/// * `path` - Ruta al archivo CSV del dataset Iris
/// 
/// # Retorna
/// * `Result<IrisDataset, Box<dyn Error>>` - Dataset procesado o error
/// 
/// # Funcionalidad
/// 1. Lee el CSV y parsea las columnas numéricas
/// 2. Convierte especies a valores numéricos (setosa=0, versicolor=1, virginica=2)
/// 3. Normaliza features usando min-max scaling (0.0 a 1.0)
/// 4. Divide datos en entrenamiento/prueba (80/20) aleatoriamente
 fn load_iris_dataset(path: &str) -> Result<IrisDataset, Box<dyn Error>> {
    // 1. Leer CSV y parsear datos
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    
    let mut raw_features: Vec<Vec<f64>> = Vec::new();
    let mut raw_labels: Vec<String> = Vec::new();
    
    // Leer cada fila del CSV
    for result in rdr.records() {
        let record = result?;
        
        // Parsear features numéricas (primeras 4 columnas)
        let features: Result<Vec<f64>, _> = record.iter()
            .take(4)
            .map(|s| s.parse::<f64>())
            .collect();
        
        match features {
            Ok(feat) => {
                raw_features.push(feat);
                // La última columna es la especie
                if let Some(species) = record.get(4) {
                    raw_labels.push(species.to_string());
                }
            }
            Err(e) => eprintln!("Error parseando fila: {}", e),
        }
    }
    
    if raw_features.is_empty() {
        return Err("No se pudieron leer datos del archivo CSV".into());
    }
    
    // 2. Convertir especies a valores numéricos
    let species_to_num: HashMap<String, usize> = [
        ("setosa".to_string(), 0),
        ("versicolor".to_string(), 1),
        ("virginica".to_string(), 2),
    ].iter().cloned().collect();
    
    let labels: Vec<usize> = raw_labels.iter()
        .filter_map(|species| species_to_num.get(species).copied())
        .collect();
    
    if labels.len() != raw_features.len() {
        return Err("Mismatch entre número de features y labels".into());
    }
    
    // 3. Normalización min-max (0.0 a 1.0)
    let num_features = raw_features[0].len();
    let mut min_vals = vec![f64::INFINITY; num_features];
    let mut max_vals = vec![f64::NEG_INFINITY; num_features];
    
    // Encontrar min y max de cada feature
    for sample in &raw_features {
        for (i, &value) in sample.iter().enumerate() {
            min_vals[i] = min_vals[i].min(value);
            max_vals[i] = max_vals[i].max(value);
        }
    }
    
    // Aplicar normalización min-max
    let normalized_features: Vec<Vec<f64>> = raw_features.iter()
        .map(|sample| {
            sample.iter().enumerate()
                .map(|(i, &value)| {
                    if max_vals[i] != min_vals[i] {
                        (value - min_vals[i]) / (max_vals[i] - min_vals[i])
                    } else {
                        0.0 // Si min == max, asignar 0
                    }
                })
                .collect()
        })
        .collect();
    
    // 4. Crear índices y mezclar aleatoriamente
    let mut indices: Vec<usize> = (0..normalized_features.len()).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);
    
    // 5. Dividir en entrenamiento (80%) y prueba (20%)
    let train_size = (normalized_features.len() as f64 * 0.8) as usize;
    
    let mut train_features = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_features = Vec::new();
    let mut test_labels = Vec::new();
    
    for (i, &idx) in indices.iter().enumerate() {
        if i < train_size {
            train_features.push(normalized_features[idx].clone());
            train_labels.push(labels[idx]);
        } else {
            test_features.push(normalized_features[idx].clone());
            test_labels.push(labels[idx]);
        }
    }
    
    Ok(IrisDataset {
        train_features,
        train_labels,
        test_features,
        test_labels,
    })
}

/// Función de activación sigmoid
/// Formula: σ(x) = 1 / (1 + e^(-x))
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivada de la función sigmoid
/// Formula: σ'(x) = σ(x) * (1 - σ(x))
fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// ========== FUNCIÓN ReLU ==========
/// Función de activación ReLU (Rectified Linear Unit)
/// Fórmula: ReLU(x) = max(0, x)
/// Rango: [0, +∞)
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

/// Derivada de ReLU
/// Fórmula: ReLU'(x) = 1 si x > 0, 0 si x ≤ 0
fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Función softmax para clasificación multiclase
/// Formula: softmax(x_i) = e^(x_i) / Σ(e^(x_j))
fn softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f64> = x.iter().map(|&val| (val - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&val| val / sum).collect()
}

/// Función de pérdida: Cross-Entropy Loss
/// Formula: L = -Σ(y_true * log(y_pred))
fn cross_entropy_loss(predicted: &[f64], actual: usize, num_classes: usize) -> f64 {
    let mut one_hot = vec![0.0; num_classes];
    one_hot[actual] = 1.0;
    
    let mut loss = 0.0;
    for i in 0..num_classes {
        loss -= one_hot[i] * predicted[i].max(1e-15).ln();
    }
    loss
}

/// Parámetros del optimizador Adam
#[derive(Clone)]
struct AdamOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Vec<Vec<f64>>, // Primer momento
    v: Vec<Vec<f64>>, // Segundo momento
    t: f64, // Contador de tiempo
}

impl AdamOptimizer {
    fn new(learning_rate: f64, shape: &[usize]) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        
        for i in 0..shape.len() - 1 {
            m.push(vec![0.0; shape[i] * shape[i + 1]]);
            v.push(vec![0.0; shape[i] * shape[i + 1]]);
        }
        
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m,
            v,
            t: 0.0,
        }
    }
    
    fn update(&mut self, weights: &mut [Vec<f64>], gradients: &[Vec<f64>]) {
        self.t += 1.0;
        
        for (layer_idx, (weight_layer, grad_layer)) in weights.iter_mut().zip(gradients.iter()).enumerate() {
            for (i, (weight, &grad)) in weight_layer.iter_mut().zip(grad_layer.iter()).enumerate() {
                // Actualizar momentos
                self.m[layer_idx][i] = self.beta1 * self.m[layer_idx][i] + (1.0 - self.beta1) * grad;
                self.v[layer_idx][i] = self.beta2 * self.v[layer_idx][i] + (1.0 - self.beta2) * grad * grad;
                
                // Corrección de bias
                let m_hat = self.m[layer_idx][i] / (1.0 - self.beta1.powf(self.t));
                let v_hat = self.v[layer_idx][i] / (1.0 - self.beta2.powf(self.t));
                
                // Actualizar peso
                *weight -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }
}

/// Red Neuronal para clasificación del dataset Iris
 struct IrisNeuralNetwork {
    weights: Vec<Vec<f64>>, // Pesos de cada capa
    biases: Vec<Vec<f64>>,  // Sesgos de cada capa
    layer_sizes: Vec<usize>, // Tamaño de cada capa
    optimizer: AdamOptimizer,
}

impl IrisNeuralNetwork {
    /// Crea una nueva red neuronal
    /// 
    /// # Arquitectura
    /// - Capa de entrada: 4 neuronas (features de Iris)
    /// - Capa oculta: 8 neuronas con activación sigmoid
    /// - Capa de salida: 3 neuronas con activación softmax (3 especies)
     fn new(learning_rate: f64) -> Self {
        let layer_sizes = vec![4, 16, 8, 3]; // 4 inputs -> 8 hidden -> 3 outputs
        let mut rng = rand::thread_rng();
        
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        // Inicializar pesos y sesgos para cada capa
        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];
            
            // Inicialización Xavier/Glorot
            let limit = (6.0 / (input_size + output_size) as f64).sqrt();
            
            let layer_weights: Vec<f64> = (0..input_size * output_size)
                .map(|_| rng.gen_range(-limit..limit))
                .collect();
                
            let layer_biases: Vec<f64> = vec![0.0; output_size];
            
            weights.push(layer_weights);
            biases.push(layer_biases);
        }
        
        let optimizer = AdamOptimizer::new(learning_rate, &layer_sizes);
        
        Self {
            weights,
            biases,
            layer_sizes,
            optimizer,
        }
    }
    
    /// Forward pass: propaga la entrada a través de la red
    fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut activations = vec![input.to_vec()];
        let mut current_input = input.to_vec();
        
        for (layer_idx, (weights, biases)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let input_size = self.layer_sizes[layer_idx];
            let output_size = self.layer_sizes[layer_idx + 1];
            
            let mut layer_output = vec![0.0; output_size];
            
            // Multiplicación matriz-vector: output = weights * input + bias
            for i in 0..output_size {
                for j in 0..input_size {
                    layer_output[i] += weights[i * input_size + j] * current_input[j];
                }
                layer_output[i] += biases[i];
            }
            
            // Aplicar función de activación
            if layer_idx == self.weights.len() - 1 {
                // Última capa: softmax
                layer_output = softmax(&layer_output);
            } else {
                // Capas ocultas: sigmoid
                layer_output = layer_output.iter().map(|&x| relu(x)).collect();
            }
            
            activations.push(layer_output.clone());
            current_input = layer_output;
        }
        
        (current_input, activations)
    }
    
    /// Backpropagation: calcula gradientes y actualiza pesos
    fn backward(&mut self, input: &[f64], target: usize, predicted: &[f64], activations: &[Vec<f64>]) {
        let mut gradients = vec![Vec::new(); self.weights.len()];
        let mut delta = vec![0.0; 3]; // Error en la capa de salida
        
        // Error en la capa de salida (cross-entropy + softmax)
        for i in 0..3 {
            delta[i] = predicted[i] - if i == target { 1.0 } else { 0.0 };
        }
        
        // Calcular gradientes para cada capa (hacia atrás)
        for layer_idx in (0..self.weights.len()).rev() {
            let input_size = self.layer_sizes[layer_idx];
            let output_size = self.layer_sizes[layer_idx + 1];
            
            let layer_input = &activations[layer_idx];
            let mut layer_gradients = vec![0.0; input_size * output_size];
            
            // Calcular gradientes de pesos
            for i in 0..output_size {
                for j in 0..input_size {
                    layer_gradients[i * input_size + j] = delta[i] * layer_input[j];
                }
            }
            
            gradients[layer_idx] = layer_gradients;
            
            // Calcular delta para la capa anterior (si no es la primera)
            if layer_idx > 0 {
                let mut next_delta = vec![0.0; input_size];
                
                for j in 0..input_size {
                    for i in 0..output_size {
                        next_delta[j] += delta[i] * self.weights[layer_idx][i * input_size + j];
                    }
                    // Aplicar derivada de sigmoid para capas ocultas
                    next_delta[j] *= relu_derivative(layer_input[j]);
                }
                
                delta = next_delta;
            }
        }
        
        // Actualizar pesos usando Adam
        self.optimizer.update(&mut self.weights, &gradients);
    }
    
    /// Entrena la red neuronal
    fn train(&mut self, features: &[Vec<f64>], labels: &[usize], val_features: &[Vec<f64>], val_labels: &[usize], epochs: usize) -> (Vec<f64>, Vec<f64>) {
        println!("Iniciando entrenamiento...");
        
        let mut train_losses = Vec::new();
        let mut val_losses = Vec::new();
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;
            
            for (input, &target) in features.iter().zip(labels.iter()) {
                // Forward pass
                let (predicted, activations) = self.forward(input);
                
                // Calcular pérdida
                let loss = cross_entropy_loss(&predicted, target, 3);
                total_loss += loss;
                
                // Verificar predicción
                let predicted_class = predicted.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap().0;
                    
                if predicted_class == target {
                    correct_predictions += 1;
                }
                
                // Backward pass
                self.backward(input, target, &predicted, &activations);
            }
            
            // Calcular pérdida promedio de entrenamiento
            let avg_train_loss = total_loss / features.len() as f64;
            train_losses.push(avg_train_loss);
            
            // Calcular pérdida de validación
            let mut val_loss = 0.0;
            for (input, &target) in val_features.iter().zip(val_labels.iter()) {
                let (predicted, _) = self.forward(input);
                val_loss += cross_entropy_loss(&predicted, target, 3);
            }
            let avg_val_loss = val_loss / val_features.len() as f64;
            val_losses.push(avg_val_loss);
            
            // Mostrar progreso cada 100 epochs
            if epoch % 100 == 0 {
                let accuracy = correct_predictions as f64 / features.len() as f64;
                println!("Epoch {}: Train Loss = {:.4}, Val Loss = {:.4}, Accuracy = {:.2}%", 
                        epoch, avg_train_loss, avg_val_loss, accuracy * 100.0);
            }
        }
        
        (train_losses, val_losses)
    }
    
    /// Evalúa la red neuronal en el conjunto de prueba
     fn evaluate(&self, features: &[Vec<f64>], labels: &[usize]) -> f64 {
        let mut correct_predictions = 0;
        
        for (input, &target) in features.iter().zip(labels.iter()) {
            let (predicted, _) = self.forward(input);
            let predicted_class = predicted.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap().0;
                
            if predicted_class == target {
                correct_predictions += 1;
            }
        }
        
        correct_predictions as f64 / features.len() as f64
    }
    
    /// Predice la clase de una muestra
     fn predict(&self, input: &[f64]) -> (usize, Vec<f64>) {
        let (predicted, _) = self.forward(input);
        let predicted_class = predicted.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
            
        (predicted_class, predicted)
    }
}

fn plot_training_curve(train_errors: &[f64], val_errors: &[f64], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_epoch = train_errors.len();
    let max_val = train_errors
        .iter()
        .chain(val_errors.iter())
        .cloned()
        .fold(f64::MIN, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Curva de Entrenamiento", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..max_epoch, 0f64..max_val)?;

    chart.configure_mesh()
        .x_desc("Épocas")
        .y_desc("Error")
        .draw()?;

    // Línea de entrenamiento
    chart.draw_series(LineSeries::new(
        train_errors.iter().enumerate().map(|(i, &e)| (i, e)),
        &RED,
    ))?.label("Entrenamiento").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Línea de validación
    chart.draw_series(LineSeries::new(
        val_errors.iter().enumerate().map(|(i, &e)| (i, e)),
        &BLUE,
    ))?.label("Validación").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE)
        .draw()?;

    Ok(())
}

// Función principal de ejemplo
fn main() -> Result<(), Box<dyn Error>> {
    println!("🌸 Red Neuronal para Dataset Iris 🌸");
    
    // Cargar dataset
    println!("Cargando dataset...");
    let dataset = load_iris_dataset("dataset.csv")?;
    
    println!("Dataset cargado exitosamente!");
    println!("Entrenamiento: {} muestras", dataset.train_features.len());
    println!("Prueba: {} muestras", dataset.test_features.len());
    
    // Dividir datos de entrenamiento en train/validation (70/10)
    let train_size = (dataset.train_features.len() as f64 * 0.875) as usize; // 70% del total
    let val_features = dataset.train_features[train_size..].to_vec();
    let val_labels = dataset.train_labels[train_size..].to_vec();
    let train_features = dataset.train_features[..train_size].to_vec();
    let train_labels = dataset.train_labels[..train_size].to_vec();
    
    println!("Entrenamiento: {} muestras", train_features.len());
    println!("Validación: {} muestras", val_features.len());
    
    // Crear y entrenar la red neuronal
    let mut nn = IrisNeuralNetwork::new(0.001); // Learning rate = 0.01
    
    // Entrenar por 1000 epochs y obtener historial de pérdidas
    let (train_losses, val_losses) = nn.train(&train_features, &train_labels, &val_features, &val_labels, 10000);
    
    // Graficar curva de entrenamiento
    println!("\n📊 Generando gráfica de entrenamiento...");
    plot_training_curve(&train_losses, &val_losses, "training_curve.png")?;
    
    // Evaluar en conjunto de prueba
    println!("\n🎯 Evaluando en conjunto de prueba...");
    let test_accuracy = nn.evaluate(&dataset.test_features, &dataset.test_labels);
    println!("Precisión en prueba: {:.2}%", test_accuracy * 100.0);
    
    // Evaluar en conjunto de entrenamiento
    let train_accuracy = nn.evaluate(&dataset.train_features, &dataset.train_labels);
    println!("Precisión en entrenamiento: {:.2}%", train_accuracy * 100.0);
    
    // Ejemplo de predicción individual
    println!("\n🔍 Ejemplo de predicción:");
    if !dataset.test_features.is_empty() {
        let sample = &dataset.test_features[0];
        let actual = dataset.test_labels[0];
        let (predicted, probabilities) = nn.predict(sample);
        
        let species = ["Setosa", "Versicolor", "Virginica"];
        println!("Muestra: {:?}", sample);
        println!("Clase real: {} ({})", actual, species[actual]);
        println!("Clase predicha: {} ({})", predicted, species[predicted]);
        println!("Probabilidades: {:?}", probabilities);
    }
    
    Ok(())
}
