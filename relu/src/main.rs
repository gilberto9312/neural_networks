use std::f64::consts::E;
use std::thread;
use std::time::{Duration, Instant};
use std::sync::mpsc;

/// Estructura para representar una neurona simple
#[derive(Debug, Clone)]
pub struct Neuron {
    pub synapses: f64,
    pub threshold: f64,
}

impl Neuron {
    pub fn new(value: f64) -> Self {
        Self {
            synapses: value,
            threshold: 0.0,
        }
    }
}

/// Estructura para medir rendimiento
#[derive(Debug)]
struct PerformanceResult {
    activation_name: String,
    duration: Duration,
    results: Vec<f64>,
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

/// ========== FUNCIÓN SIGMOID ==========
/// Función de activación Sigmoid
/// Fórmula: σ(x) = 1 / (1 + e^(-x))
/// Rango: (0, 1)
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + E.powf(-z))
}

/// Derivada de la función Sigmoid
/// Fórmula: σ'(x) = σ(x) * (1 - σ(x))
fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// ========== FUNCIÓN TANH ==========
/// Función de activación Tanh (Tangente Hiperbólica)
/// Fórmula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// Rango: (-1, 1)
fn tanh_activation(x: f64) -> f64 {
    let exp_x = E.powf(x);
    let exp_neg_x = E.powf(-x);
    (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
}

/// Derivada de la función Tanh
/// Fórmula: tanh'(x) = 1 - tanh²(x)
fn tanh_derivative(x: f64) -> f64 {
    let t = tanh_activation(x);
    1.0 - (t * t)
}

/// ========== FUNCIONES AUXILIARES ==========
/// Función para aplicar activación a una neurona
fn apply_activation(neuron: &mut Neuron, activation_fn: fn(f64) -> f64) {
    neuron.threshold = activation_fn(neuron.synapses);
}

/// Función para procesar muchos valores con una función de activación (para benchmark)
fn process_large_dataset(inputs: &[f64], activation_fn: fn(f64) -> f64) -> Vec<f64> {
    inputs.iter().map(|&x| activation_fn(x)).collect()
}

/// ========== COMPARACIONES Y ANÁLISIS ==========
/// Función para comparar las tres activaciones
fn compare_all_activations() {
    println!("=== COMPARACIÓN COMPLETA: ReLU vs TANH vs SIGMOID ===\n");
    
    let test_values = vec![-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0];
    
    println!("| Input  | ReLU   | Sigmoid | Tanh   | ReLU' | Sig'  | Tanh' |");
    println!("|--------|--------|---------|--------|-------|-------|-------|");
    
    for &x in &test_values {
        let relu_val = relu(x);
        let sig_val = sigmoid(x);
        let tanh_val = tanh_activation(x);
        let relu_deriv = relu_derivative(x);
        let sig_deriv = sigmoid_derivative(x);
        let tanh_deriv = tanh_derivative(x);
        
        println!("| {:6.1} | {:6.3} | {:7.3} | {:6.3} | {:5.1} | {:5.3} | {:5.3} |", 
                 x, relu_val, sig_val, tanh_val, relu_deriv, sig_deriv, tanh_deriv);
    }
}

/// Análisis detallado de por qué ReLU es más eficiente
fn analyze_relu_efficiency() {
    println!("\n=== ¿POR QUÉ ReLU ES MÁS EFICIENTE? ===\n");
    
    println!("🚀 VENTAJAS DE ReLU:");
    println!("1. COMPUTACIÓN SIMPLE:");
    println!("   • ReLU: max(0, x) → Una comparación y una selección");
    println!("   • Sigmoid: 1/(1 + e^(-x)) → Exponencial + división");
    println!("   • Tanh: (e^x - e^(-x))/(e^x + e^(-x)) → Dos exponenciales + operaciones");
    
    println!("\n2. PROBLEMA DEL GRADIENTE DESVANECIENTE:");
    println!("   • ReLU: Gradiente = 1 (si x > 0) → No se desvanece");
    println!("   • Sigmoid/Tanh: Gradientes → 0 en valores extremos");
    
    println!("\n3. SPARSITY (DISPERSIÓN):");
    println!("   • ReLU produce muchos ceros → Redes más dispersas");
    println!("   • Menos neuronas activas = menos cómputo");
    
    println!("\n⚠️ DESVENTAJAS DE ReLU:");
    println!("• 'Dying ReLU': Neuronas pueden 'morir' (siempre salida 0)");
    println!("• No diferenciable en x = 0");
    println!("• No acotada superiormente");
}

/// Demostración de transformación lineal → no-lineal
fn linear_to_nonlinear_demo() {
    println!("\n=== TRANSFORMACIÓN: LINEAL → NO-LINEAL ===\n");
    
    // Simulamos una función lineal simple: y = 2x + 1
    let linear_inputs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    
    println!("📊 ENTRADA LINEAL (y = 2x + 1):");
    let linear_outputs: Vec<f64> = linear_inputs.iter()
        .map(|&x| 2.0 * x + 1.0)
        .collect();
    
    for (i, (&input, &output)) in linear_inputs.iter().zip(linear_outputs.iter()).enumerate() {
        println!("  x[{}] = {:.1} → y = {:.1}", i, input, output);
    }
    
    println!("\n🔄 APLICANDO ACTIVACIONES NO-LINEALES:");
    
    // ReLU
    println!("\n🔸 Con ReLU:");
    for (i, &linear_out) in linear_outputs.iter().enumerate() {
        let nonlinear = relu(linear_out);
        println!("  Linear: {:.1} → ReLU: {:.1}", linear_out, nonlinear);
    }
    
    // Sigmoid
    println!("\n🔸 Con Sigmoid:");
    for (i, &linear_out) in linear_outputs.iter().enumerate() {
        let nonlinear = sigmoid(linear_out);
        println!("  Linear: {:.1} → Sigmoid: {:.3}", linear_out, nonlinear);
    }
    
    // Tanh
    println!("\n🔸 Con Tanh:");
    for (i, &linear_out) in linear_outputs.iter().enumerate() {
        let nonlinear = tanh_activation(linear_out);
        println!("  Linear: {:.1} → Tanh: {:.3}", linear_out, nonlinear);
    }
    
    println!("\n💡 CLAVE: Las funciones de activación introducen no-linealidad");
    println!("   Sin ellas, una red neuronal sería solo una regresión lineal compleja!");
}

/// Benchmark con multithreading
fn multithreaded_performance_test() {
    println!("\n=== BENCHMARK CON MULTITHREADING ===\n");
    
    // Generar un dataset grande para el benchmark
    let large_dataset: Vec<f64> = (0..1_000_000)
        .map(|i| (i as f64 / 100_000.0) - 5.0) // Rango de -5 a 5
        .collect();
    
    println!("🏁 Procesando {} valores con cada función de activación...", large_dataset.len());
    println!("🧵 Usando multithreading para comparar rendimiento\n");
    
    let (tx, rx) = mpsc::channel();
    
    // Clonar el dataset para cada hilo
    let dataset_relu = large_dataset.clone();
    let dataset_sigmoid = large_dataset.clone();
    let dataset_tanh = large_dataset.clone();
    
    let tx1 = tx.clone();
    let tx2 = tx.clone();
    let tx3 = tx.clone();
    
    // Hilo para ReLU
    let     relu_handle = thread::spawn(move || {
        let start = Instant::now();
        let results = process_large_dataset(&dataset_relu, relu);
        let duration = start.elapsed();
        
        tx1.send(PerformanceResult {
            activation_name: "ReLU".to_string(),
            duration,
            results,
        }).unwrap();
    });
    
    // Hilo para Sigmoid
    let sigmoid_handle = thread::spawn(move || {
        let start = Instant::now();
        let results = process_large_dataset(&dataset_sigmoid, sigmoid);
        let duration = start.elapsed();
        
        tx2.send(PerformanceResult {
            activation_name: "Sigmoid".to_string(),
            duration,
            results,
        }).unwrap();
    });
    
    // Hilo para Tanh
    let tanh_handle = thread::spawn(move || {
        let start = Instant::now();
        let results = process_large_dataset(&dataset_tanh, tanh_activation);
        let duration = start.elapsed();
        
        tx3.send(PerformanceResult {
            activation_name: "Tanh".to_string(),
            duration,
            results,
        }).unwrap();
    });
    
    // Recoger resultados en orden de llegada
    let mut results = Vec::new();
    for _ in 0..3 {
        if let Ok(result) = rx.recv() {
            println!("✅ {} terminó en: {:.4} segundos", 
                     result.activation_name, 
                     result.duration.as_secs_f64());
            results.push(result);
        }
    }
    
    // Esperar a que todos los hilos terminen
    relu_handle.join().unwrap();
    sigmoid_handle.join().unwrap();
    tanh_handle.join().unwrap();
    
    // Ordenar por tiempo de ejecución
    results.sort_by(|a, b| a.duration.cmp(&b.duration));
    
    println!("\n🏆 RANKING DE RENDIMIENTO:");
    for (i, result) in results.iter().enumerate() {
        let medal = match i {
            0 => "🥇",
            1 => "🥈", 
            2 => "🥉",
            _ => "  ",
        };
        println!("{} {}: {:.4} segundos", 
                 medal, result.activation_name, result.duration.as_secs_f64());
    }
    
    if let Some(fastest) = results.first() {
        if let Some(slowest) = results.last() {
            let speedup = slowest.duration.as_secs_f64() / fastest.duration.as_secs_f64();
            println!("\n⚡ {} es {:.2}x más rápido que {}", 
                     fastest.activation_name, speedup, slowest.activation_name);
        }
    }
}

/// Red neuronal ejemplo con las tres activaciones
fn neural_network_comparison() {
    println!("\n=== RED NEURONAL: COMPARACIÓN DE ACTIVACIONES ===\n");
    
    let inputs = vec![-2.5, -1.0, 0.0, 1.5, 3.0];
    println!("🔢 Entradas: {:?}\n", inputs);
    
    // Crear neuronas con cada activación
    let mut relu_neurons: Vec<Neuron> = inputs.iter()
        .map(|&x| Neuron::new(x))
        .collect();
    
    let mut sigmoid_neurons: Vec<Neuron> = inputs.iter()
        .map(|&x| Neuron::new(x))
        .collect();
    
    let mut tanh_neurons: Vec<Neuron> = inputs.iter()
        .map(|&x| Neuron::new(x))
        .collect();
    
    // Aplicar activaciones
    for neuron in &mut relu_neurons {
        apply_activation(neuron, relu);
    }
    
    for neuron in &mut sigmoid_neurons {
        apply_activation(neuron, sigmoid);
    }
    
    for neuron in &mut tanh_neurons {
        apply_activation(neuron, tanh_activation);
    }
    
    // Mostrar resultados
    println!("🔸 ReLU:");
    for (i, neuron) in relu_neurons.iter().enumerate() {
        println!("  Neurona {}: {:.2} → {:.3}", i+1, neuron.synapses, neuron.threshold);
    }
    
    println!("\n🔸 Sigmoid:");
    for (i, neuron) in sigmoid_neurons.iter().enumerate() {
        println!("  Neurona {}: {:.2} → {:.3}", i+1, neuron.synapses, neuron.threshold);
    }
    
    println!("\n🔸 Tanh:");
    for (i, neuron) in tanh_neurons.iter().enumerate() {
        println!("  Neurona {}: {:.2} → {:.3}", i+1, neuron.synapses, neuron.threshold);
    }
    
    // Análisis de sparsity (ReLU)
    let relu_zeros = relu_neurons.iter()
        .filter(|n| n.threshold == 0.0)
        .count();
    
    println!("\n📊 ANÁLISIS DE DISPERSIÓN:");
    println!("• ReLU produjo {} ceros de {} neuronas ({:.1}% sparse)", 
             relu_zeros, relu_neurons.len(), 
             (relu_zeros as f64 / relu_neurons.len() as f64) * 100.0);
    println!("• Sigmoid y Tanh nunca producen ceros exactos");
    println!("• Menor actividad = mayor eficiencia computacional");
}

fn main() {
    println!("🧠 RETO DE REDES NEURONALES: ReLU vs SIGMOID vs TANH");
    println!("===================================================\n");
    
    // Ejecutar todas las demostraciones
    compare_all_activations();
    analyze_relu_efficiency();
    linear_to_nonlinear_demo();
    neural_network_comparison();
    multithreaded_performance_test();
    
    println!("\n=== RESUMEN EJECUTIVO ===");
    println!("🎯 ReLU: La elección por defecto para capas ocultas");
    println!("   • Más rápido computacionalmente");
    println!("   • Evita el gradiente desvaneciente");
    println!("   • Produce sparsity natural");
    
    println!("🎯 Sigmoid: Para salidas de clasificación binaria");
    println!("   • Interpretación probabilística");
    println!("   • Salidas en rango (0,1)");
    
    println!("🎯 Tanh: Mejor que Sigmoid para capas ocultas");
    println!("   • Centrada en cero");
    println!("   • Gradientes más fuertes que Sigmoid");
    
    println!("\n🏆 CONCLUSIÓN: ReLU revolucionó las redes neuronales profundas");
    println!("   por su simplicidad y eficiencia computacional.");
    
    println!("\n✅ ¡Reto del día completado! Ahora entiendes por qué ReLU domina el deep learning.");
}