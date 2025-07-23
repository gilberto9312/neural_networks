// Perceptrón simple entrenable - Sin librerías externas
// Aprende compuerta AND paso a paso
use std::f64::consts::E;

#[derive(Debug)]
struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl Perceptron {
    // Constructor: inicializa con pesos aleatorios pequeños
    fn new(num_inputs: usize, learning_rate: f64) -> Self {
        let mut weights = Vec::new();
        
        // Inicialización simple de pesos (entre -0.5 y 0.5)
        for i in 0..num_inputs {
            // Usamos una función simple para generar valores "aleatorios"
            let pseudo_random = (i as f64 * E).sin();
            weights.push(pseudo_random * 0.5);
        }
        
        Self {
            weights,
            bias: 0.1, // Bias inicial pequeño
            learning_rate,
        }
    }
    
    // Forward Pass: calcula la predicción
    fn predict(&self, inputs: &[f64]) -> f64 {
        // 1. Suma ponderada: w₁x₁ + w₂x₂ + ... + bias
        let mut weighted_sum = self.bias;
        
        for i in 0..inputs.len() {
            weighted_sum += self.weights[i] * inputs[i];
        }
        
        // 2. Función de activación escalón (step function)
        if weighted_sum >= 0.0 {
            1.0
        } else {
            0.0
        }
    }
    
    // Entrenamiento: ajusta pesos usando la regla del perceptrón
    fn train_step(&mut self, inputs: &[f64], expected_output: f64) {
        // 1. Hacer predicción
        let prediction = self.predict(inputs);
        
        // 2. Calcular error
        let error = expected_output - prediction;
        
        // 3. Ajustar pesos: nuevo_peso = peso_actual + (lr × error × entrada)
        for i in 0..self.weights.len() {
            self.weights[i] += self.learning_rate * error * inputs[i];
        }
        
        // 4. Ajustar bias: nuevo_bias = bias_actual + (lr × error)
        self.bias += self.learning_rate * error;
    }
    
    // Entrenar con múltiples ejemplos durante varias épocas
    fn train(&mut self, training_data: &[(Vec<f64>, f64)], epochs: usize) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            
            for (inputs, expected) in training_data {
                let prediction = self.predict(inputs);
                let error = expected - prediction;
                total_error += error.abs();
                
                self.train_step(inputs, *expected);
            }
            
            // Mostrar progreso cada 10 épocas
            if epoch % 2 == 0 || epoch == epochs - 1 {
                println!("Época {}: Error total = {:.2}", epoch, total_error);
            }
            
            // Si no hay error, el perceptrón ya aprendió
            if total_error == 0.0 {
                println!("¡Perceptrón entrenado perfectamente en {} épocas!", epoch);
                break;
            }
        }
    }
    
    // Función auxiliar para mostrar el estado actual
    fn show_state(&self) {
        println!("Pesos: {:?}", self.weights);
        println!("Bias: {:.3}", self.bias);
    }
}

fn main() {
    println!("=== Entrenando Perceptrón para compuerta AND ===\n");
    
    // Crear perceptrón con 2 entradas y tasa de aprendizaje 0.1
    //let mut perceptron = Perceptron::new(2, 0.1);
    let mut perceptron = Perceptron::new(2, 0.01);
    
    println!("Estado inicial:");
    perceptron.show_state();
    println!();
    
    // Datos de entrenamiento para compuerta AND
    // Formato: (entradas, salida_esperada)
    // let training_data = vec![
    //     (vec![0.0, 0.0], 0.0), // 0 AND 0 = 0
    //     (vec![0.0, 1.0], 0.0), // 0 AND 1 = 0
    //     (vec![1.0, 0.0], 0.0), // 1 AND 0 = 0
    //     (vec![1.0, 1.0], 1.0), // 1 AND 1 = 1
    // ];
    let training_data = add_noise_to_data(&create_training_data_and(), 0.8);
    println!("Datos de entrenamiento (compuerta AND):");
    for (inputs, expected) in &training_data {
        println!("{:?} -> {}", inputs, expected);
    }
    println!();
    
    // Entrenar el perceptrón
    perceptron.train(&training_data, 300);
    
    println!("\nEstado final:");
    perceptron.show_state();
    
    // Probar el perceptrón entrenado
    println!("\n=== Probando el perceptrón entrenado ===");
    for (inputs, expected) in &training_data {
        let prediction = perceptron.predict(inputs);
        let correct = if prediction == *expected { "✓" } else { "✗" };
        println!("Entrada: {:?} -> Predicción: {}, Esperado: {} {}", 
                 inputs, prediction, expected, correct);
    }
    
    
}

// Funciones auxiliares para experimentar

// Función para entrenar con diferentes compuertas lógicas
fn create_training_data_or() -> Vec<(Vec<f64>, f64)> {
    vec![
        (vec![0.0, 0.0], 0.0), // 0 OR 0 = 0
        (vec![0.0, 1.0], 1.0), // 0 OR 1 = 1
        (vec![1.0, 0.0], 1.0), // 1 OR 0 = 1
        (vec![1.0, 1.0], 1.0), // 1 OR 1 = 1
    ]
}

fn create_training_data_and() -> Vec<(Vec<f64>, f64)> {
    vec![
        (vec![0.0, 0.0], 0.0), // 0 AND 0 = 0
        (vec![0.0, 1.0], 0.0), // 0 AND 1 = 0
        (vec![1.0, 0.0], 0.0), // 1 AND 0 = 0
        (vec![1.0, 1.0], 1.0), // 1 AND 1 = 1
    ]
}

// Función para probar con datos ruidosos
fn add_noise_to_data(data: &[(Vec<f64>, f64)], noise_level: f64) -> Vec<(Vec<f64>, f64)> {
    data.iter().map(|(inputs, output)| {
        let noisy_inputs: Vec<f64> = inputs.iter()
            .enumerate()
            .map(|(i, &x)| {
                // Ruido simple usando índice
                let noise = ((i as f64 + 1.0) * 1.23).sin() * noise_level;
                (x + noise).max(0.0).min(1.0) // Mantener en rango [0,1]
            })
            .collect();
        (noisy_inputs, *output)
    }).collect()
}