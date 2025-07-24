// Estructura para una red neuronal multicapa simple
#[derive(Debug)]
struct NeuralNetwork {
    // Pesos entre entrada y capa oculta
    weights_input_hidden: Vec<Vec<f64>>,
    // Bias de la capa oculta
    bias_hidden: Vec<f64>,
    
    // Pesos entre capa oculta y salida
    weights_hidden_output: Vec<Vec<f64>>,
    // Bias de la capa de salida
    bias_output: Vec<f64>,
    
    learning_rate: f64,
}

impl NeuralNetwork {
    // Constructor: inicializa una red con arquitectura específica
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        // Inicializar pesos entrada → oculta
        let mut weights_input_hidden = Vec::new();
        for i in 0..input_size {
            let mut row = Vec::new();
            for j in 0..hidden_size {
                // Pesos pequeños aleatorios (simulado con función determinista)
                let weight = ((i + j) as f64 * 0.7) % 1.0 - 0.5; // Entre -0.5 y 0.5
                row.push(weight);
            }
            weights_input_hidden.push(row);
        }
        
        // Inicializar bias capa oculta
        let mut bias_hidden = Vec::new();
        for i in 0..hidden_size {
            bias_hidden.push((i as f64 * 0.3) % 0.4 - 0.2); // Entre -0.2 y 0.2
        }
        
        // Inicializar pesos oculta → salida
        let mut weights_hidden_output = Vec::new();
        for i in 0..hidden_size {
            let mut row = Vec::new();
            for j in 0..output_size {
                let weight = ((i * 2 + j) as f64 * 0.5) % 1.0 - 0.5;
                row.push(weight);
            }
            weights_hidden_output.push(row);
        }
        
        // Inicializar bias capa salida
        let mut bias_output = Vec::new();
        for i in 0..output_size {
            bias_output.push((i as f64 * 0.1) % 0.3 - 0.15);
        }
        
        Self {
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
            learning_rate,
        }
    }
    
    // Función de activación sigmoide
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    // Forward pass: calcula salida de la red
    fn forward(&self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // 1. Calcular activaciones de capa oculta
        let mut hidden = Vec::new();
        
        for h in 0..self.bias_hidden.len() {
            let mut sum = self.bias_hidden[h];
            
            // Sumar contribuciones de todas las entradas
            for (i, &input) in inputs.iter().enumerate() {
                sum += input * self.weights_input_hidden[i][h];
            }
            
            hidden.push(Self::sigmoid(sum));
        }
        
        // 2. Calcular activaciones de capa de salida
        let mut output = Vec::new();
        
        for o in 0..self.bias_output.len() {
            let mut sum = self.bias_output[o];
            
            // Sumar contribuciones de capa oculta
            for (h, &hidden_val) in hidden.iter().enumerate() {
                sum += hidden_val * self.weights_hidden_output[h][o];
            }
            
            output.push(Self::sigmoid(sum));
        }
        
        (hidden, output)
    }
    
    // Entrenamiento con backpropagation
    fn train(&mut self, inputs: &[f64], targets: &[f64]) -> f64 {
        // 1. Forward pass
        let (hidden, output) = self.forward(inputs);
        
        // 2. Calcular errores de capa de salida
        let mut output_errors = Vec::new();
        let mut total_error = 0.0;
        
        for (o, (&out, &target)) in output.iter().zip(targets.iter()).enumerate() {
            let error = target - out;
            total_error += 0.5 * error.powi(2);
            
            // Delta para backpropagation (derivada de sigmoide)
            let delta = error * out * (1.0 - out);
            output_errors.push(delta);
        }
        
        // 3. Calcular errores de capa oculta (backpropagation)
        let mut hidden_errors = Vec::with_capacity(hidden.len());
        
        for h in 0..hidden.len() {
            let mut error_sum = 0.0;
            
            // Sumar errores propagados desde capa de salida
            for (o, &output_error) in output_errors.iter().enumerate() {
                error_sum += output_error * self.weights_hidden_output[h][o];
            }
            
            // Delta para capa oculta
            let delta = error_sum * hidden[h] * (1.0 - hidden[h]);
            hidden_errors.push(delta);
        }
        
        // 4. Actualizar pesos oculta → salida
        for h in 0..hidden.len() {
            for (o, &output_error) in output_errors.iter().enumerate() {
                self.weights_hidden_output[h][o] += self.learning_rate * output_error * hidden[h];
            }
        }
        
        // 5. Actualizar bias de salida
        for (o, &output_error) in output_errors.iter().enumerate() {
            self.bias_output[o] += self.learning_rate * output_error;
        }
        
        // 6. Actualizar pesos entrada → oculta
        for (i, &input) in inputs.iter().enumerate() {
            for (h, &hidden_error) in hidden_errors.iter().enumerate() {
                self.weights_input_hidden[i][h] += self.learning_rate * hidden_error * input;
            }
        }
        
        // 7. Actualizar bias de capa oculta
        for (h, &hidden_error) in hidden_errors.iter().enumerate() {
            self.bias_hidden[h] += self.learning_rate * hidden_error;
        }
        
        total_error
    }
    
    // Predicción
    fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let (_, output) = self.forward(inputs);
        output
    }
    
    // Predicción binaria (para clasificación)
    fn predict_binary(&self, inputs: &[f64]) -> Vec<bool> {
        let output = self.predict(inputs);
        output.iter().map(|&x| x > 0.5).collect()
    }
}

fn main() {
    println!("=== Red Neuronal Multicapa para XOR ===\n");
    
    // Crear red: 2 entradas, 3 neuronas ocultas, 1 salida
    let mut network = NeuralNetwork::new(2, 3, 1, 1.0);

    
    // Datos de entrenamiento: función XOR
    let training_data = [
        ([0.0, 0.0], [0.0]), // 0 XOR 0 = 0
        ([0.0, 1.0], [1.0]), // 0 XOR 1 = 1
        ([1.0, 0.0], [1.0]), // 1 XOR 0 = 1
        ([1.0, 1.0], [0.0]), // 1 XOR 1 = 0
    ];
    
    println!("Datos de entrenamiento (función XOR):");
    for (inputs, targets) in &training_data {
        println!("  {:?} → {:?}", inputs, targets);
    }
    println!();
    
    println!("Estado inicial de la red:");
    for (inputs, _) in &training_data {
        let output = network.predict(inputs);
        println!("  {:?} → {:.4}", inputs, output[0]);
    }
    println!();
    
    // Entrenar la red
    let epochs = 5000;
    for epoch in 0..epochs {
        let mut total_error = 0.0;
        
        // Entrenar con cada ejemplo
        for (inputs, targets) in &training_data {
            let error = network.train(inputs, targets);
            total_error += error;
        }
        
        // Mostrar progreso
        if epoch % 500 == 0 {
            println!("Época {}: Error total = {:.6}", epoch, total_error);
        }
    }
    
    println!("\n=== Resultados después del entrenamiento ===");
    
    println!("\nPruebas finales:");
    let mut all_correct = true;
    
    for (inputs, expected) in &training_data {
        let output = network.predict(inputs);
        let prediction = network.predict_binary(inputs);
        let expected_bool = expected[0] > 0.5;
        let correct = prediction[0] == expected_bool;
        
        if !correct {
            all_correct = false;
        }
        
        println!(
            "Entrada: {:?} → Salida: {:.4}, Predicción: {}, Esperado: {}, {}",
            inputs,
            output[0],
            prediction[0] as u8,
            expected[0] as u8,
            if correct { "✓" } else { "✗" }
        );
    }
    
    println!("\n🎉 Resultado: {}", 
        if all_correct { 
            "¡La red aprendió XOR correctamente!" 
        } else { 
            "La red aún no domina XOR. Intenta más épocas o ajustar parámetros." 
        }
    );
    
    // Ejemplo adicional: probar valores intermedios
    println!("\n=== Pruebas con valores no binarios ===");
    let test_cases = [
        [0.3, 0.7],
        [0.8, 0.2],
        [0.5, 0.5],
    ];
    
    for inputs in &test_cases {
        let output = network.predict(inputs);
        println!("Entrada: {:?} → Salida: {:.4}", inputs, output[0]);
    }
    
    println!("\n=== Arquitectura de la red ===");
    println!("- Entradas: {}", network.weights_input_hidden.len());
    println!("- Neuronas ocultas: {}", network.bias_hidden.len());
    println!("- Salidas: {}", network.bias_output.len());
    println!("- Total de parámetros: {}", 
        network.weights_input_hidden.len() * network.weights_input_hidden[0].len() +
        network.bias_hidden.len() +
        network.weights_hidden_output.len() * network.weights_hidden_output[0].len() +
        network.bias_output.len()
    );
}