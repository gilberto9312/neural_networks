// Estructura para una red neuronal multicapa con Momentum optimizer
#[derive(Debug)]
struct NeuralNetwork {
    // Pesos y bias (igual que antes)
    weights_input_hidden: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_output: Vec<f64>,
    
    // NUEVAS ESTRUCTURAS PARA MOMENTUM
    // Velocidades para pesos entrada → oculta
    velocity_weights_input_hidden: Vec<Vec<f64>>,
    // Velocidades para bias oculta
    velocity_bias_hidden: Vec<f64>,
    // Velocidades para pesos oculta → salida
    velocity_weights_hidden_output: Vec<Vec<f64>>,
    // Velocidades para bias salida
    velocity_bias_output: Vec<f64>,
    
    // Parámetros de optimización
    learning_rate: f64,
    momentum: f64, // Factor de momentum (β)
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64, momentum: f64) -> Self {
        // Inicializar pesos entrada → oculta
        let mut weights_input_hidden = Vec::new();
        let mut velocity_weights_input_hidden = Vec::new(); // NUEVO
        
        for i in 0..input_size {
            let mut row = Vec::new();
            let mut velocity_row = Vec::new(); // NUEVO
            
            for j in 0..hidden_size {
                let weight = ((i + j) as f64 * 0.7) % 1.0 - 0.5;
                row.push(weight);
                velocity_row.push(0.0); // Velocidad inicial = 0
            }
            weights_input_hidden.push(row);
            velocity_weights_input_hidden.push(velocity_row); // NUEVO
        }
        
        // Inicializar bias capa oculta
        let mut bias_hidden = Vec::new();
        let mut velocity_bias_hidden = Vec::new(); // NUEVO
        
        for i in 0..hidden_size {
            bias_hidden.push((i as f64 * 0.3) % 0.4 - 0.2);
            velocity_bias_hidden.push(0.0); // NUEVO
        }
        
        // Inicializar pesos oculta → salida
        let mut weights_hidden_output = Vec::new();
        let mut velocity_weights_hidden_output = Vec::new(); // NUEVO
        
        for i in 0..hidden_size {
            let mut row = Vec::new();
            let mut velocity_row = Vec::new(); // NUEVO
            
            for j in 0..output_size {
                let weight = ((i * 2 + j) as f64 * 0.5) % 1.0 - 0.5;
                row.push(weight);
                velocity_row.push(0.0); // NUEVO
            }
            weights_hidden_output.push(row);
            velocity_weights_hidden_output.push(velocity_row); // NUEVO
        }
        
        // Inicializar bias capa salida
        let mut bias_output = Vec::new();
        let mut velocity_bias_output = Vec::new(); // NUEVO
        
        for i in 0..output_size {
            bias_output.push((i as f64 * 0.1) % 0.3 - 0.15);
            velocity_bias_output.push(0.0); // NUEVO
        }
        
        Self {
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
            
            // NUEVAS ESTRUCTURAS
            velocity_weights_input_hidden,
            velocity_bias_hidden,
            velocity_weights_hidden_output,
            velocity_bias_output,
            
            learning_rate,
            momentum,
        }
    }
    
    // Función de activación sigmoide (sin cambios)
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    // Forward pass (sin cambios)
    fn forward(&self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // 1. Calcular activaciones de capa oculta
        let mut hidden = Vec::new();
        
        for h in 0..self.bias_hidden.len() {
            let mut sum = self.bias_hidden[h];
            
            for (i, &input) in inputs.iter().enumerate() {
                sum += input * self.weights_input_hidden[i][h];
            }
            
            hidden.push(Self::sigmoid(sum));
        }
        
        // 2. Calcular activaciones de capa de salida
        let mut output = Vec::new();
        
        for o in 0..self.bias_output.len() {
            let mut sum = self.bias_output[o];
            
            for (h, &hidden_val) in hidden.iter().enumerate() {
                sum += hidden_val * self.weights_hidden_output[h][o];
            }
            
            output.push(Self::sigmoid(sum));
        }
        
        (hidden, output)
    }
    
    // FUNCIÓN CLAVE: Actualizar con Momentum
    fn update_with_momentum(weight: &mut f64, velocity: &mut f64, gradient: f64, learning_rate: f64, momentum: f64) {
        // Paso 1: Actualizar velocidad
        // v_t = β * v_{t-1} + (1 - β) * gradient_t
        *velocity = momentum * (*velocity) + gradient;
        // Paso 2: Actualizar peso usando la velocidad
        // weight_t = weight_{t-1} - learning_rate * v_t
        *weight -= learning_rate * (*velocity);
        //*weight += learning_rate * (*velocity); 
    }
    
    // Entrenamiento con backpropagation + Momentum
    fn train(&mut self, inputs: &[f64], targets: &[f64]) -> f64 {
        // 1. Forward pass
        let (hidden, output) = self.forward(inputs);
        
        // 2. Calcular errores de capa de salida
        let mut output_errors = Vec::new();
        let mut total_error = 0.0;
        
        for (o, (&out, &target)) in output.iter().zip(targets.iter()).enumerate() {
            let error = target - out;
            total_error += 0.5 * error.powi(2);
            
            let delta = -error * out * (1.0 - out);
            output_errors.push(delta);
        }
        
        // 3. Calcular errores de capa oculta
        let mut hidden_errors = Vec::with_capacity(hidden.len());
        
        for h in 0..hidden.len() {
            let mut error_sum = 0.0;
            
            for (o, &output_error) in output_errors.iter().enumerate() {
                error_sum += output_error * self.weights_hidden_output[h][o];
            }
            
            let delta = error_sum * hidden[h] * (1.0 - hidden[h]);
            hidden_errors.push(delta);
        }
        
        // 4. ACTUALIZAR PESOS OCULTA → SALIDA CON MOMENTUM
        for h in 0..hidden.len() {
            for (o, &output_error) in output_errors.iter().enumerate() {
                let gradient = output_error * hidden[h];
                Self::update_with_momentum(&mut self.weights_hidden_output[h][o], &mut self.velocity_weights_hidden_output[h][o], gradient, self.learning_rate, self.momentum);
            }
        }
        
        // 5. ACTUALIZAR BIAS DE SALIDA CON MOMENTUM
        for (o, &output_error) in output_errors.iter().enumerate() {
            let gradient = output_error;
            Self::update_with_momentum(
                &mut self.bias_output[o],
                &mut self.velocity_bias_output[o],
                gradient,
                self.learning_rate, self.momentum
            );
        }
        
        // 6. ACTUALIZAR PESOS ENTRADA → OCULTA CON MOMENTUM
        for (i, &input) in inputs.iter().enumerate() {
            for (h, &hidden_error) in hidden_errors.iter().enumerate() {
                let gradient = hidden_error * input;
                Self::update_with_momentum(
                    &mut self.weights_input_hidden[i][h],
                    &mut self.velocity_weights_input_hidden[i][h],
                    gradient,
                    self.learning_rate, self.momentum
                );
            }
        }
        
        // 7. ACTUALIZAR BIAS DE CAPA OCULTA CON MOMENTUM
        for (h, &hidden_error) in hidden_errors.iter().enumerate() {
            let gradient = hidden_error;
            Self::update_with_momentum(
                &mut self.bias_hidden[h],
                &mut self.velocity_bias_hidden[h],
                gradient,
                self.learning_rate, self.momentum
            );
        }
        
        total_error
    }
    
    // Predicción (sin cambios)
    fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let (_, output) = self.forward(inputs);
        output
    }
    
    // Predicción binaria (sin cambios)
    fn predict_binary(&self, inputs: &[f64]) -> Vec<bool> {
        let output = self.predict(inputs);
        output.iter().map(|&x| x > 0.5).collect()
    }
    
    // FUNCIÓN PARA DIAGNOSTICAR MOMENTUM
    fn print_momentum_stats(&self) {
        println!("=== Estadísticas de Momentum ===");
        
        // Velocidad promedio de pesos entrada → oculta
        let avg_velocity_ih: f64 = self.velocity_weights_input_hidden
            .iter()
            .flat_map(|row| row.iter())
            .map(|&v| v.abs())
            .sum::<f64>() / (self.velocity_weights_input_hidden.len() * self.velocity_weights_input_hidden[0].len()) as f64;
        
        println!("Velocidad promedio (entrada → oculta): {:.6}", avg_velocity_ih);
        
        // Velocidad promedio de pesos oculta → salida
        let avg_velocity_ho: f64 = self.velocity_weights_hidden_output
            .iter()
            .flat_map(|row| row.iter())
            .map(|&v| v.abs())
            .sum::<f64>() / (self.velocity_weights_hidden_output.len() * self.velocity_weights_hidden_output[0].len()) as f64;
        
        println!("Velocidad promedio (oculta → salida): {:.6}", avg_velocity_ho);
    }

    fn debug_network_state(&self) {
        println!("=== Debug Network State ===");
        
        // Verificar si los pesos están en rangos normales
        let weight_stats = self.weights_hidden_output
            .iter()
            .flat_map(|row| row.iter())
            .fold((f64::INFINITY, f64::NEG_INFINITY, 0.0, 0), 
                |(min, max, sum, count), &w| {
                    (min.min(w), max.max(w), sum + w, count + 1)
                });
        
        println!("Pesos oculta→salida: min={:.4}, max={:.4}, promedio={:.4}", 
                 weight_stats.0, weight_stats.1, weight_stats.2 / weight_stats.3 as f64);
        
        // Verificar velocidades
        let velocity_stats = self.velocity_weights_hidden_output
            .iter()
            .flat_map(|row| row.iter())
            .fold((f64::INFINITY, f64::NEG_INFINITY, 0.0, 0), 
                |(min, max, sum, count), &v| {
                    (min.min(v.abs()), max.max(v.abs()), sum + v.abs(), count + 1)
                });
        
        println!("Velocidades (abs): min={:.6}, max={:.6}, promedio={:.6}", 
                 velocity_stats.0, velocity_stats.1, velocity_stats.2 / velocity_stats.3 as f64);
    }
}

fn main() {
    println!("=== Red Neuronal con Momentum Optimizer ===\n");
    
    // Crear red con momentum
    // Parámetros: input_size, hidden_size, output_size, learning_rate, momentum
        let mut network = NeuralNetwork::new(
        2,      // input_size
        4,      // hidden_size  
        1,      // output_size
        0.1,    // learning_rate (REDUCIDO de 0.5 a 0.1)
        0.8     // momentum (REDUCIDO de 0.9 a 0.8)
    );
    
    // Datos de entrenamiento: función XOR
    let training_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];
    
    println!("Configuración:");
    println!("- Learning rate: {}", network.learning_rate);
    println!("- Momentum (β): {}", network.momentum);
    println!();
    
    println!("Datos de entrenamiento (función XOR):");
    for (inputs, targets) in &training_data {
        println!("  {:?} → {:?}", inputs, targets);
    }
    println!();
    
    println!("Estado inicial:");
    for (inputs, _) in &training_data {
        let output = network.predict(inputs);
        println!("  {:?} → {:.4}", inputs, output[0]);
    }
    println!();
    
    // Entrenar la red
    let epochs = 5000;
    for epoch in 0..epochs {
        let mut total_error = 0.0;
        
        for (inputs, targets) in &training_data {
            let error = network.train(inputs, targets);
            total_error += error;
        }
        
        if epoch % 500 == 0 {
            println!("Época {}: Error total = {:.6}", epoch, total_error);
            if epoch > 0 {
                network.print_momentum_stats();
                network.debug_network_state();
            }
            println!();
        }
    }
    
    println!("=== Resultados finales ===");
    
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
            "¡La red con Momentum aprendió XOR correctamente!" 
        } else { 
            "La red aún necesita más entrenamiento." 
        }
    );
    
    // Estadísticas finales
    network.print_momentum_stats();
    
    println!("\n=== Comparación conceptual ===");
    println!("SGD básico: weight = weight - lr * gradient");
    println!("Con Momentum:");
    println!("  velocity = β * velocity + (1-β) * gradient");
    println!("  weight = weight - lr * velocity");
    println!("Beneficios: suaviza oscilaciones, escapa mínimos locales, convergencia más rápida");
}