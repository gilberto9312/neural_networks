

use plotters::prelude::*;
use std::collections::HashMap;

// ================ TRAITS PARA REGULARIZACIÓN ================
trait Regularizer {
    fn compute_loss(&self, network: &NeuralNetwork) -> f64;
    fn apply_to_gradient(&self, gradient: f64, weight: f64) -> f64;
}

// ================ REGULARIZADOR L1 ================
#[derive(Debug, Clone)]
struct L1Regularizer {
    lambda: f64,
}

impl L1Regularizer {
    fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Regularizer for L1Regularizer {
    fn compute_loss(&self, network: &NeuralNetwork) -> f64 {
        let mut l1_loss = 0.0;
        
        // L1 de pesos entrada → oculta
        for row in &network.weights_input_hidden {
            for &weight in row {
                l1_loss += self.lambda * weight.abs();
            }
        }
        
        // L1 de pesos oculta → salida
        for row in &network.weights_hidden_output {
            for &weight in row {
                l1_loss += self.lambda * weight.abs();
            }
        }
        
        l1_loss
    }
    
    fn apply_to_gradient(&self, gradient: f64, weight: f64) -> f64 {
        // L1: añade λ * sign(weight) al gradiente
        gradient + self.lambda * weight.signum()
    }
    
    
}

// ================ REGULARIZADOR L2 ================
#[derive(Debug, Clone)]
struct L2Regularizer {
    lambda: f64,
}

impl L2Regularizer {
    fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Regularizer for L2Regularizer {
    fn compute_loss(&self, network: &NeuralNetwork) -> f64 {
        let mut l2_loss = 0.0;
        
        // L2 de pesos entrada → oculta
        for row in &network.weights_input_hidden {
            for &weight in row {
                l2_loss += 0.5 * self.lambda * weight * weight ;
            }
        }
        
        // L2 de pesos oculta → salida
        for row in &network.weights_hidden_output {
            for &weight in row {
                l2_loss += 0.5 * self.lambda * weight * weight;
            }
        }
        
        l2_loss
    }
    
    fn apply_to_gradient(&self, gradient: f64, weight: f64) -> f64 {
        // L2: añade λ * weight al gradiente
        gradient + self.lambda * weight
    }
    
   
}

// ================ REGULARIZADOR DROPOUT ================
#[derive(Debug, Clone)]
struct DropoutRegularizer {
    rate: f64,
}

impl DropoutRegularizer {
    fn new(rate: f64) -> Self {
        Self { rate }
    }
    
    // Dropout no contribuye a la función de pérdida
    fn apply_dropout(&self, hidden: &mut Vec<f64>, rng_state: &mut u64, training: bool) -> Vec<bool> {
        let mut mask = vec![true; hidden.len()];
        
        if !training {
            return mask; // Sin dropout durante evaluación
        }
        
        for i in 0..hidden.len() {
            // Generar número aleatorio
            *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = ((*rng_state / 65536) % 32768) as f64 / 32768.0;
            
            if random_val < self.rate {
                hidden[i] = 0.0; // Apagar neurona
                mask[i] = false;
            } else {
                // Escalar por (1 - dropout_rate) para mantener la escala esperada
                hidden[i] /= 1.0 - self.rate;
            }
        }
        
        mask
    }
}

impl Regularizer for DropoutRegularizer {
    fn compute_loss(&self, _network: &NeuralNetwork) -> f64 {
        0.0 // Dropout no añade pérdida explícita
    }
    
    fn apply_to_gradient(&self, gradient: f64, _weight: f64) -> f64 {
        gradient // Dropout no modifica gradientes directamente
    }
    
    
}

// ================ RED NEURONAL MODULAR ================
#[derive(Debug, Clone)]
struct NeuralNetwork {
    weights_input_hidden: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_output: Vec<f64>,
    learning_rate: f64,
    dropout: Option<DropoutRegularizer>,
    rng_state: u64,
}

impl NeuralNetwork {
    fn new_with_seed(
        input_size: usize, 
        hidden_size: usize, 
        output_size: usize, 
        learning_rate: f64, 
        seed: u64
    ) -> Self {
        let mut rng_state = seed;
        
        let mut next_random = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            (rng_state / 65536) % 32768
        };
        
        let mut random_weight = || ((next_random() as f64) / 32768.0) * 2.0 - 1.0;
        
        // Inicializar pesos entrada → oculta
        let mut weights_input_hidden = Vec::new();
        for _i in 0..input_size {
            let mut row = Vec::new();
            for _j in 0..hidden_size {
                row.push(random_weight() * 0.5);
            }
            weights_input_hidden.push(row);
        }
        
        let mut bias_hidden = Vec::new();
        for _i in 0..hidden_size {
            bias_hidden.push(random_weight() * 0.1);
        }
        
        // Inicializar pesos oculta → salida
        let mut weights_hidden_output = Vec::new();
        for _i in 0..hidden_size {
            let mut row = Vec::new();
            for _j in 0..output_size {
                row.push(random_weight() * 0.5);
            }
            weights_hidden_output.push(row);
        }
        
        let mut bias_output = Vec::new();
        for _i in 0..output_size {
            bias_output.push(random_weight() * 0.1);
        }
        
        Self {
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
            learning_rate,
            dropout: None,
            rng_state,
        }
    }
    
    // Configurar dropout
    fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout = Some(DropoutRegularizer::new(rate));
        self
    }
    
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    // ================ FORWARD PASS ================
    fn forward(&mut self, inputs: &[f64], training: bool) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
        // Capa oculta
        let mut hidden = Vec::new();
        for h in 0..self.bias_hidden.len() {
            let mut sum = self.bias_hidden[h];
            for (i, &input) in inputs.iter().enumerate() {
                sum += input * self.weights_input_hidden[i][h];
            }
            hidden.push(Self::sigmoid(sum));
        }
        
        // Aplicar dropout si está configurado
        let dropout_mask = if let Some(ref dropout) = self.dropout {
            dropout.apply_dropout(&mut hidden, &mut self.rng_state, training)
        } else {
            vec![true; hidden.len()]
        };
        
        // Capa de salida
        let mut output = Vec::new();
        for o in 0..self.bias_output.len() {
            let mut sum = self.bias_output[o];
            for (h, &hidden_val) in hidden.iter().enumerate() {
                sum += hidden_val * self.weights_hidden_output[h][o];
            }
            output.push(Self::sigmoid(sum));
        }
        
        (hidden, output, dropout_mask)
    }
    
    // ================ GRADIENTES CON REGULARIZADOR OPCIONAL ================
    fn compute_gradients<R: Regularizer>(&mut self, inputs: &[f64], targets: &[f64], regularizer: Option<&R>) -> (NetworkGradients, f64) {
        let (hidden, output, dropout_mask) = self.forward(inputs, true);
        
        // Error base
        let mut output_errors = Vec::new();
        let mut base_error = 0.0;
        
        for (&out, &target) in output.iter().zip(targets.iter()) {
            let error = target - out;
            base_error += 0.5 * error.powi(2);
            let delta = -error * out * (1.0 - out);
            output_errors.push(delta);
        }
        
        // Añadir pérdida de regularización si hay regularizador
        let reg_loss = if let Some(reg) = regularizer {
            reg.compute_loss(self)
        } else {
            0.0
        };
        let total_error = base_error + reg_loss;
        
        // Errores de capa oculta (considerando dropout)
        let mut hidden_errors = Vec::new();
        for h in 0..hidden.len() {
            let mut error_sum = 0.0;
            for (o, &output_error) in output_errors.iter().enumerate() {
                error_sum += output_error * self.weights_hidden_output[h][o];
            }
            
            let delta = if dropout_mask[h] {
                error_sum * hidden[h] * (1.0 - hidden[h])
            } else {
                0.0
            };
            hidden_errors.push(delta);
        }
        
        // ================ GRADIENTES CON REGULARIZACIÓN ================
        let mut grad_weights_hidden_output = vec![vec![0.0; self.weights_hidden_output[0].len()]; self.weights_hidden_output.len()];
        for h in 0..hidden.len() {
            for (o, &output_error) in output_errors.iter().enumerate() {
                let base_grad = output_error * hidden[h];
                let final_grad = if let Some(reg) = regularizer {
                    reg.apply_to_gradient(base_grad, self.weights_hidden_output[h][o])
                } else {
                    base_grad
                };
                grad_weights_hidden_output[h][o] = final_grad;
            }
        }
        
        let mut grad_weights_input_hidden = vec![vec![0.0; self.weights_input_hidden[0].len()]; self.weights_input_hidden.len()];
        for (i, &input) in inputs.iter().enumerate() {
            for (h, &hidden_error) in hidden_errors.iter().enumerate() {
                let base_grad = hidden_error * input;
                let final_grad = if let Some(reg) = regularizer {
                    reg.apply_to_gradient(base_grad, self.weights_input_hidden[i][h])
                } else {
                    base_grad
                };
                grad_weights_input_hidden[i][h] = final_grad;
            }
        }
        
        let gradients = NetworkGradients {
            grad_weights_input_hidden,
            grad_bias_hidden: hidden_errors.clone(),
            grad_weights_hidden_output,
            grad_bias_output: output_errors.clone(),
        };
        
        (gradients, total_error)
    }
    
    // Evaluación sin dropout
    fn evaluate(&mut self, inputs: &[f64]) -> Vec<f64> {
        let (_, output, _) = self.forward(inputs, false);
        output
    }
}

// ================ ESTRUCTURAS AUXILIARES ================
#[derive(Debug, Clone)]
struct NetworkGradients {
    grad_weights_input_hidden: Vec<Vec<f64>>,
    grad_bias_hidden: Vec<f64>,
    grad_weights_hidden_output: Vec<Vec<f64>>,
    grad_bias_output: Vec<f64>,
}

struct AdamOptimizer {
    m_weights_input_hidden: Vec<Vec<f64>>,
    m_bias_hidden: Vec<f64>,
    m_weights_hidden_output: Vec<Vec<f64>>,
    m_bias_output: Vec<f64>,
    v_weights_input_hidden: Vec<Vec<f64>>,
    v_bias_hidden: Vec<f64>,
    v_weights_hidden_output: Vec<Vec<f64>>,
    v_bias_output: Vec<f64>,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: f64,
}

impl AdamOptimizer {
    fn new(network: &NeuralNetwork, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        let m_weights_input_hidden = vec![vec![0.0; network.weights_input_hidden[0].len()]; network.weights_input_hidden.len()];
        let m_bias_hidden = vec![0.0; network.bias_hidden.len()];
        let m_weights_hidden_output = vec![vec![0.0; network.weights_hidden_output[0].len()]; network.weights_hidden_output.len()];
        let m_bias_output = vec![0.0; network.bias_output.len()];
        
        let v_weights_input_hidden = vec![vec![0.0; network.weights_input_hidden[0].len()]; network.weights_input_hidden.len()];
        let v_bias_hidden = vec![0.0; network.bias_hidden.len()];
        let v_weights_hidden_output = vec![vec![0.0; network.weights_hidden_output[0].len()]; network.weights_hidden_output.len()];
        let v_bias_output = vec![0.0; network.bias_output.len()];
        
        Self {
            m_weights_input_hidden, m_bias_hidden, m_weights_hidden_output, m_bias_output,
            v_weights_input_hidden, v_bias_hidden, v_weights_hidden_output, v_bias_output,
            beta1, beta2, epsilon, t: 0.0,
        }
    }
    
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &NetworkGradients) {
        self.t += 1.0;
        let bias_correction1 = 1.0 - self.beta1.powf(self.t);
        let bias_correction2 = 1.0 - self.beta2.powf(self.t);
        
        // Actualizar pesos entrada → oculta
        for i in 0..network.weights_input_hidden.len() {
            for j in 0..network.weights_input_hidden[i].len() {
                let grad = gradients.grad_weights_input_hidden[i][j];
                self.m_weights_input_hidden[i][j] = self.beta1 * self.m_weights_input_hidden[i][j] + (1.0 - self.beta1) * grad;
                self.v_weights_input_hidden[i][j] = self.beta2 * self.v_weights_input_hidden[i][j] + (1.0 - self.beta2) * grad * grad;
                let m_hat = self.m_weights_input_hidden[i][j] / bias_correction1;
                let v_hat = self.v_weights_input_hidden[i][j] / bias_correction2;
                network.weights_input_hidden[i][j] -= network.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
        
        // Bias oculta
        for i in 0..network.bias_hidden.len() {
            let grad = gradients.grad_bias_hidden[i];
            self.m_bias_hidden[i] = self.beta1 * self.m_bias_hidden[i] + (1.0 - self.beta1) * grad;
            self.v_bias_hidden[i] = self.beta2 * self.v_bias_hidden[i] + (1.0 - self.beta2) * grad * grad;
            let m_hat = self.m_bias_hidden[i] / bias_correction1;
            let v_hat = self.v_bias_hidden[i] / bias_correction2;
            network.bias_hidden[i] -= network.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
        
        // Pesos oculta → salida
        for i in 0..network.weights_hidden_output.len() {
            for j in 0..network.weights_hidden_output[i].len() {
                let grad = gradients.grad_weights_hidden_output[i][j];
                self.m_weights_hidden_output[i][j] = self.beta1 * self.m_weights_hidden_output[i][j] + (1.0 - self.beta1) * grad;
                self.v_weights_hidden_output[i][j] = self.beta2 * self.v_weights_hidden_output[i][j] + (1.0 - self.beta2) * grad * grad;
                let m_hat = self.m_weights_hidden_output[i][j] / bias_correction1;
                let v_hat = self.v_weights_hidden_output[i][j] / bias_correction2;
                network.weights_hidden_output[i][j] -= network.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
        
        // Bias salida
        for i in 0..network.bias_output.len() {
            let grad = gradients.grad_bias_output[i];
            self.m_bias_output[i] = self.beta1 * self.m_bias_output[i] + (1.0 - self.beta1) * grad;
            self.v_bias_output[i] = self.beta2 * self.v_bias_output[i] + (1.0 - self.beta2) * grad * grad;
            let m_hat = self.m_bias_output[i] / bias_correction1;
            let v_hat = self.v_bias_output[i] / bias_correction2;
            network.bias_output[i] -= network.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

// ================ FUNCIÓN DE ENTRENAMIENTO GENÉRICA ================
fn train_network<R: Regularizer>(
    mut network: NeuralNetwork,
    mut optimizer: AdamOptimizer,
    training_data: &[([f64; 2], [f64; 1])],
    validation_data: &[([f64; 2], [f64; 1])],
    max_epochs: usize,
    target_error: f64,
    regularizer: Option<R>,
    name: &str,
) -> (f64, f64, bool, usize) {
    println!("\n🔹 Entrenando con: {}", name);
    
    let mut converged = false;
    let mut epochs_completed = 0;
    
    for epoch in 0..max_epochs {
        // Entrenamiento
        let mut train_error = 0.0;
        for (inputs, targets) in training_data {
            let (gradients, error) = network.compute_gradients(inputs, targets, regularizer.as_ref());
            optimizer.update(&mut network, &gradients);
            train_error += error;
        }
        
        // Validación (sin regularización en el error mostrado)
        let mut val_error = 0.0;
        for (inputs, targets) in validation_data {
            let output = network.evaluate(inputs);
            for (&out, &target) in output.iter().zip(targets.iter()) {
                let error = target - out;
                val_error += 0.5 * error.powi(2);
            }
        }
        
        epochs_completed = epoch + 1;
        
        // Mostrar progreso cada 500 épocas
        if epoch % 500 == 0 {
            println!("  {}: Época {}, Error Val = {:.6}", name, epoch, val_error);
        }
        
        // Verificar convergencia usando error de validación
        if val_error < target_error {
            converged = true;
            println!("🎉 {} CONVERGIÓ en época {} con error {:.6}!", name, epoch, val_error);
            break;
        }
    }
    
    // Error final
    let mut final_train = 0.0;
    let mut final_val = 0.0;
    
    for (inputs, targets) in training_data {
        let output = network.evaluate(inputs);
        for (&out, &target) in output.iter().zip(targets.iter()) {
            let error = target - out;
            final_train += 0.5 * error.powi(2);
        }
    }
    
    for (inputs, targets) in validation_data {
        let output = network.evaluate(inputs);
        for (&out, &target) in output.iter().zip(targets.iter()) {
            let error = target - out;
            final_val += 0.5 * error.powi(2);
        }
    }
    
    // Mostrar resultado final
    let status = if converged { "✅ APRENDIÓ" } else { "❌ NO CONVERGIÓ" };
    println!("  → {} en {} épocas: Train={:.6}, Val={:.6}", status, epochs_completed, final_train, final_val);
    
    (final_train, final_val, converged, epochs_completed)
}



// ================ ESTRUCTURA PARA MÉTRICAS POR ÉPOCA ================
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    pub name: String,
    pub train_err: f64,
    pub val_err: f64,
    pub converged: bool,
    pub epochs: usize
}

// ================ REGISTRADOR DE MÉTRICAS ================
#[derive(Debug, Clone)]
pub struct MetricsLogger {
    pub metrics: Vec<EpochMetrics>,
}

impl MetricsLogger {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }
    
    pub fn log_epoch(&mut self, metrics: EpochMetrics) {
        self.metrics.push(metrics);
    }
    
    // Método para generar todas las gráficas
    pub fn plot_all_metrics(&self, filename: &str, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(filename, (width, height)).into_drawing_area();
        root.fill(&WHITE)?;

        // Dividir en 1 fila x 2 columnas
        let areas = root.split_evenly((1, 2));

        // Ajustado: Solo se grafican errores
        self.plot_train_val_error(&areas[0])?; // A la izquierda
        self.plot_convergence_summary(&areas[1])?; // A la derecha (opcional, ver más abajo)

        root.present()?;
        println!("📊 Gráficas guardadas en: {}", filename);
        Ok(())
    }
    
    pub fn plot_train_val_error(
        &self,
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.metrics.is_empty() {
            return Ok(());
        }

        let epochs: Vec<f64> = (0..self.metrics.len()).map(|i| i as f64).collect();
        let train_errs: Vec<f64> = self.metrics.iter().map(|m| m.train_err).collect();
        let val_errs: Vec<f64> = self.metrics.iter().map(|m| m.val_err).collect();

        let min_val = train_errs
            .iter()
            .chain(val_errs.iter())
            .fold(f64::INFINITY, |a, &b| a.min(b))
            * 0.9;

        let max_val = train_errs
            .iter()
            .chain(val_errs.iter())
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            * 1.1;

        let mut chart = ChartBuilder::on(area)
            .caption("📉 Train vs Val Error", ("Arial", 20).into_font().color(&BLACK))
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0f64..epochs.len() as f64, min_val..max_val)?;

        chart.configure_mesh()
            .x_desc("Época")
            .y_desc("Error")
            .draw()?;

        chart.draw_series(LineSeries::new(
            epochs.iter().zip(train_errs.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Train Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

        chart.draw_series(LineSeries::new(
            epochs.iter().zip(val_errs.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Val Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &RED));

        chart.configure_series_labels().draw()?;

        Ok(())
    }

    pub fn plot_convergence_summary(
        &self,
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.metrics.is_empty() {
            return Ok(());
        }

        let converged_count = self.metrics.iter().filter(|m| m.converged).count();
        let not_converged_count = self.metrics.len() - converged_count;

        let max_y = (converged_count.max(not_converged_count) as f64 * 1.2).ceil();

        let mut chart = ChartBuilder::on(area)
            .caption("📌 Convergencia", ("Arial", 20).into_font().color(&BLACK))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..2, 0f64..max_y)?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .x_labels(2)
            .x_label_formatter(&|x| match x {
                0 => "Converged".to_string(),
                1 => "Not Converged".to_string(),
                _ => "".to_string(),
            })
            .y_desc("Cantidad")
            .draw()?;

        chart.draw_series(vec![
            Rectangle::new([(0, 0.0), (1, converged_count as f64)], GREEN.filled()),
            Rectangle::new([(1, 0.0), (2, not_converged_count as f64)], RED.filled()),
        ])?;

        Ok(())
    }
}

// ================ EJEMPLO DE USO INTEGRADO ================
fn ejemplo_con_red_neuronal(results: MetricsLogger) -> Result<(), Box<dyn std::error::Error>> {
    // Generar gráficas al final del entrenamiento
    results.plot_all_metrics("training_metrics.png", 1800, 1200)?;
    
    Ok(())
}

// ================ INTEGRACIÓN CON TU CÓDIGO EXISTENTE ================

// Modifica tu función de entrenamiento para incluir el logger


// ================ FUNCIÓN PRINCIPAL ================
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 COMPARACIÓN MODULAR DE REGULARIZACIÓN");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let training_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.1, 1.0], [0.0]),
    ];
    
    let validation_data = training_data.clone();
    let max_epochs = 1000;
    let target_error = 0.01; // Error objetivo para convergencia
    
    println!("📊 Datos de entrenamiento (función XOR):");
    for (inputs, targets) in &training_data {
        println!("   {:?} → {:?}", inputs, targets);
    }
    println!("🎯 Error objetivo: {}, Épocas máximas: {}\n", target_error, max_epochs);
    
    let mut results: Vec<EpochMetrics> = Vec::new();
    
    // 1. Sin regularización
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let (train_err, val_err, converged, epochs) = train_network(
            network, optimizer, &training_data, &validation_data, max_epochs, target_error,
            None::<L1Regularizer>, "Sin regularización"
        );
        results.push(EpochMetrics {
                    name: "Sin regularización".to_string(),
                    train_err,
                    val_err,
                    converged,
                    epochs,
                });

        }
    
    // 2. Solo L1
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let l1_reg = L1Regularizer::new(0.0001);
        let (train_err, val_err, converged, epochs) = train_network(
            network, optimizer, &training_data, &validation_data, max_epochs, target_error,
            Some(l1_reg), "L1 Regularización (λ=0.0001)"
        );
        results.push(EpochMetrics {
                    name: "Sin regularización".to_string(),
                    train_err,
                    val_err,
                    converged,
                    epochs,
                });
    }
    
    // 3. Solo L2
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let l2_reg = L2Regularizer::new(0.0001);
        let (train_err, val_err, converged, epochs) = train_network(
            network, optimizer, &training_data, &validation_data, max_epochs, target_error,
            Some(l2_reg), "L2 Regularización (λ=0.0001)"
        );
        results.push(EpochMetrics {
                    name: "Sin regularización".to_string(),
                    train_err,
                    val_err,
                    converged,
                    epochs,
                });
    }
    
    // 4. Solo Dropout
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42).with_dropout(0.5);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let (train_err, val_err, converged, epochs) = train_network(
            network, optimizer, &training_data, &validation_data, max_epochs, target_error,
            None::<L1Regularizer>, "Dropout (50%)"
        );
        results.push(EpochMetrics {
                    name: "Sin regularización".to_string(),
                    train_err,
                    val_err,
                    converged,
                    epochs,
                });
    }
    
    // 5. L2 + Dropout (combinado)
    {
        let network = NeuralNetwork::new_with_seed(2, 8, 1, 0.1, 42).with_dropout(0.3);
        let optimizer = AdamOptimizer::new(&network, 0.9, 0.999, 1e-8);
        let l2_reg = L2Regularizer::new(0.0001); // λ más pequeño al combinar
        let (train_err, val_err, converged, epochs) = train_network(
            network, optimizer, &training_data, &validation_data, max_epochs, target_error,
            Some(l2_reg), "L2 + Dropout"
        );
        results.push(EpochMetrics {
                    name: "Sin regularización".to_string(),
                    train_err,
                    val_err,
                    converged,
                    epochs,
                });
    }
    
    // Resumen final
    println!("\n📈 RESUMEN FINAL");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for metric in &results {
        let status = if metric.converged { "✅" } else { "❌" };
        println!("🔹 {} {} → Épocas: {}, Val Error: {:.6}", 
                 metric.name, status, metric.epochs, metric.val_err);
    }
    
    // Mostrar el mejor
    if let Some(best) = results.iter()
    .filter(|m| m.converged )  // o m.converged si es bool
    .min_by(|a, b| {
        a.val_err
            .partial_cmp(&b.val_err)
            .unwrap_or(a.epochs.partial_cmp(&b.epochs).unwrap())
    })  {
        println!("📌 Mejor modelo: {} con error de validación {:.6}", best.name, best.val_err);
    } else {
        println!("\n😞 Ningún método convergió al error objetivo {}", target_error);
    }
    
    println!("\n✅ Comparación completada!");

    println!("🧠 RED NEURONAL CON VISUALIZACIÓN DE MÉTRICAS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Ejecutar ejemplo con datos 
    let logger = MetricsLogger { metrics: results };
    ejemplo_con_red_neuronal(logger)?;
    
    println!("✅ Entrenamiento completado y gráficas generadas!");
    
    Ok(())
}