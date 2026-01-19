// Dataset loader y preparación

use crate::tokenizer::SimpleTokenizer;

pub const AFRICA_GALORE_SAMPLE: &str = r#"
Abeni was a talented artist who lived in Lagos. She painted vibrant scenes of daily life in the bustling markets.
Her favorite subjects were the colorful fabrics and the smiling faces of vendors.

The ancient city of Timbuktu held many secrets within its libraries. Scholars from across the world came to study
the precious manuscripts preserved there for centuries.

Kwame loved to play football with his friends after school. They would meet at the dusty field near the baobab tree
and play until sunset painted the sky orange and purple.

The rhythmic sounds of drums echoed through the village as people gathered for the celebration. Traditional dancers
moved gracefully to the beat wearing elaborate costumes decorated with beads and shells.

Amara opened her small café in Accra serving delicious jollof rice and plantains. The aroma of her cooking attracted
customers from far and wide who came to taste her famous dishes.

In the savanna elephants roamed freely across the vast grasslands. The herds moved together searching for water
during the dry season led by the wise matriarch.

The craftsman carefully carved intricate patterns into the wooden mask. His skilled hands had learned this art
from his father who learned it from his grandfather continuing a tradition spanning generations.

Children sang songs while walking to the well to fetch water. They balanced heavy jugs on their heads with
remarkable ease learned from years of practice.

The marketplace was alive with activity as vendors called out their prices. Fresh vegetables fruits and spices
created a rainbow of colors that delighted the eyes of shoppers.

Under the acacia tree the elders sat discussing important matters affecting the community. Their wisdom guided
the younger generation through challenges both old and new.
"#;

/// Dataset de secuencias tokenizadas
pub struct TextDataset {
    pub sequences: Vec<Vec<usize>>,
    pub seq_len: usize,
}

impl TextDataset {
    /// Crea dataset desde texto
    ///
    /// # Argumentos
    /// * `text` - Texto completo
    /// * `tokenizer` - Tokenizer entrenado
    /// * `seq_len` - Longitud de cada secuencia
    pub fn from_text(text: &str, tokenizer: &SimpleTokenizer, seq_len: usize) -> Self {
        // Tokenizar todo el texto
        let all_ids = tokenizer.encode(text, false);

        // Dividir en secuencias
        let mut sequences = Vec::new();

        for i in 0..all_ids.len().saturating_sub(seq_len) {
            sequences.push(all_ids[i..i + seq_len].to_vec());
        }

        Self { sequences, seq_len }
    }

    /// Obtiene un batch de datos
    ///
    /// # Argumentos
    /// * `batch_size` - Tamaño del batch
    ///
    /// # Retorna
    /// (inputs, targets) donde targets son inputs shifted por 1
    pub fn get_batch(&self, batch_size: usize) -> Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        if self.sequences.is_empty() {
            return None;
        }

        let mut rng = thread_rng();
        let batch_indices: Vec<usize> = (0..self.sequences.len())
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size.min(self.sequences.len()))
            .copied()
            .collect();

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for &idx in &batch_indices {
            let seq = &self.sequences[idx];
            if seq.len() > 1 {
                inputs.push(seq[..seq.len() - 1].to_vec());
                targets.push(seq[1..].to_vec());
            }
        }

        Some((inputs, targets))
    }

    pub fn len(&self) -> usize {
        self.sequences.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset() {
        let corpus = vec![AFRICA_GALORE_SAMPLE.to_string()];
        let tokenizer = SimpleTokenizer::from_corpus(&corpus, 2);
        
        let dataset = TextDataset::from_text(AFRICA_GALORE_SAMPLE, &tokenizer, 32);
        
        assert!(dataset.len() > 0);
        
        let batch = dataset.get_batch(4);
        assert!(batch.is_some());
    }
}
