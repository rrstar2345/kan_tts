use std::path::Path;
use tokenizers::Tokenizer;

/// Kannada tokenizer backed by the Hugging Face tokenizer.json.
///
/// This uses the exact tokenizer configuration that the MMS model was
/// trained with, producing identical token IDs to the Python
/// transformers/tokenizers implementation.
pub struct KannadaTokenizer {
    tokenizer: Tokenizer,
}

impl KannadaTokenizer {
    /// Load tokenizer from tokenizer.json.
    pub fn new<P: AsRef<Path>>(
        tokenizer_path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

        Ok(Self { tokenizer })
    }

    /// Convert text into model input IDs.
    pub fn tokenize(
        &self,
        text: &str,
    ) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Tokenization failed: {e}"))?;

        let ids = encoding
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect::<Vec<_>>();

        if ids.is_empty() {
            return Err("Tokenizer produced no tokens".into());
        }

        Ok(ids)
    }

    #[allow(dead_code)]
    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    #[allow(dead_code)]
    /// Decode token IDs back to text (useful for testing).
    pub fn decode(
        &self,
        ids: &[i64],
    ) -> Result<String, Box<dyn std::error::Error>> {
        let ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();

        self.tokenizer
            .decode(&ids, true)
            .map_err(|e| format!("Decode failed: {e}").into())
    }
}