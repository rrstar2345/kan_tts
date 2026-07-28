use unicode_normalization::UnicodeNormalization;

/// Converts Kannada text to token IDs using character-level tokenization
/// This is a simplified tokenizer; for production, load proper tokenizer JSON
pub fn tokenize(text: &str) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    // Normalize text
    let normalized: String = text.nfc().collect();

    // Load tokenizer vocab (you'll need to create this)
    // For now, using character-level encoding
    let tokens: Vec<i64> = normalized
        .chars()
        .filter_map(|ch| {
            // Map characters to token IDs
            // This is a placeholder - you need proper vocab mapping
            Some(ch as i64)
        })
        .collect();

    if tokens.is_empty() {
        return Err("No tokens generated".into());
    }

    Ok(tokens)
}

/// Load tokenizer vocabulary from JSON file
pub fn load_vocab(path: &str) -> Result<std::collections::HashMap<String, i64>, Box<dyn std::error::Error>> {
    let vocab_content = std::fs::read_to_string(path)?;
    let vocab: std::collections::HashMap<String, i64> = serde_json::from_str(&vocab_content)?;
    Ok(vocab)
}
