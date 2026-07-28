use ort::{session::Session, value::TensorRef};
use ndarray::{Array2};
use std::path::Path;

pub struct KannadaTtsModel {
    session: Session,
    sample_rate: u32,
}

impl KannadaTtsModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        
        Ok(KannadaTtsModel {
            session,
            sample_rate: 24000, // MMS-TTS uses 24kHz
        })
    }

    pub fn synthesize(&self, token_ids: &[i64]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Create input tensor: shape (1, seq_len)
        let input_shape = vec![1, token_ids.len() as i64];
        let input_array = Array2::from_shape_vec(
            (1, token_ids.len()),
            token_ids.to_vec(),
        )?;
        
        // Run inference
        let outputs = self.session.run(ort::inputs![TensorRef::from_array_view(&input_array)?])?;

        // Extract waveform from output
        // MMS-TTS outputs "waveform" tensor of shape (1, audio_length)
        let waveform_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let waveform_view = waveform_tensor.view();
        
        // Flatten to 1D
        let waveform: Vec<f32> = waveform_view
            .into_iter()
            .cloned()
            .collect();

        Ok(waveform)
    }

    pub fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
