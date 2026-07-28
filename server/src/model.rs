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
            sample_rate: 16000, // per config.json "sampling_rate" for mms-tts-kan
        })
    }

    pub fn synthesize(
        &mut self,
        token_ids: &[i64],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let seq_len = token_ids.len();
        tracing::info!("synthesize: seq_len={} token_ids={:?}", seq_len, token_ids);

        // Create input tensor: shape (1, seq_len)
        let input_array = Array2::from_shape_vec(
            (1, seq_len),
            token_ids.to_vec(),
        )?;

        // Attention mask: all ones, same shape as input_ids
        let attention_mask_array = Array2::from_shape_vec(
            (1, seq_len),
            vec![1i64; seq_len],
        )?;

        tracing::info!(
            "synthesize: input_ids shape={:?} attention_mask shape={:?}",
            input_array.shape(),
            attention_mask_array.shape()
        );

        for outlet in self.session.inputs().iter() {
            tracing::info!("model expects input: {:?}", outlet);
        }

        // Run inference
        let outputs = self.session.run(
            ort::inputs![
                "input_ids" => TensorRef::from_array_view(&input_array)?,
                "attention_mask" => TensorRef::from_array_view(&attention_mask_array)?,
            ]
        )?;

        // Output is (&Shape, &[f32])
        let (shape, waveform) = outputs[0].try_extract_tensor::<f32>()?;
        tracing::info!("synthesize: output shape={:?} len={}", shape, waveform.len());

        Ok(waveform.to_vec())
    }

    pub fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }
}