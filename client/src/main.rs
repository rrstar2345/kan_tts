use clap::Parser;
use reqwest::Client;
use shared::TtsRequest;
// use std::fs::File;
// use std::io::BufWriter;

#[derive(Parser, Debug)]
#[command(name = "kan_tts")]
#[command(about = "Kannada Text-to-Speech CLI", long_about = None)]
struct Args {
    /// Text to synthesize
    #[arg(short, long)]
    text: String,

    /// Output WAV file path
    #[arg(short, long)]
    output: String,

    /// Server address (default: http://127.0.0.1:8080)
    #[arg(short, long, default_value = "http://127.0.0.1:8080")]
    server: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("🎤 Kannada TTS Client");
    println!("Text: {}", args.text);
    println!("Output: {}", args.output);
    println!();

    // Create HTTP client
    let client = Client::new();

    // Send request to server
    let request = TtsRequest {
        text: args.text.clone(),
    };

    println!("📡 Connecting to server: {}", args.server);
    
    let response = client
        .post(format!("{}/synthesize", args.server))
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        eprintln!("✗ Server error: {}", response.status());
        return Err("HTTP request failed".into());
    }

    let tts_response = response.json::<shared::TtsResponse>().await?;

    if !tts_response.success {
        eprintln!("✗ Synthesis failed: {}", tts_response.message);
        return Err(tts_response.message.into());
    }

    // Save WAV file
    println!("💾 Saving audio to: {}", args.output);
    save_wav(&args.output, &tts_response.audio_data, tts_response.sample_rate)?;

    println!("✓ Audio saved successfully!");
    println!("Sample rate: {} Hz", tts_response.sample_rate);
    println!("Audio duration: {:.2} seconds", 
             tts_response.audio_data.len() as f32 / tts_response.sample_rate as f32);

    Ok(())
}

fn save_wav(path: &str, audio_data: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    // Convert f32 to i16 (scale to [-32768, 32767])
    for &sample in audio_data {
        let sample_i16 = (sample * 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}
