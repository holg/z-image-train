//! Model format conversion: safetensors ↔ .bpk
//!
//! Supports converting transformer, autoencoder, and text-encoder models
//! between HuggingFace safetensors format and Burn's native .bpk format.

use std::path::PathBuf;

use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    store::{BurnpackStore, ModuleSnapshot, SafetensorsStore},
};
use qwen3_burn::Qwen3Config;
use z_image::modules::{
    ae::AutoEncoderConfig,
    transformer::ZImageModelConfig,
};

type B = NdArray;

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    Transformer,
    Autoencoder,
    TextEncoder,
}

impl std::str::FromStr for ModelType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "transformer" | "t" => Ok(ModelType::Transformer),
            "autoencoder" | "ae" | "vae" => Ok(ModelType::Autoencoder),
            "text-encoder" | "te" | "text_encoder" => Ok(ModelType::TextEncoder),
            _ => Err(format!(
                "Unknown model type '{s}'. Use: transformer, autoencoder, or text-encoder"
            )),
        }
    }
}

/// Convert arguments (from CLI).
pub struct ConvertArgs {
    pub input: PathBuf,
    pub output: PathBuf,
    pub model_type: ModelType,
    pub overwrite: bool,
}

/// Run model format conversion.
pub fn run_convert(args: &ConvertArgs) -> Result<(), String> {
    let output_ext = args
        .output
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    eprintln!(
        "Converting {:?}: {} -> {}",
        args.model_type,
        args.input.display(),
        args.output.display()
    );

    if !args.input.exists() {
        return Err(format!("Input file not found: {}", args.input.display()));
    }

    if args.output.exists() && !args.overwrite {
        return Err(format!(
            "Output file already exists: {}. Use --overwrite to overwrite.",
            args.output.display()
        ));
    }

    let device = NdArrayDevice::Cpu;

    match args.model_type {
        ModelType::Transformer => {
            eprintln!("Initializing transformer model...");
            let mut model = ZImageModelConfig::default().init::<B>(&device);

            eprintln!("Loading weights from {}...", args.input.display());
            model
                .load_weights(&args.input)
                .map_err(|e| format!("Failed to load transformer: {e:?}"))?;

            eprintln!("Saving to {}...", args.output.display());
            save_model(&model, &args.output, &output_ext, args.overwrite)?;
        }
        ModelType::Autoencoder => {
            eprintln!("Initializing autoencoder model...");
            let mut model = AutoEncoderConfig::flux_ae().init_with_encoder::<B>(&device);

            eprintln!("Loading weights from {}...", args.input.display());
            model
                .load_weights(&args.input)
                .map_err(|e| format!("Failed to load autoencoder: {e:?}"))?;

            eprintln!("Saving to {}...", args.output.display());
            save_model(&model, &args.output, &output_ext, args.overwrite)?;
        }
        ModelType::TextEncoder => {
            eprintln!("Initializing text encoder model...");
            let mut model = Qwen3Config::z_image_text_encoder().init::<B>(&device);

            eprintln!("Loading weights from {}...", args.input.display());
            model
                .load_weights(&args.input)
                .map_err(|e| format!("Failed to load text encoder: {e:?}"))?;

            eprintln!("Saving to {}...", args.output.display());
            save_model(&model, &args.output, &output_ext, args.overwrite)?;
        }
    }

    eprintln!("Done!");
    Ok(())
}

fn save_model<M: burn::module::Module<B> + ModuleSnapshot<B>>(
    model: &M,
    output: &PathBuf,
    output_ext: &str,
    overwrite: bool,
) -> Result<(), String> {
    match output_ext {
        "bpk" => {
            let mut store = BurnpackStore::from_file(output).overwrite(overwrite);
            model
                .save_into(&mut store)
                .map_err(|e| format!("Failed to save .bpk: {e}"))?;
        }
        "safetensors" => {
            let mut store = SafetensorsStore::from_file(output).overwrite(overwrite);
            model
                .save_into(&mut store)
                .map_err(|e| format!("Failed to save .safetensors: {e}"))?;
        }
        _ => {
            return Err(format!(
                "Unknown output format '.{output_ext}'. Use .bpk or .safetensors"
            ));
        }
    }
    Ok(())
}
