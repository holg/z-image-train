//! Model format conversion: safetensors ↔ .bpk
//!
//! Supports converting transformer, autoencoder, and text-encoder models
//! between HuggingFace safetensors format and Burn's native .bpk format.
//! Optionally casts weights to a different dtype (e.g. bf16 → f32).

use std::path::PathBuf;

use burn::{
    backend::NdArray,
    module::Module,
    store::{BurnpackStore, ModuleSnapshot, ModuleStore, SafetensorsStore, TensorSnapshot},
    tensor::{DType, FloatDType},
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
    pub dtype: Option<FloatDType>,
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

    // If dtype conversion is requested, work at the snapshot level to avoid
    // backend dtype limitations (NdArray/WGPU don't support bf16 tensors).
    if let Some(target_dtype) = args.dtype {
        return run_dtype_convert(args, target_dtype, &output_ext);
    }

    // Format-only conversion: load model on CPU and save in new format.
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    match args.model_type {
        ModelType::Transformer => {
            eprintln!("Initializing transformer model...");
            let mut model = ZImageModelConfig::default().init::<B>(&device);

            eprintln!("Loading weights from {}...", args.input.display());
            model
                .load_weights(&args.input)
                .map_err(|e| format!("Failed to load transformer: {e:?}"))?;

            save_model(&model, &args.output, &output_ext, args.overwrite)?;
        }
        ModelType::Autoencoder => {
            eprintln!("Initializing autoencoder model...");
            let mut model = AutoEncoderConfig::flux_ae().init_with_encoder::<B>(&device);

            eprintln!("Loading weights from {}...", args.input.display());
            model
                .load_weights(&args.input)
                .map_err(|e| format!("Failed to load autoencoder: {e:?}"))?;

            save_model(&model, &args.output, &output_ext, args.overwrite)?;
        }
        ModelType::TextEncoder => {
            eprintln!("Initializing text encoder model...");
            let mut model = Qwen3Config::z_image_text_encoder().init::<B>(&device);

            eprintln!("Loading weights from {}...", args.input.display());
            model
                .load_weights(&args.input)
                .map_err(|e| format!("Failed to load text encoder: {e:?}"))?;

            save_model(&model, &args.output, &output_ext, args.overwrite)?;
        }
    }

    eprintln!("Done!");
    Ok(())
}

/// Convert dtype at the snapshot level, bypassing backend limitations.
/// Reads raw tensor data from the input file, casts each tensor's bytes
/// using TensorData::convert_dtype (pure CPU, no backend needed), then
/// writes them out via BurnpackWriter.
fn run_dtype_convert(
    args: &ConvertArgs,
    target_dtype: FloatDType,
    output_ext: &str,
) -> Result<(), String> {
    let target_dt = match target_dtype {
        FloatDType::F64 => DType::F64,
        FloatDType::F32 | FloatDType::Flex32 => DType::F32,
        FloatDType::F16 => DType::F16,
        FloatDType::BF16 => DType::BF16,
    };

    // Read all snapshots from input
    eprintln!("Reading tensor snapshots...");
    let mut store = BurnpackStore::from_file(&args.input);
    let snapshots = store
        .get_all_snapshots()
        .map_err(|e| format!("Failed to read .bpk: {e}"))?
        .clone();

    let total = snapshots.len();
    eprintln!("Found {} tensors, casting to {:?}...", total, target_dt);

    // Cast each tensor's data and build new snapshots
    let mut converted: Vec<TensorSnapshot> = Vec::with_capacity(total);
    for (i, (_name, snapshot)) in snapshots.into_iter().enumerate() {
        if (i + 1) % 50 == 0 || i + 1 == total {
            eprint!("\r  Tensor {}/{}...", i + 1, total);
        }
        let data = snapshot
            .to_data()
            .map_err(|e| format!("Failed to read tensor: {e}"))?;
        let data = data.convert_dtype(target_dt);

        let path_stack = snapshot.path_stack.clone().unwrap_or_default();
        let container_stack = snapshot.container_stack.clone().unwrap_or_default();
        let tensor_id = snapshot.tensor_id.clone().unwrap_or_default();

        converted.push(TensorSnapshot::from_data(data, path_stack, container_stack, tensor_id));
    }
    eprintln!("\r  Cast {} tensors.          ", total);

    // Write out via BurnpackWriter
    eprintln!("Saving to {}...", args.output.display());
    match output_ext {
        "bpk" => {
            use burn::store::BurnpackWriter;
            let writer = BurnpackWriter::new(converted);
            writer
                .write_to_file(&args.output)
                .map_err(|e| format!("Failed to save .bpk: {e}"))?;
        }
        "safetensors" => {
            // For safetensors output, load into model and save
            return Err("Dtype conversion to .safetensors not yet supported. Use .bpk output.".into());
        }
        _ => {
            return Err(format!(
                "Unknown output format '.{output_ext}'. Use .bpk or .safetensors"
            ));
        }
    }

    eprintln!("Done!");
    Ok(())
}

fn save_model<M: Module<B> + ModuleSnapshot<B>>(
    model: &M,
    output: &PathBuf,
    output_ext: &str,
    overwrite: bool,
) -> Result<(), String> {
    eprintln!("Saving to {}...", output.display());
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
