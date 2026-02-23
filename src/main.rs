#![recursion_limit = "256"]
//! Z-Image Training CLI
//!
//! Standalone tool for LoRA fine-tuning and model format conversion.
//!
//! # Build
//!
//! ```sh
//! # macOS (Metal GPU):
//! cargo build --release --features metal
//!
//! # Linux/Windows (Vulkan GPU):
//! cargo build --release --features vulkan
//!
//! # Auto-detect GPU:
//! cargo build --release
//! ```
//!
//! # Usage
//!
//! ```sh
//! # Train LoRA
//! z-image-train train --model-dir /path/to/models --dataset /path/to/data --output lora.safetensors
//!
//! # Convert safetensors to .bpk
//! z-image-train convert --input model.safetensors --output model.bpk --model-type transformer
//!
//! # Generate with LoRA
//! z-image-train generate --model-dir /path/to/models --lora lora.safetensors --prompt "a cat"
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod convert;
mod dataset;
mod lora_io;
mod train;

use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};

type Backend = Wgpu<f32, i32>;
type TrainBackend = Autodiff<Backend>;

#[derive(Parser)]
#[command(name = "z-image-train", about = "Z-Image LoRA training and model conversion")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a LoRA adapter on a dataset of images + captions
    Train {
        /// Directory containing model weights (.bpk or .safetensors) and tokenizer.json
        #[arg(long)]
        model_dir: PathBuf,

        /// Directory containing training images + matching .txt caption files
        #[arg(long)]
        dataset: PathBuf,

        /// Output path for LoRA weights
        #[arg(long, default_value = "lora_weights.safetensors")]
        output: PathBuf,

        /// LoRA rank (low-rank dimension)
        #[arg(long, default_value_t = 16)]
        rank: usize,

        /// LoRA alpha scaling factor
        #[arg(long, default_value_t = 16.0)]
        alpha: f32,

        /// Number of training epochs
        #[arg(long, default_value_t = 100)]
        epochs: usize,

        /// Learning rate
        #[arg(long, default_value_t = 1e-4)]
        lr: f64,

        /// Training image resolution (images are resized to this)
        #[arg(long, default_value_t = 256)]
        resolution: usize,

        /// Save checkpoint every N steps (0 = only at end)
        #[arg(long, default_value_t = 0)]
        save_every: usize,
    },

    /// Convert model weights between safetensors and .bpk formats
    Convert {
        /// Input model file (.safetensors or .bpk)
        #[arg(long)]
        input: PathBuf,

        /// Output model file (.bpk or .safetensors)
        #[arg(long)]
        output: PathBuf,

        /// Model type: transformer, autoencoder, or text-encoder
        #[arg(long, default_value = "transformer")]
        model_type: String,

        /// Cast weights to dtype: f32, f16, or bf16
        #[arg(long)]
        dtype: Option<String>,

        /// Overwrite existing output file
        #[arg(long)]
        overwrite: bool,
    },

    /// Inspect a .bpk file: print tensor names, shapes, and dtypes
    Inspect {
        /// Path to the .bpk file to inspect
        #[arg(long)]
        input: PathBuf,

        /// Max number of tensors to display (0 = all)
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },

    /// Generate an image using base model + LoRA weights
    Generate {
        /// Directory containing model weights
        #[arg(long)]
        model_dir: PathBuf,

        /// Path to LoRA weights (.safetensors)
        #[arg(long)]
        lora: PathBuf,

        /// Text prompt
        #[arg(long)]
        prompt: String,

        /// Output image path
        #[arg(long, default_value = "output.png")]
        output: PathBuf,

        /// Image width
        #[arg(long, default_value_t = 512)]
        width: usize,

        /// Image height
        #[arg(long, default_value_t = 512)]
        height: usize,

        /// Number of inference steps
        #[arg(long, default_value_t = 8)]
        steps: usize,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Command::Train {
            model_dir,
            dataset,
            output,
            rank,
            alpha,
            epochs,
            lr,
            resolution,
            save_every,
        } => {
            let device = WgpuDevice::default();
            let args = train::TrainArgs {
                model_dir,
                dataset_dir: dataset,
                output,
                rank,
                alpha,
                epochs,
                lr,
                resolution,
                save_every,
            };
            train::run_training::<TrainBackend>(&args, &device)
        }

        Command::Convert {
            input,
            output,
            model_type,
            dtype,
            overwrite,
        } => {
            let model_type: convert::ModelType = match model_type.parse() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let dtype = match dtype.as_deref() {
                None => None,
                Some("f32") => Some(burn::tensor::FloatDType::F32),
                Some("f16") => Some(burn::tensor::FloatDType::F16),
                Some("bf16") => Some(burn::tensor::FloatDType::BF16),
                Some(d) => {
                    eprintln!("Error: Unknown dtype '{d}'. Use: f32, f16, or bf16");
                    return ExitCode::FAILURE;
                }
            };
            let args = convert::ConvertArgs {
                input,
                output,
                model_type,
                overwrite,
                dtype,
            };
            convert::run_convert(&args)
        }

        Command::Inspect { input, limit } => run_inspect(input, limit),

        Command::Generate {
            model_dir,
            lora,
            prompt,
            output,
            width,
            height,
            steps,
        } => run_generate(model_dir, lora, prompt, output, width, height, steps),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_inspect(input: PathBuf, limit: usize) -> Result<(), String> {
    use burn::store::{BurnpackStore, ModuleStore};
    use std::collections::HashMap;

    eprintln!("Inspecting: {}", input.display());

    let mut store = BurnpackStore::from_file(&input);
    let snapshots = store
        .get_all_snapshots()
        .map_err(|e| format!("Failed to read .bpk: {e}"))?
        .clone();

    let total = snapshots.len();
    eprintln!("Total tensors: {}\n", total);

    let mut dtype_counts: HashMap<String, usize> = HashMap::new();
    let display_count = if limit == 0 { total } else { limit.min(total) };

    for (i, (name, snapshot)) in snapshots.iter().enumerate() {
        let data: burn::tensor::TensorData = snapshot
            .to_data()
            .map_err(|e| format!("Failed to read tensor '{name}': {e}"))?;
        let dtype_str = format!("{:?}", data.dtype);
        *dtype_counts.entry(dtype_str.clone()).or_insert(0) += 1;

        if i < display_count {
            eprintln!("  [{:>4}] {:<80} {:?}  {:?}", i, name, data.shape, data.dtype);
        }
    }

    if display_count < total {
        eprintln!("  ... ({} more tensors, use --limit 0 to show all)", total - display_count);
    }

    eprintln!("\n--- Dtype Summary ---");
    for (dtype, count) in &dtype_counts {
        eprintln!("  {}: {} tensors", dtype, count);
    }

    Ok(())
}

fn run_generate(
    model_dir: PathBuf,
    lora_path: PathBuf,
    prompt: String,
    output: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
) -> Result<(), String> {
    use burn::tensor::{DType, Tensor};
    use qwen3_burn::{Qwen3Config, Qwen3Model, Qwen3Tokenizer};
    use z_image::modules::ae::{AutoEncoder, AutoEncoderConfig};
    use z_image::modules::transformer::{ZImageModel, ZImageModelConfig};
    use z_image::modules::transformer::lora_transformer::ZImageModelLora;

    let device = WgpuDevice::default();

    // Find model files
    let tokenizer_path = train::find_tokenizer_pub(&model_dir)?;
    let te_path = train::find_model_file_pub(&model_dir, &["qwen3_4b_text_encoder"])?;
    let transformer_path = train::find_model_file_pub(
        &model_dir,
        &["z_image_turbo_f16", "z_image_turbo_bf16", "z_image_turbo", "z_image"],
    )?;
    let ae_path = train::find_model_file_pub(&model_dir, &["ae"])?;

    // Load LoRA config
    eprintln!("Loading LoRA config from {}...", lora_path.display());
    let lora_config = lora_io::load_lora_config(&lora_path)?
        .ok_or_else(|| format!("No LoRA config found for {}", lora_path.display()))?;

    // Load tokenizer + text encoder
    eprintln!("Loading tokenizer...");
    let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    eprintln!("Loading text encoder...");
    let mut text_encoder: Qwen3Model<Backend> =
        Qwen3Config::z_image_text_encoder().init(&device);
    text_encoder
        .load_weights(&te_path)
        .map_err(|e| format!("Failed to load text encoder: {e:?}"))?;

    // Compute text embedding
    eprintln!("Computing text embedding for: \"{}\"", prompt);
    let prompt_embedding =
        z_image::compute_prompt_embedding(&prompt, &tokenizer, &text_encoder, &device)
            .map_err(|e| format!("Failed to compute embedding: {e:?}"))?;

    drop(text_encoder);
    drop(tokenizer);

    // Load transformer + LoRA
    eprintln!("Loading transformer...");
    let mut transformer: ZImageModel<Backend> = ZImageModelConfig::default().init(&device);
    transformer
        .load_weights(&transformer_path)
        .map_err(|e| format!("Failed to load transformer: {e:?}"))?;

    eprintln!("Applying LoRA weights...");
    let mut lora_model = ZImageModelLora::from_base(transformer, &lora_config, &device);
    lora_io::load_lora_weights(&mut lora_model, &lora_path)?;

    // Merge LoRA into base for fast inference
    // (For now, just use the LoRA model directly - merge can be added later)

    // Load autoencoder (decoder only)
    eprintln!("Loading autoencoder...");
    let mut ae: AutoEncoder<Backend> = AutoEncoderConfig::flux_ae().init(&device);
    ae.load_weights(&ae_path)
        .map_err(|e| format!("Failed to load autoencoder: {e:?}"))?;

    // Generate
    eprintln!("Generating {}x{} image with {} steps...", width, height, steps);

    let vae_scale = 16;
    if width % vae_scale != 0 || height % vae_scale != 0 {
        return Err(format!("Width and height must be a multiple of {vae_scale}"));
    }

    let latent_width = 2 * (width / vae_scale);
    let latent_height = 2 * (height / vae_scale);
    let latents_shape = [1, 16, latent_height, latent_width];

    let latents = Tensor::<Backend, 4>::random(
        latents_shape,
        burn::tensor::Distribution::Normal(0., 1.),
        &device,
    );

    let image_seq_len = (latents_shape[2] / 2) * (latents_shape[3] / 2);
    let mu = {
        let m = (1.15 - 0.5) / (4096.0 - 256.0);
        let b = 0.5 - m * 256.0;
        image_seq_len as f32 * m + b
    };

    let mut scheduler =
        z_image::scheduler::FlowMatchEulerDiscreteScheduler::<Backend>::new(1000, 3.0, false, &device);
    scheduler.set_timesteps(Some(steps), &device, None, Some(mu.into()), None);

    let timesteps = scheduler.timesteps();
    let num_steps = timesteps.dims()[0];
    let timesteps_vec: Vec<f32> = timesteps
        .into_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();

    let mut latents = latents;
    for (i, &t) in timesteps_vec.iter().enumerate() {
        if t == 0. && i == num_steps - 1 {
            continue;
        }
        eprint!("\r  Step {}/{}...", i + 1, num_steps);
        let timestep =
            Tensor::<Backend, 1>::from_floats([t], &device).expand([latents_shape[0]]);
        let timestep: Tensor<Backend, 1> = timestep / 1000.;
        let noise_pred =
            lora_model.forward(latents.clone(), timestep, prompt_embedding.clone());
        latents = scheduler.step(-noise_pred, t, latents);
    }
    eprintln!("\r  Generation complete.        ");

    let latents_f32 = latents.cast(DType::F32);
    let image = ae.decode(latents_f32);

    use burn::vision::utils::{
        ColorDisplayOpts, ImageDimOrder, TensorDisplayOptions, save_tensor_as_image,
    };
    save_tensor_as_image(
        image.cast(DType::F32),
        TensorDisplayOptions {
            dim_order: ImageDimOrder::Nchw,
            color_opts: ColorDisplayOpts::Rgb,
            batch_opts: None,
            width_out: width,
            height_out: height,
        },
        &output,
    )
    .map_err(|e| format!("Failed to save image: {e}"))?;

    eprintln!("Image saved to {}", output.display());
    Ok(())
}
