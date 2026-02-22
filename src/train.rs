//! LoRA training pipeline for Z-Image.
//!
//! Orchestrates the full training flow:
//! 1. Load all models (text encoder, VAE, transformer)
//! 2. Pre-compute latent cache
//! 3. Free encoders from VRAM
//! 4. Wrap transformer in LoRA
//! 5. Run flow matching training loop
//! 6. Save LoRA weights

use std::path::{Path, PathBuf};

use burn::{
    Tensor,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{Distribution, backend::AutodiffBackend},
};
use qwen3_burn::{Qwen3Config, Qwen3Model, Qwen3Tokenizer};
use z_image::modules::{
    ae::{AutoEncoder, AutoEncoderConfig},
    transformer::{ZImageModel, ZImageModelConfig, lora_transformer::{LoraConfig, ZImageModelLora}},
};

use crate::dataset::TrainingDataset;
use crate::lora_io;

/// Training arguments (from CLI).
pub struct TrainArgs {
    pub model_dir: PathBuf,
    pub dataset_dir: PathBuf,
    pub output: PathBuf,
    pub rank: usize,
    pub alpha: f32,
    pub epochs: usize,
    pub lr: f64,
    pub resolution: usize,
    pub save_every: usize,
}

/// Find a model file in the model directory, preferring .bpk over .safetensors.
pub fn find_model_file_pub(model_dir: &Path, base_names: &[&str]) -> Result<PathBuf, String> {
    find_model_file(model_dir, base_names)
}

/// Find tokenizer (public for use from main.rs).
pub fn find_tokenizer_pub(model_dir: &Path) -> Result<PathBuf, String> {
    find_tokenizer(model_dir)
}

fn find_model_file(model_dir: &Path, base_names: &[&str]) -> Result<PathBuf, String> {
    for base in base_names {
        let bpk = model_dir.join(format!("{base}.bpk"));
        if bpk.exists() {
            return Ok(bpk);
        }
        let st = model_dir.join(format!("{base}.safetensors"));
        if st.exists() {
            return Ok(st);
        }
    }
    Err(format!(
        "Could not find model file in {}: tried {:?} with .bpk/.safetensors extensions",
        model_dir.display(),
        base_names
    ))
}

fn find_tokenizer(model_dir: &Path) -> Result<PathBuf, String> {
    for name in ["qwen3-tokenizer.json", "tokenizer.json"] {
        let path = model_dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(format!(
        "Could not find tokenizer.json in {}",
        model_dir.display()
    ))
}

/// Pre-computed latent cache item.
struct CachedItem<B: Backend> {
    latent: Tensor<B, 4>,
    text_embedding: Tensor<B, 3>,
}

/// Pre-compute all latents and text embeddings on a given backend.
fn precompute_cache<B: Backend>(
    dataset: &TrainingDataset,
    autoencoder: &AutoEncoder<B>,
    tokenizer: &Qwen3Tokenizer,
    text_encoder: &Qwen3Model<B>,
    resolution: usize,
    device: &B::Device,
) -> Result<Vec<CachedItem<B>>, String> {
    use burn::tensor::Int;
    use crate::dataset::load_image_tensor;

    let total = dataset.len();
    let mut items = Vec::with_capacity(total);

    for (i, item) in dataset.items.iter().enumerate() {
        eprint!(
            "\r  Caching [{}/{}] {}...",
            i + 1,
            total,
            item.image_path.file_name().unwrap_or_default().to_string_lossy()
        );

        let image_tensor = load_image_tensor::<B>(
            &item.image_path,
            resolution as u32,
            resolution as u32,
            device,
        )?;
        let latent = autoencoder.encode(image_tensor);

        // Compute text embedding
        let (input_ids_vec, attention_mask_vec) = tokenizer
            .encode_prompt(&item.caption)
            .map_err(|e| format!("Tokenization error: {e}"))?;

        let seq_len = input_ids_vec.len();
        let input_ids = Tensor::<B, 1, Int>::from_data(input_ids_vec.as_slice(), device)
            .reshape([1, seq_len]);
        let attention_mask = Tensor::<B, 1>::from_data(
            attention_mask_vec
                .iter()
                .map(|&b| if b { 1.0f32 } else { 0.0f32 })
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        )
        .greater_elem(0.5)
        .reshape([1, seq_len]);

        let text_embedding = text_encoder.encode(input_ids, attention_mask.clone());
        let text_embedding =
            z_image::extract_valid_embeddings(text_embedding, attention_mask);

        items.push(CachedItem {
            latent,
            text_embedding,
        });
    }

    eprintln!("\r  Cached {total}/{total} items.                    ");
    Ok(items)
}

/// Run the full training pipeline.
///
/// Uses B::InnerBackend for model loading and cache precomputation (no gradient tracking),
/// then wraps the transformer in LoRA on the AutodiffBackend for training.
pub fn run_training<B: AutodiffBackend>(args: &TrainArgs, device: &B::Device) -> Result<(), String> {
    eprintln!("=== Z-Image LoRA Training ===");
    eprintln!("  Model dir:  {}", args.model_dir.display());
    eprintln!("  Dataset:    {}", args.dataset_dir.display());
    eprintln!("  Output:     {}", args.output.display());
    eprintln!("  LoRA rank:  {}, alpha: {}", args.rank, args.alpha);
    eprintln!("  Epochs:     {}, LR: {}", args.epochs, args.lr);
    eprintln!("  Resolution: {}x{}", args.resolution, args.resolution);
    eprintln!();

    // Note: Autodiff<B>::Device = B::Device, so `device` works for both backends.

    // Load dataset
    eprintln!("[1/7] Loading dataset...");
    let dataset = TrainingDataset::from_directory(&args.dataset_dir)?;
    eprintln!("  Found {} image+caption pairs", dataset.len());

    // Find model files
    let tokenizer_path = find_tokenizer(&args.model_dir)?;
    let te_path = find_model_file(&args.model_dir, &["qwen3_4b_text_encoder"])?;
    let transformer_path = find_model_file(
        &args.model_dir,
        &["z_image_turbo_bf16", "z_image_turbo", "z_image"],
    )?;
    let ae_path = find_model_file(&args.model_dir, &["ae"])?;

    eprintln!("  Tokenizer:   {}", tokenizer_path.display());
    eprintln!("  Text encoder: {}", te_path.display());
    eprintln!("  Transformer:  {}", transformer_path.display());
    eprintln!("  Autoencoder:  {}", ae_path.display());

    // Load tokenizer
    eprintln!("\n[2/7] Loading tokenizer...");
    let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

    // Load text encoder on the autodiff backend (forward-only, no grads needed)
    eprintln!("[3/7] Loading text encoder...");
    let mut text_encoder: Qwen3Model<B> = Qwen3Config::z_image_text_encoder().init(device);
    text_encoder
        .load_weights(&te_path)
        .map_err(|e| format!("Failed to load text encoder: {e:?}"))?;
    eprintln!("  Text encoder loaded.");

    // Load autoencoder with encoder
    eprintln!("[4/7] Loading autoencoder (with encoder)...");
    let mut autoencoder: AutoEncoder<B> = AutoEncoderConfig::flux_ae().init_with_encoder(device);
    autoencoder
        .load_weights(&ae_path)
        .map_err(|e| format!("Failed to load autoencoder: {e:?}"))?;
    eprintln!("  Autoencoder loaded.");

    // Pre-compute latent cache
    eprintln!("[5/7] Pre-computing latent cache...");
    let cache = precompute_cache(
        &dataset,
        &autoencoder,
        &tokenizer,
        &text_encoder,
        args.resolution,
        device,
    )?;

    // Free encoder models
    drop(text_encoder);
    drop(autoencoder);
    drop(tokenizer);
    eprintln!("  Freed text encoder and autoencoder from memory.");

    // Load transformer and wrap in LoRA
    eprintln!("[6/7] Loading transformer and wrapping with LoRA...");
    let mut transformer: ZImageModel<B> = ZImageModelConfig::default().init(device);
    transformer
        .load_weights(&transformer_path)
        .map_err(|e| format!("Failed to load transformer: {e:?}"))?;

    let lora_config = LoraConfig {
        rank: args.rank,
        alpha: args.alpha,
        target_attention: true,
        target_feed_forward: true,
        target_refiners: true,
    };

    let lora_model = ZImageModelLora::from_base(transformer, &lora_config, device);
    eprintln!(
        "  LoRA model ready. Trainable params: {}",
        lora_model.lora_param_count()
    );

    // Run training loop
    eprintln!("[7/7] Training...");
    let lora_model = train_loop(lora_model, &cache, args, device)?;

    // Save LoRA weights
    eprintln!("\nSaving LoRA weights to {}...", args.output.display());
    lora_io::save_lora_weights(&lora_model, &lora_config, &args.output)?;
    eprintln!("Done! LoRA weights saved.");

    Ok(())
}

/// Flow matching training loop.
fn train_loop<B: AutodiffBackend>(
    mut lora_model: ZImageModelLora<B>,
    cache: &[CachedItem<B>],
    args: &TrainArgs,
    device: &B::Device,
) -> Result<ZImageModelLora<B>, String> {
    let steps_per_epoch = cache.len();
    let total_steps = args.epochs * steps_per_epoch;

    eprintln!(
        "  {} epochs x {} steps = {} total steps",
        args.epochs, steps_per_epoch, total_steps
    );

    let mut optim = AdamWConfig::new()
        .with_weight_decay(1e-2)
        .init::<B, ZImageModelLora<B>>();

    let mut global_step = 0;

    for epoch in 0..args.epochs {
        let mut epoch_loss = 0.0f64;

        for (step, item) in cache.iter().enumerate() {
            let image_latent = item.latent.clone();
            let text_embedding = item.text_embedding.clone();

            // Sample random timestep
            let t_val: f32 =
                Tensor::<B, 1>::random([1], Distribution::Uniform(0.0, 1.0), device)
                    .into_data()
                    .as_slice::<f32>()
                    .expect("f32 slice")[0];

            // Sample noise
            let latent_dims = image_latent.dims();
            let noise: Tensor<B, 4> =
                Tensor::random(latent_dims, Distribution::Normal(0.0, 1.0), device);

            // Noisy latents: (1-t)*noise + t*latent
            let noisy_latents =
                noise.clone() * (1.0 - t_val) + image_latent.clone() * t_val;

            // Velocity target: latent - noise
            let v_target = image_latent - noise;

            // Forward pass
            let timestep = Tensor::<B, 1>::from_floats([t_val], device);
            let v_predicted = lora_model.forward(noisy_latents, timestep, text_embedding);

            // MSE loss
            let diff = v_predicted - v_target;
            let loss = (diff.clone() * diff).mean();

            let loss_val = loss
                .clone()
                .into_data()
                .as_slice::<f32>()
                .expect("f32 slice")[0];

            // Backward + optimizer step
            let grads = GradientsParams::from_grads(loss.backward(), &lora_model);
            lora_model = optim.step(args.lr, lora_model, grads);

            epoch_loss += loss_val as f64;
            global_step += 1;

            eprint!(
                "\r  Epoch {}/{} | Step {}/{} | Loss: {:.6}    ",
                epoch + 1,
                args.epochs,
                step + 1,
                steps_per_epoch,
                loss_val
            );

            // Checkpoint save
            if args.save_every > 0 && global_step % args.save_every == 0 {
                let checkpoint_path =
                    args.output.with_extension(format!("step{global_step}.safetensors"));
                eprintln!("\n  Saving checkpoint: {}", checkpoint_path.display());
                let config = LoraConfig {
                    rank: args.rank,
                    alpha: args.alpha,
                    target_attention: true,
                    target_feed_forward: true,
                    target_refiners: true,
                };
                lora_io::save_lora_weights(&lora_model, &config, &checkpoint_path)?;
            }
        }

        let avg_loss = epoch_loss / steps_per_epoch as f64;
        eprintln!(
            "\r  Epoch {}/{} complete | Avg loss: {:.6}                    ",
            epoch + 1,
            args.epochs,
            avg_loss
        );
    }

    Ok(lora_model)
}
