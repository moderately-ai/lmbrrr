use std::path::PathBuf;

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use image::{imageops::FilterType, DynamicImage, RgbImage};

use crate::config::PreprocessorConfig;

#[derive(Clone, Debug)]
pub struct ImageMeta {
    pub path: PathBuf,
    pub grid: (usize, usize),
    pub patch_count: usize,
}

#[derive(Debug)]
pub struct ProcessedImages {
    pub pixel_values: Tensor,
    pub target_sizes: Vec<(usize, usize)>,
    pub images: Vec<ImageMeta>,
}

pub fn preprocess_paths(
    paths: &[PathBuf],
    cfg: &PreprocessorConfig,
    device: &Device,
) -> Result<ProcessedImages> {
    let images = paths
        .iter()
        .map(|path| {
            let image = image::open(path)
                .with_context(|| format!("open image {}", path.display()))?
                .to_rgb8();
            Ok((path.clone(), image))
        })
        .collect::<Result<Vec<_>>>()?;
    preprocess_rgb_images(&images, cfg, device)
}

pub fn preprocess_rgb_images(
    inputs: &[(PathBuf, RgbImage)],
    cfg: &PreprocessorConfig,
    device: &Device,
) -> Result<ProcessedImages> {
    let mut all_patch_values = Vec::new();
    let mut all_target_sizes = Vec::new();
    let mut images = Vec::new();

    for (path, image) in inputs {
        let image_size = (image.height() as usize, image.width() as usize);
        let best_grid = if cfg.slice_mode {
            get_sliced_grid(image_size, cfg.max_slice_nums, cfg.scale_resolution)
        } else {
            None
        };

        let (source_h, source_w) = find_best_resize(
            image_size,
            cfg.scale_resolution,
            cfg.patch_size,
            best_grid.is_none(),
        );
        let source_img = resize_rgb(&image, source_h, source_w);

        let mut patches = vec![source_img];
        let mut patch_height = 0usize;
        let mut patch_width = 0usize;
        if let Some(grid) = best_grid {
            let (refine_h, refine_w) =
                get_refine_size(image_size, grid, cfg.scale_resolution, cfg.patch_size);
            let refine_img = resize_rgb(&image, refine_h, refine_w);
            let grid_y = grid.0;
            let grid_x = grid.1;
            patch_height = refine_h / grid_y;
            patch_width = refine_w / grid_x;
            patches.extend(divide_to_patches(&refine_img, patch_height, patch_width));
        }

        let first_index = all_patch_values.len();
        for (idx, patch) in patches.iter().enumerate() {
            let chw = normalize_rgb(patch, cfg.image_mean, cfg.image_std);
            let packed = reshape_by_patch(
                &chw,
                patch.height() as usize,
                patch.width() as usize,
                cfg.patch_size,
            );
            all_patch_values.push(packed);
            if idx == 0 {
                all_target_sizes.push((source_h / cfg.patch_size, source_w / cfg.patch_size));
            } else {
                all_target_sizes
                    .push((patch_height / cfg.patch_size, patch_width / cfg.patch_size));
            }
        }

        images.push(ImageMeta {
            path: path.clone(),
            grid: best_grid.unwrap_or((0, 0)),
            patch_count: all_patch_values.len() - first_index,
        });
    }

    let total_width = all_patch_values
        .iter()
        .map(|values| values.len() / (3 * cfg.patch_size))
        .sum::<usize>();
    let mut packed = Vec::with_capacity(3 * cfg.patch_size * total_width);
    for c in 0..3 {
        for row in 0..cfg.patch_size {
            for values in &all_patch_values {
                let width = values.len() / (3 * cfg.patch_size);
                let start = (c * cfg.patch_size + row) * width;
                packed.extend_from_slice(&values[start..start + width]);
            }
        }
    }
    let pixel_values = Tensor::from_vec(packed, (1, 3, cfg.patch_size, total_width), device)?;

    Ok(ProcessedImages {
        pixel_values,
        target_sizes: all_target_sizes,
        images,
    })
}

fn resize_rgb(image: &RgbImage, height: usize, width: usize) -> RgbImage {
    DynamicImage::ImageRgb8(image.clone())
        .resize_exact(width as u32, height as u32, FilterType::CatmullRom)
        .to_rgb8()
}

fn normalize_rgb(image: &RgbImage, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let mut out = vec![0f32; 3 * width as usize * height as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                let value = pixel[c] as f32 / 255.0;
                out[c * width as usize * height as usize + y * width as usize + x] =
                    (value - mean[c]) / std[c];
            }
        }
    }
    out
}

fn divide_to_patches(image: &RgbImage, patch_height: usize, patch_width: usize) -> Vec<RgbImage> {
    let (_, height) = image.dimensions();
    let (width, _) = image.dimensions();
    let mut patches = Vec::new();
    for y in (0..height as usize).step_by(patch_height) {
        for x in (0..width as usize).step_by(patch_width) {
            patches.push(
                image::imageops::crop_imm(
                    image,
                    x as u32,
                    y as u32,
                    patch_width as u32,
                    patch_height as u32,
                )
                .to_image(),
            );
        }
    }
    patches
}

fn reshape_by_patch(chw: &[f32], height: usize, width: usize, patch_size: usize) -> Vec<f32> {
    let blocks_h = height / patch_size;
    let blocks_w = width / patch_size;
    let num_patches = blocks_h * blocks_w;
    let out_width = num_patches * patch_size;
    let mut out = vec![0f32; 3 * patch_size * out_width];

    for c in 0..3 {
        for block_y in 0..blocks_h {
            for block_x in 0..blocks_w {
                let patch_idx = block_y * blocks_w + block_x;
                for patch_y in 0..patch_size {
                    for patch_x in 0..patch_size {
                        let src_y = block_y * patch_size + patch_y;
                        let src_x = block_x * patch_size + patch_x;
                        let src = c * height * width + src_y * width + src_x;
                        let dst = (c * patch_size + patch_y) * out_width
                            + patch_idx * patch_size
                            + patch_x;
                        out[dst] = chw[src];
                    }
                }
            }
        }
    }
    out
}

fn find_best_resize(
    image_size: (usize, usize),
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let (mut height, mut width) = image_size;
    if height * width > scale_resolution * scale_resolution || allow_upscale {
        let aspect_ratio = width as f64 / height as f64;
        height = (scale_resolution as f64 / aspect_ratio.sqrt()) as usize;
        width = (height as f64 * aspect_ratio) as usize;
    }
    let divisor = patch_size * 4;
    (
        ensure_divide(height, divisor),
        ensure_divide(width, divisor),
    )
}

fn get_refine_size(
    image_size: (usize, usize),
    grid: (usize, usize),
    scale_resolution: usize,
    patch_size: usize,
) -> (usize, usize) {
    let (height, width) = image_size;
    let (grid_y, grid_x) = grid;
    let refine_width = ensure_divide(width, grid_x);
    let refine_height = ensure_divide(height, grid_y);
    let (best_height, best_width) = find_best_resize(
        (refine_height / grid_y, refine_width / grid_x),
        scale_resolution,
        patch_size,
        true,
    );
    (best_height * grid_y, best_width * grid_x)
}

fn get_sliced_grid(
    image_size: (usize, usize),
    max_slice_nums: usize,
    scale_resolution: usize,
) -> Option<(usize, usize)> {
    let (original_height, original_width) = image_size;
    let log_ratio = (original_width as f64 / original_height as f64).ln();
    let ratio = original_width as f64 * original_height as f64
        / (scale_resolution * scale_resolution) as f64;
    let multiple = ratio.ceil().min(max_slice_nums as f64) as usize;
    if multiple <= 1 {
        return None;
    }

    let mut best_grid = (1usize, 1usize);
    let mut min_error = f64::INFINITY;
    for num_slices in [multiple.saturating_sub(1), multiple, multiple + 1] {
        if num_slices <= 1 || num_slices > max_slice_nums {
            continue;
        }
        for num_rows in 1..=num_slices {
            if num_slices % num_rows == 0 {
                let num_cols = num_slices / num_rows;
                let error = (log_ratio - (num_rows as f64 / num_cols as f64).ln()).abs();
                if error < min_error {
                    best_grid = (num_cols, num_rows);
                    min_error = error;
                }
            }
        }
    }
    Some(best_grid)
}

fn ensure_divide(length: usize, divisor: usize) -> usize {
    let rounded = ((length as f64 / divisor as f64).round() as usize) * divisor;
    rounded.max(divisor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_is_patch_and_merge_aligned() {
        assert_eq!(find_best_resize((100, 200), 448, 14, false), (112, 224));
        assert_eq!(ensure_divide(447, 56), 448);
    }

    #[test]
    fn patch_reshape_matches_expected_shape() {
        let chw = vec![1f32; 3 * 28 * 42];
        let packed = reshape_by_patch(&chw, 28, 42, 14);
        assert_eq!(packed.len(), 3 * 14 * (28 * 42 / 14));
    }
}
