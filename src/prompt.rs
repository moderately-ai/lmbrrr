use crate::image_processor::ProcessedImages;

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const IMAGE_START: &str = "<image>";
pub const IMAGE_END: &str = "</image>";
pub const IMAGE_ID_START: &str = "<image_id>";
pub const IMAGE_ID_END: &str = "</image_id>";
pub const SLICE_START: &str = "<slice>";
pub const SLICE_END: &str = "</slice>";

pub fn chat_prompt(user_prompt: &str, image_count: usize, enable_thinking: bool) -> String {
    let mut content = String::new();
    for _ in 0..image_count {
        content.push_str(IMAGE_PAD);
        content.push('\n');
    }
    content.push_str(user_prompt);
    let generation_prompt = if enable_thinking {
        "<think>\n"
    } else {
        "<think>\n\n</think>\n\n"
    };
    format!("{IM_START}user\n{content}{IM_END}\n{IM_START}assistant\n{generation_prompt}")
}

pub fn expand_image_placeholders(
    mut text: String,
    images: &ProcessedImages,
    use_image_id: bool,
    downsample_mode: &str,
) -> anyhow::Result<String> {
    let image_token_divisor = if downsample_mode == "4x" { 4 } else { 16 };
    let mut flat_index = 0usize;
    for (global_index, image_meta) in images.images.iter().enumerate() {
        if !text.contains(IMAGE_PAD) {
            anyhow::bail!(
                "prompt contains fewer image placeholders than image inputs: missing image {global_index}"
            );
        }

        let mut local = String::new();
        let target_sizes = &images.target_sizes[flat_index..flat_index + image_meta.patch_count];
        let source_tokens = target_sizes[0].0 * target_sizes[0].1 / image_token_divisor;
        if use_image_id {
            local.push_str(IMAGE_ID_START);
            local.push_str(&global_index.to_string());
            local.push_str(IMAGE_ID_END);
        }
        local.push_str(IMAGE_START);
        local.push_str(&IMAGE_PAD.repeat(source_tokens));
        local.push_str(IMAGE_END);

        let (num_rows, num_cols) = image_meta.grid;
        if num_rows > 0 && num_cols > 0 {
            let per_slice_tokens = target_sizes
                .get(1)
                .map(|(h, w)| h * w / image_token_divisor)
                .unwrap_or(0);
            let slice = format!(
                "{SLICE_START}{}{SLICE_END}",
                IMAGE_PAD.repeat(per_slice_tokens)
            );
            for row in 0..num_rows {
                if row == 0 {
                    local.push('\n');
                }
                for _ in 0..num_cols {
                    local.push_str(&slice);
                }
                if row + 1 < num_rows {
                    local.push('\n');
                }
            }
        }

        text = text.replacen(IMAGE_PAD, &local, 1);
        flat_index += image_meta.patch_count;
    }

    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_prompt_defaults_to_closed_thinking_block() {
        let prompt = chat_prompt("hello", 0, false);
        assert!(prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn chat_prompt_can_enable_thinking() {
        let prompt = chat_prompt("hello", 0, true);
        assert!(prompt.ends_with("<think>\n"));
        assert!(!prompt.ends_with("</think>\n\n"));
    }
}
