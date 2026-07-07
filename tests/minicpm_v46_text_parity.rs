use lmbrrr::prompt::chat_prompt;
use lmbrrr::{
    image_processor::{ImageMeta, ProcessedImages},
    prompt::expand_image_placeholders,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "openbmb/MiniCPM-V-4.6";
const REVISION: &str = "main";
const TOKENIZER_PATH: &str = "docs/research/models/minicpm-v-4.6/hf-model/tokenizer.json";
const FIXTURE: &str = include_str!("../evals/fixtures/minicpm_v46_text_prompts.json");
const TRANSFORMERS_TEXT_LOGITS_FIXTURE: &str =
    include_str!("../evals/fixtures/minicpm_v46_transformers_text_logits.json");
const TRANSFORMERS_IMAGE_EXPANSION_FIXTURE: &str =
    include_str!("../evals/fixtures/minicpm_v46_transformers_image_expansion.json");

#[derive(Clone, Copy, Debug)]
struct PromptCase {
    id: &'static str,
    user_prompt: &'static str,
    image_count: usize,
    enable_thinking: bool,
}

#[derive(Debug, Deserialize)]
struct PromptFixture {
    schema_version: u32,
    model_id: String,
    revision: String,
    tokenizer_path: String,
    cases: Vec<PromptFixtureCase>,
}

#[derive(Debug, Deserialize)]
struct PromptFixtureCase {
    id: String,
    user_prompt: String,
    image_count: usize,
    enable_thinking: bool,
    rendered_prompt: String,
    prompt_token_count: usize,
    token_ids: Vec<u32>,
}

fn prompt_cases() -> Vec<PromptCase> {
    vec![
        PromptCase {
            id: "text_closed_thinking_short",
            user_prompt: "What is the capital of France?",
            image_count: 0,
            enable_thinking: false,
        },
        PromptCase {
            id: "text_open_thinking_math",
            user_prompt: "Solve 17 * 23. Think carefully.",
            image_count: 0,
            enable_thinking: true,
        },
        PromptCase {
            id: "text_closed_thinking_long_reasoning",
            user_prompt: "Solve this carefully. A lab runs three model-evaluation batches. Batch A has 18 prompts and each prompt takes 7 seconds. Batch B has twice as many prompts, but each prompt takes 5 seconds. Batch C has 12 prompts, each taking 11 seconds, and can only start after Batch A finishes. If Batch A and Batch B start together, what is the earliest time when all three batches are complete?",
            image_count: 0,
            enable_thinking: false,
        },
        PromptCase {
            id: "single_image_closed_thinking",
            user_prompt: "What causes this phenomenon?",
            image_count: 1,
            enable_thinking: false,
        },
    ]
}

fn load_tokenizer() -> Tokenizer {
    Tokenizer::from_file(TOKENIZER_PATH).expect("load vendored MiniCPM tokenizer")
}

#[test]
fn minicpm_v46_text_fixture_matches_runner() {
    let fixture: PromptFixture = serde_json::from_str(FIXTURE).expect("parse prompt fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.model_id, MODEL_ID);
    assert_eq!(fixture.revision, REVISION);
    assert_eq!(fixture.tokenizer_path, TOKENIZER_PATH);

    let tokenizer = load_tokenizer();
    let cases = prompt_cases();
    assert_eq!(fixture.cases.len(), cases.len());

    for (case, expected) in cases.iter().zip(fixture.cases.iter()) {
        assert_eq!(expected.id, case.id);
        assert_eq!(expected.user_prompt, case.user_prompt);
        assert_eq!(expected.image_count, case.image_count);
        assert_eq!(expected.enable_thinking, case.enable_thinking);

        let rendered_prompt = chat_prompt(case.user_prompt, case.image_count, case.enable_thinking);
        assert_eq!(expected.rendered_prompt, rendered_prompt, "{}", case.id);

        let token_ids = tokenizer
            .encode(rendered_prompt, false)
            .expect("tokenize rendered prompt")
            .get_ids()
            .to_vec();
        assert_eq!(expected.prompt_token_count, token_ids.len(), "{}", case.id);
        assert_eq!(expected.token_ids, token_ids, "{}", case.id);
    }
}

#[test]
fn image_placeholder_expansion_matches_minicpm_shape() {
    let prompt = chat_prompt("What is shown?", 1, false);
    let images = ProcessedImages {
        pixel_values: candle::Tensor::zeros(
            (1, 3, 14, 14),
            candle::DType::F32,
            &candle::Device::Cpu,
        )
        .expect("dummy pixel tensor"),
        target_sizes: vec![(4, 4), (4, 4), (4, 4)],
        images: vec![ImageMeta {
            path: "image.png".into(),
            grid: (1, 2),
            patch_count: 3,
        }],
    };

    let expanded =
        expand_image_placeholders(prompt, &images, true, "16x").expect("expand placeholder");

    assert!(expanded.contains(
        "<image_id>0</image_id><image><|image_pad|></image>\n<slice><|image_pad|></slice>"
    ));
    assert!(expanded
        .contains("<slice><|image_pad|></slice><slice><|image_pad|></slice>\nWhat is shown?"));
}

#[test]
fn transformers_oracle_fixtures_are_well_formed() {
    let text_fixture: Value =
        serde_json::from_str(TRANSFORMERS_TEXT_LOGITS_FIXTURE).expect("parse text logits fixture");
    assert_eq!(text_fixture["model_id"], MODEL_ID);
    assert_eq!(text_fixture["revision"], REVISION);
    assert!(text_fixture["weights_dir"].as_str().is_some());

    let text_cases = text_fixture["cases"].as_array().expect("text cases");
    let logits_cases = text_cases
        .iter()
        .filter(|case| case["image_count"].as_u64() == Some(0))
        .collect::<Vec<_>>();
    assert_eq!(logits_cases.len(), 3);
    for case in logits_cases {
        let top_token_ids = case["next_token_logits"]["top_token_ids"]
            .as_array()
            .expect("top token ids");
        let top_logits = case["next_token_logits"]["top_logits"]
            .as_array()
            .expect("top logits");
        assert_eq!(top_token_ids.len(), 10, "{}", case["id"]);
        assert_eq!(top_logits.len(), 10, "{}", case["id"]);
    }

    let image_fixture: Value = serde_json::from_str(TRANSFORMERS_IMAGE_EXPANSION_FIXTURE)
        .expect("parse image expansion fixture");
    assert_eq!(image_fixture["model_id"], MODEL_ID);
    let image_case = image_fixture["cases"]
        .as_array()
        .expect("image cases")
        .iter()
        .find(|case| case["id"] == "single_image_closed_thinking")
        .expect("single-image case");
    let expanded_token_ids = image_case["expanded_token_ids"]
        .as_array()
        .expect("expanded token ids");
    assert_eq!(
        image_case["expanded_prompt_token_count"].as_u64(),
        Some(expanded_token_ids.len() as u64)
    );
    assert_eq!(expanded_token_ids.len(), 211);
    assert!(expanded_token_ids
        .iter()
        .any(|token| token.as_u64() == Some(248078)));
    assert!(expanded_token_ids
        .iter()
        .any(|token| token.as_u64() == Some(248088)));
}

#[test]
#[ignore = "prints the JSON fixture used by minicpm_v46_text_fixture_matches_runner"]
fn regenerate_minicpm_v46_text_fixture() {
    let tokenizer = load_tokenizer();
    let cases = prompt_cases()
        .into_iter()
        .map(|case| {
            let rendered_prompt =
                chat_prompt(case.user_prompt, case.image_count, case.enable_thinking);
            let token_ids = tokenizer
                .encode(rendered_prompt.clone(), false)
                .expect("tokenize rendered prompt")
                .get_ids()
                .to_vec();
            json!({
                "id": case.id,
                "user_prompt": case.user_prompt,
                "image_count": case.image_count,
                "enable_thinking": case.enable_thinking,
                "rendered_prompt": rendered_prompt,
                "prompt_token_count": token_ids.len(),
                "token_ids": token_ids,
            })
        })
        .collect::<Vec<_>>();

    let fixture = json!({
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "tokenizer_path": TOKENIZER_PATH,
        "source": "MiniCPM-V-4.6 chat_template.jinja, mirrored by lmbrrr::prompt::chat_prompt",
        "cases": cases,
    });
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
