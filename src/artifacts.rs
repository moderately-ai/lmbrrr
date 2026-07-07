use std::{
    collections::HashSet,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};

#[derive(Clone, Debug)]
pub struct ArtifactOverrides {
    pub config: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub generation_config: Option<PathBuf>,
    pub preprocessor: Option<PathBuf>,
    pub weights: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct Artifacts {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub generation_config: Option<PathBuf>,
    pub preprocessor: Option<PathBuf>,
    pub weights: Vec<PathBuf>,
}

pub fn resolve_model_artifacts(
    model_id: &str,
    revision: &str,
    overrides: ArtifactOverrides,
) -> Result<Artifacts> {
    let api = Api::new().context("create Hugging Face cache API")?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let config = overrides
        .config
        .map(Ok)
        .unwrap_or_else(|| repo.get("config.json").context("download config.json"))?;
    let tokenizer = overrides.tokenizer.map(Ok).unwrap_or_else(|| {
        repo.get("tokenizer.json")
            .context("download tokenizer.json")
    })?;
    let generation_config = match overrides.generation_config {
        Some(path) => Some(path),
        None => repo.get("generation_config.json").ok(),
    };
    let preprocessor = match overrides.preprocessor {
        Some(path) => Some(path),
        None => repo.get("preprocessor_config.json").ok(),
    };

    let weights = if overrides.weights.is_empty() {
        hub_load_safetensors(&repo, "model.safetensors.index.json").or_else(|_| {
            repo.get("model.safetensors")
                .context("download model.safetensors")
                .map(|p| vec![p])
        })?
    } else if overrides.weights.len() == 1 && overrides.weights[0].is_dir() {
        local_load_safetensors(&overrides.weights[0], "model.safetensors.index.json")?
    } else {
        overrides.weights
    };

    Ok(Artifacts {
        config,
        tokenizer,
        generation_config,
        preprocessor,
        weights,
    })
}

fn hub_load_safetensors(repo: &ApiRepo, json_file: &str) -> Result<Vec<PathBuf>> {
    let json_path = repo
        .get(json_file)
        .with_context(|| format!("download {json_file}"))?;
    let json_file =
        File::open(&json_path).with_context(|| format!("open {}", json_path.display()))?;
    let json: serde_json::Value = serde_json::from_reader(json_file)
        .with_context(|| format!("parse {}", json_path.display()))?;
    let weight_map = json
        .get("weight_map")
        .and_then(|value| value.as_object())
        .with_context(|| format!("missing weight_map in {}", json_path.display()))?;
    let mut filenames = HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            filenames.insert(file.to_string());
        }
    }
    filenames
        .into_iter()
        .map(|file| repo.get(&file).with_context(|| format!("download {file}")))
        .collect()
}

fn local_load_safetensors(dir: &Path, json_file: &str) -> Result<Vec<PathBuf>> {
    let json_path = dir.join(json_file);
    let json_file =
        File::open(&json_path).with_context(|| format!("open {}", json_path.display()))?;
    let json: serde_json::Value = serde_json::from_reader(json_file)
        .with_context(|| format!("parse {}", json_path.display()))?;
    let weight_map = json
        .get("weight_map")
        .and_then(|value| value.as_object())
        .with_context(|| format!("missing weight_map in {}", json_path.display()))?;
    let mut filenames = HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            filenames.insert(dir.join(file));
        }
    }
    Ok(filenames.into_iter().collect())
}
