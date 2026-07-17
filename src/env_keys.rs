//! Single source of truth for every environment-variable name the runtime
//! reads. The `from_env` config resolvers and command entrypoints reference
//! these constants instead of string literals, so the whole tunable surface
//! is greppable in one place and [`KNOWN_LMBRRR_KEYS`] (the typo-guard
//! registry) cannot drift from the keys that are actually consumed.
//!
//! Compile-time keys read via `env!` (`LMBRRR_GIT_REV`, `LMBRRR_CANDLE_PIN`)
//! are intentionally absent: `env!` requires a string literal and is resolved
//! by the build script, not at runtime.

// --- Tensor-op (matmul2d) route — mm2d.rs / ModelCtx ---
pub const MM2D: &str = "LMBRRR_MM2D";
pub const MM2D_MIN_N: &str = "LMBRRR_MM2D_MIN_N";
pub const MM2D_BODY_MIN_M: &str = "LMBRRR_MM2D_BODY_MIN_M";
pub const MM2D_HEAD_MIN_N: &str = "LMBRRR_MM2D_HEAD_MIN_N";
pub const MM2D_SPLITK: &str = "LMBRRR_MM2D_SPLITK";
pub const MM2D_SPLIT_TGS: &str = "LMBRRR_MM2D_SPLIT_TGS";
pub const MM2D_PLANE_CACHE: &str = "LMBRRR_MM2D_PLANE_CACHE";
pub const MM2D_CACHE_DIR: &str = "LMBRRR_MM2D_CACHE_DIR";
pub const MM2D_PLANAR: &str = "LMBRRR_MM2D_PLANAR";
pub const FUSED_VERIFY_ARGMAX: &str = "LMBRRR_FUSED_VERIFY_ARGMAX";

// --- Kernel-fusion route gates — KernelRouteConfig (qwen35 tree) ---
pub const UNFUSED_RMSNORM: &str = "LMBRRR_UNFUSED_RMSNORM";
pub const UNFUSED_SDPA: &str = "LMBRRR_UNFUSED_SDPA";
pub const UNFUSED_RECONSTRUCT: &str = "LMBRRR_UNFUSED_RECONSTRUCT";
pub const UNFUSED_DELTANET: &str = "LMBRRR_UNFUSED_DELTANET";
pub const FUSED_ATTN_PREP: &str = "LMBRRR_FUSED_ATTN_PREP";
pub const FUSED_MTP_FC: &str = "LMBRRR_FUSED_MTP_FC";
pub const DELTANET_V2: &str = "LMBRRR_DELTANET_V2";
pub const DELTANET_SEQUENTIAL: &str = "LMBRRR_DELTANET_SEQUENTIAL";
pub const DELTANET_PREFILL_FUSED: &str = "LMBRRR_DELTANET_PREFILL_FUSED";
pub const DELTANET_PREFILL_CAP: &str = "LMBRRR_DELTANET_PREFILL_CAP";

// --- Decode-loop path selection — DecodeConfig (generate.rs) ---
pub const ASYNC_READBACK: &str = "LMBRRR_ASYNC_READBACK";
pub const FUSED_ARGMAX: &str = "LMBRRR_FUSED_ARGMAX";

// --- Weight-pack sidecar — PackConfig (pack.rs) ---
pub const PACK: &str = "LMBRRR_PACK";

// --- MTP head-quantization bisection hook — RuntimeConfig ---
pub const MTP_Q_ONLY: &str = "LMBRRR_MTP_Q_ONLY";

// --- Spec-run command knobs — SpecRunConfig (commands/dspark.rs) ---
pub const LOOP_TIMING: &str = "LMBRRR_LOOP_TIMING";
pub const READVANCE_ROLLBACK: &str = "LMBRRR_READVANCE_ROLLBACK";
pub const SPEC_FENCED_TIMING: &str = "LMBRRR_SPEC_FENCED_TIMING";
pub const MTP_ADAPTIVE_DEPTH: &str = "LMBRRR_MTP_ADAPTIVE_DEPTH";
pub const PROPOSE_TIMING: &str = "LMBRRR_PROPOSE_TIMING";

// --- Diagnostics — command scope (commands/diag.rs) ---
pub const VT_PROFILE: &str = "LMBRRR_VT_PROFILE";

/// Every `LMBRRR_*` key the runtime resolves via `std::env::var`.
/// `RuntimeConfig::from_env` warns on any `LMBRRR_*` variable present in the
/// environment but absent from this list — the guard against silently-typoed
/// flags in the bench/suite scripts (a real past failure class).
pub const KNOWN_LMBRRR_KEYS: &[&str] = &[
    MM2D,
    MM2D_MIN_N,
    MM2D_BODY_MIN_M,
    MM2D_HEAD_MIN_N,
    MM2D_SPLITK,
    MM2D_SPLIT_TGS,
    MM2D_PLANE_CACHE,
    MM2D_CACHE_DIR,
    MM2D_PLANAR,
    FUSED_VERIFY_ARGMAX,
    UNFUSED_RMSNORM,
    UNFUSED_SDPA,
    UNFUSED_RECONSTRUCT,
    UNFUSED_DELTANET,
    FUSED_ATTN_PREP,
    FUSED_MTP_FC,
    DELTANET_V2,
    DELTANET_SEQUENTIAL,
    DELTANET_PREFILL_FUSED,
    DELTANET_PREFILL_CAP,
    ASYNC_READBACK,
    FUSED_ARGMAX,
    PACK,
    MTP_Q_ONLY,
    LOOP_TIMING,
    READVANCE_ROLLBACK,
    SPEC_FENCED_TIMING,
    MTP_ADAPTIVE_DEPTH,
    PROPOSE_TIMING,
    VT_PROFILE,
];

// --- External (OS / framework) vars, not part of the LMBRRR tunable surface ---
/// Metal's undocumented gate for GPU capture (`lmbrrr trace`/`--gpu-capture-*`).
pub const METAL_CAPTURE_ENABLED: &str = "METAL_CAPTURE_ENABLED";
/// Home directory, for the default plane-cache location.
pub const HOME: &str = "HOME";
