//! The CUDA execution provider's contract at the one point where it is
//! observable without a GPU, a model file or the CUDA build: asking for
//! CUDA must never quietly produce a CPU session.
//!
//! Both tests use a path that does not exist. `Model::from_config` picks
//! the ONNX backend from the `.onnx` extension before it touches the file,
//! so the session builder runs and its error is what comes back — which is
//! the thing under test.
#![cfg(all(feature = "ort-backend", feature = "decode"))]

use charon_audio::{ExecutionProvider, Separator, SeparatorConfig};

fn cuda_config() -> SeparatorConfig {
    let mut config = SeparatorConfig::htdemucs("/nonexistent/htdemucs.onnx");
    config.model.onnx.execution_provider = ExecutionProvider::Cuda;
    config
}

fn cuda_error() -> String {
    match Separator::new(cuda_config()) {
        Ok(_) => panic!("asking for CUDA produced a separator"),
        Err(e) => e.to_string(),
    }
}

/// Without the feature, `with_cuda` is the `Err` arm: the caller is told
/// the build has no CUDA rather than handed a CPU session that looks fine
/// and runs at a ninth of the speed.
#[test]
#[cfg(not(feature = "cuda"))]
fn cuda_without_the_feature_is_a_clear_error() {
    let error = cuda_error();
    assert!(
        error.contains("`cuda` feature"),
        "expected the missing-feature error, got: {error}"
    );
}

/// With the feature, the same request gets as far as the model file, so the
/// error is about the file. Anything else means the CUDA path was skipped.
#[test]
#[cfg(feature = "cuda")]
fn cuda_with_the_feature_reaches_the_session_builder() {
    let error = cuda_error();
    assert!(
        !error.contains("`cuda` feature"),
        "the `cuda` feature is on, so the missing-feature error is wrong: {error}"
    );
    assert!(
        !error.to_lowercase().contains("fallback"),
        "the CPU fallback was used instead of failing: {error}"
    );
}
