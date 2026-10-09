# This fork: one patch, the CUDA execution provider

`windup-bird/charon-audio-cuda` is `Valkyra-Labs/charon-audio` at `v0.1.2` plus
a single change: **`ExecutionProvider::Cuda`**, so the HTDemucs graph can run on
an NVIDIA GPU. Upstream 0.1.2 has `Cpu`, `CoreMl` and `Auto` only.

The patch is written the way upstream would take it — the CPU path is untouched
and the feature mirrors their `coreml` one — so the intent is to send it
upstream as a PR rather than carry a fork forever.

## What it changes

| Change | Why |
|---|---|
| `cuda = ["ort-backend", "ort/cuda"]` feature, beside `coreml` | The same shape as the CoreML feature, one provider down. |
| `ExecutionProvider::Cuda`, and `build_session` branching on the EP instead of a `coreml: bool` | A third provider no longer needs a second boolean. `Cpu` and `Auto` reach the same arm as before. |
| **`ConstantFolding` stays enabled on the CUDA path.** | The non-obvious one. `OnnxOptions::low_memory()` — which `ModelConfig::htdemucs` inherits — disables that optimizer to cap RAM, and with it disabled **ONNX Runtime's CUDA EP claims no nodes at all**: the whole 1201-node graph lands back on the CPU. The filter is applied only when the provider is CUDA, so the CPU path keeps upstream's memory behaviour exactly. |
| `session.disable_cpu_ep_fallback = 1` | Without it, a CUDA session that cannot take the graph silently runs on the CPU at a ninth of the speed. A front-end has to be able to tell "the GPU did the work" from "the GPU was asked for and ignored". |
| `charon ... --ep cuda` | The CLI's `parse_ep` matched exhaustively; without this the binary does not compile with the new variant. |

The same measurement also settles a comment upstream still carries: the note on
`ModelConfig::htdemucs` that "the in-graph STFT export only runs on the CPU
provider" is not accurate about the export — the export runs entirely on CUDA,
and the `low_memory()` preset is what stops it. This branch leaves that comment
alone on purpose; correcting it is upstream's call, and it is called out in the
PR description instead.

## Numbers

Measured on an RTX 4050 Laptop 6 GB, CUDA 13.4, cuDNN 9.26, ONNX Runtime 1.28
(cuda13), in-graph `htdemucs.onnx`:

| session | per 7.8 s window | full 3:39 track | VRAM |
|---|---|---|---|
| CUDA, `disabled_optimizers = []` | 214 ms | 13 s | 3.2 GB |
| CUDA, `memory_pattern = false` | 214 ms | | |
| CUDA, `disabled_optimizers = ["ConstantFolding"]` | **no node claimed** — a hard failure under `disable_cpu_ep_fallback`, a silent CPU run without it | | |
| CPU | 2055 ms | 87 s | 1.98 GB |

9.6x per window, 6.7x end to end. Because the session both builds and runs with
`disable_cpu_ep_fallback` set, all 1201 nodes are on the CUDA EP.

## Checking it

```bash
cargo test --features cuda
CHARON_HTDEMUCS_MODEL=/path/to/htdemucs.onnx \
  cargo test --release --features cuda -- --ignored htdemucs
```

`tests/cuda_ep.rs` covers the one rule that is observable without a GPU: asking
for CUDA never yields a CPU session. Whether the `cuda` feature is compiled in
or not, the request becomes an error the caller can act on.

CPU and CUDA outputs are not bit-identical — different reduction orders in fp32.
On a 9 s synthetic clip the largest sample difference is 6.2e-4 against a stem
RMS of 0.049 (-38 dB); the parity criterion upstream uses is frame RMS against
PyTorch, which both paths pass.

## Keeping up with upstream

```bash
git fetch upstream
git checkout main && git merge --ff-only upstream/main && git push
git checkout feat/cuda-ep && git rebase main
```

If upstream ever grows a CUDA provider of its own, this branch is deleted and
nothing else changes.
