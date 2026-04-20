# snowball-py

**snowball-py** is a small, pure-Python toolkit for **probabilistic question answering** in product and service discovery surveys. It maintains Bayesian beliefs over a set of targets, asks informative questions, and can simulate respondents, run benchmarks, optimise hyper-parameters, and plot learning and accuracy curves.

## What it does

- **Inference core** — Maintains posterior weights over targets from survey answers; selects the next question using entropy, how fast beliefs are moving (“velocity”), and how uncertain the top choice still is (“lack”).
- **Interactive & simulated sessions** — Run a live CLI survey or a scripted respondent for repeatable experiments.
- **Benchmarks & graphs** — Learning curves, accuracy versus number of questions, entropy over time, hyper-parameter comparisons, and a compact dashboard-style summary.
- **Hyper-parameter search** — Grid search with local refinement to tune engine settings.

Python **3.11+**, **NumPy**, and **Matplotlib** (see `pyproject.toml`).

## Install

```bash
cd snowball-py
uv sync
```

Or with pip, install from the project directory in the usual way.

## Quick start

```bash
uv run snowball              # interactive survey
uv run snowball --sim        # simulated answers
uv run snowball-bench        # benchmark suite
uv run snowball-graphs       # regenerate Matplotlib figures under plots/
```

Tests:

```bash
uv run pytest
```

## Roadmap

snowball-py is intentionally small and dependency-light today. The direction is to grow toward the same class of capabilities as a full **CPU/GPU probabilistic-QA engine** (training, persistence, many concurrent quizzes, production-oriented ops)—without sacrificing a clear Python surface.

### Acceleration (CUDA and hardware)

- **GPU-backed math** — Move the heaviest paths (posterior updates, training accumulation, candidate-question scoring at scale) behind a backend that can use **CUDA** (e.g. CuPy, or JAX on GPU), with a **NumPy CPU fallback** so tests and laptops stay simple.
- **Streams and memory** — Mirror a serious GPU stack: **non-blocking streams**, bounded **device memory pools**, and batched work so large \(Q \times A \times T\) layouts do not spend all their time allocating.
- **Precision** — Standardise on **float32** on GPU for throughput; offer **float64** where numerical stability matters (mirroring engines that only exposed float on CUDA first, then broadened types).

### Engine depth (parity with a “full” native core)

- **Offline training** — First-class **`Train`-style** APIs: ingest batches of answered questions and target labels without running an interactive quiz, for replay and dataset-driven fitting.
- **Binary KB lifecycle** — **Save/load** of the full knowledge base to disk in a **versioned** container (double-buffered writes optional), not only ad-hoc pickles.
- **Identity and sessions** — **Stable vs compact IDs** for questions, targets, and quizzes; **resume** quizzes from stored answers; **retention policies** (max quizzes, max age) for long-running services.
- **Diagnostics** — Richer **statistics** (e.g. total questions asked, per-target frequency exports) for analytics and metering.

### Operations and integration

- **Parallelism** — **Worker pools** for offline training jobs and batch benchmarks so multicore hosts are used without nested thread explosions.
- **Observability** — Structured **logging** and optional **metrics hooks** (latency per phase, GPU memory high-water marks).
- **Embeddable service** — Optional thin **HTTP or gRPC** layer around the engine for integration tests and non-Python clients, keeping the core importable as a library.

### CPU path

- **Vectorised CPU** — Where GPU is unavailable, push more work through **NumPy layout discipline**, **Numba**, or small **SIMD-friendly** kernels so the CPU story stays competitive for modest dimensions.

Order of attack is not fixed; feedback and benchmarks will drive whether CUDA landings or persistence/session APIs come first.

## Figures

Representative outputs from the bundled benchmark and graph utilities:

### Learning curve

![Learning curve](plots/learning_curve.png)

### Best configuration (learning trajectory)

![Best configuration curve](plots/best_config_curve.png)

### Accuracy vs. questions asked

![Accuracy vs questions](plots/accuracy_vs_questions.png)

### Entropy convergence

![Entropy convergence](plots/entropy_convergence.png)

### Hyper-parameter comparison

![Hyper-parameter comparison](plots/hyperparam_comparison.png)

### Optimisation landscape

![Optimisation](plots/optimisation.png)

### Dashboard summary

![Dashboard](plots/dashboard.png)
