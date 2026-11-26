# Bayesian NeMo Guardrails — test report

**Date:** November 26, 2026  
**Upstream source:** user-supplied NVIDIA `Guardrails-0.25.0.zip`  
**Runtime:** CPU only  
**Package version:** 0.1.0

## Local package tests

```text
python -m pytest -q
40 passed, 1 skipped
```

The skipped module is `tests/test_live_nemo_optional.py`: NVIDIA NeMo itself is not installed in the local Python 3.13 working environment. The NeMo package **was** separately installed and executed in Hugging Face CPU Jobs on Python 3.12.

Local tests cover:

- Nonconjugate logistic model calibration via damped Newton posterior mode search; precision and Hessian positive definiteness.
- Gaussian Laplace posterior-predictive computation with deterministic Gauss–Hermite integration.
- Deterministic corrected Gauss–Hermite reweighting against the **true unnormalized posterior**, including grid-refinement consistency tests.
- Credible-interval computation and widening under reduced calibration data.
- Snapshot round-trip, calibration failure for improper precision, invalid/missing labels, out-of-range scores, unsupported inference modes, and excessive tensor quadrature grids.
- Posterior expected-loss decisions, credible-bound gating, mandatory hard vetoes, and fail-closed missing score handling.
- Both adapter contracts with doubles: native NeMo blocks take precedence, modified content is not bypassed, unknown statuses fail closed, generation is withheld without an output scorer, and unsafe streaming is disallowed.
- FastAPI health, config and evaluation endpoints.

## Actual NVIDIA NeMo 0.25.0 engine tests

The final `src/bayesian_nemo/adapters.py` module was **embedded byte for byte** into a Python 3.12 CPU job with official `nemoguardrails==0.25.0`, and checked against actual instances of `Guardrails(..., use_iorails=False)` and `Guardrails(..., require_iorails=True)`.

Final adapter SHA-256:

```text
544d3d8601673ac16d590aedc68dfd00e5d3b1282c1cf826325810fca5352eee
```

Checks that passed on each actual engine:

| Real-engine test | LLMRails | IORails |
|---|---|---|
| Native input rail benign text passed | Pass | Pass |
| Native input rail regex block preserved | Pass | Pass |
| Bayesian-layer block withheld | Pass | Pass |
| Native output rail benign text passed | Pass | Pass |
| Native output rail regex block preserved | Pass | Pass |
| Bayesian-layer output block withheld | Pass | Pass |
| Native custom Python action registration | Pass | Not supported by design |
| Colang 1.0 action, trusted context scores, passed/blocked | Pass | Not supported by design |

The remote adapter integration harness used a minimal deterministic test `BayesianPolicy` stub to isolate NeMo compatibility. **The actual Bayesian model and policy** are exercised by the local unit and adapter tests, not by an end-to-end NeMo+calibration run. The repository also contains `tests/test_live_nemo_optional.py` using the **actual BayesianPolicy**, and GitHub Actions installs NeMo 0.25.0 and runs that module; it will provide combined verification on publication.

Remote test jobs (readable logs):

- [Actual NeMo 0.25.0 adapters, Colang integration](https://huggingface.co/jobs/Angshul/6b07e4a0d91f621b930a2835)
- [Final source adapter plus wrapped input/output](https://huggingface.co/jobs/Angshul/6b07e4ddd91f621b930a28c6)
- [Native NeMo output rail control test](https://huggingface.co/jobs/Angshul/6b07e4b9b6a90bbe997f8e03)

## Gradio / Hugging Face Space simulation test

The **separate Space package** was launched locally on port 7860 and tested with HTTP `GET /`, `GET /config`, `GET /gradio_api/info` and Gradio `/evaluate` API calls.

All eight test calls produced expected results across both `laplace` and `corrected_gh`:

| Input / output simulated scores | Expected | Actual |
|---|---|---|
| Input 0.02/0.02, output 0.02 | ALLOW | ALLOW |
| Input 0.99/0.02, output 0.02 | BLOCK | BLOCK |
| Input 0.02/0.02, output 0.99 | BLOCK | BLOCK |
| Input 0.50/0.02, output 0.02 | REVIEW | REVIEW |

## Local CPU inference timing (not a standardized benchmark)

With the provided **synthetic input model** and 2,000 repeated in-process evaluations:

| Method | Mean Python call duration |
|---|---:|
| Laplace + scalar Gauss–Hermite | ~0.08 ms |
| Posterior-corrected tensor Gauss–Hermite (1,331 precomputed nodes) | ~0.13 ms |

No remote-model inference, RAG latency, detector latency, Docker cold starts, or LLM costs are included.

## Explicitly not tested

- Real prompt-injection / harmful-content / hallucination detectors, real policy labels or production calibration.
- End-to-end LLM generation with NeMo provider calls or a real RAG pipeline.
- Robustness, out-of-distribution reliability, or any calibrated real-world safety guarantee.
- Approximation accuracy for arbitrary high-dimensional, strongly non-Gaussian or multimodal posteriors.

This is **implementation software**, not a research paper or validation study. Do not present synthetic example posteriors as real-world violation probabilities.
