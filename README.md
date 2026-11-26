# Bayesian NeMo Guardrails

**A deterministic approximate Bayesian decision layer for NVIDIA NeMo Guardrails — compatible with both `LLMRails` and `IORails`.**

An independent, CPU-friendly Python package. It **does not fork NVIDIA's repository**, train an LLM, download model weights, make extra LLM calls for inference, or independently detect unsafe content. It receives **trusted detector scores** and returns auditable **ALLOW / REVIEW / BLOCK** decisions. Native NeMo blocks always take precedence.

> **Correct mathematical description:** The statistical objective integrates over the full nonconjugate Bayesian parameter posterior. The two implemented methods are **deterministic approximations** to that posterior or its integrals—not exact full-posterior inference. There is **no MAP plug-in decision**, no Variational Bayes, and no MCMC. The posterior mode is used only to construct the Laplace proposal.
>
> **Safety caveat:** Shipped calibration data are **synthetic** for demonstration. The demo risk probabilities and hard-veto thresholds are not calibrated production safety claims. Replace models with labeled data from your real detectors before deployment.

## What it computes

For binary policy violation `H` and detector feature vector `x`, the model uses a logistic likelihood with a proper nonconjugate Gaussian prior on `θ`:

\[p(H=1\mid x,\theta)=\sigma([1,x]^\top\theta),\qquad p(\theta\mid D)\propto p(D\mid\theta)p(\theta).\]

The quantity used for the decision is a **posterior predictive risk**:

\[r(x)=\int \sigma([1,x]^\top\theta)\,p(\theta\mid D)\,d\theta.\]

Two deterministic choices are available:

1. **`laplace`**: damped Newton optimization finds the posterior mode and local Hessian; a Gaussian posterior approximation is built from its inverse. The predictive sigmoid-normal integral becomes **one-dimensional** and is evaluated with Gauss–Hermite quadrature. The resulting interval concerns uncertainty in the conditional risk attributable to model parameters, *not* a safety guarantee or frequentist confidence interval.
2. **`corrected_gh`**: for low-dimensional models, construct a tensor Gauss–Hermite grid about the Laplace approximation, then reweight every grid node by the ratio of the **true unnormalized non-Gaussian posterior** to the Gaussian proposal. This performs deterministic, nonlinear posterior integration beyond the local Gaussian approximation. A strict configurable node cap prevents dimension explosion; **no silent fallback to approximate modes**.

The corrected backend is useful as a *numerical accuracy check* and for 1–2 detector features (plus intercept), rather than a high-dimensional general-purpose solver. Both methods remain approximations; quadrature refinement and independent validation against labeled held-out examples are required before any production claim.

## Safety decisions

The model computes expected loss for every action `a`:

\[\mathbb E[L(a,H)\mid x,D]=(1-r(x))L(a,0)+r(x)L(a,1).\]

The decision is the minimum expected loss among `allow`, `review`, and `block`. Policy configuration optionally restricts release using an **upper parameter-risk credible bound**, restricts automatic blocking using a lower bound, and specifies **hard veto thresholds** that cannot be overridden by the Bayesian score. Missing/invalid features and unknown NeMo statuses **fail closed** as REVIEW; REVIEW is *not* permission to release.

## Architecture

```text
        Trusted detector scores + labeled calibration CSV
                        │ offline calibrate
                        ▼
                 BayesianLogisticModel
                   ├─ laplace (Gaussian) + 1-D Gauss–Hermite
                   └─ corrected_gh (Laplace-centered posterior correction)
                        │
              BayesianPolicy (posterior risk + cost matrix + credible gates + hard vetoes)
                        │
                ┌───────┴─────────┐
                ▼                 ▼
          LLMRailsAdapter    IORailsAdapter
          Colang action      external wrapper
                │                 │
                └───────┬─────────┘
                        ▼
          NeMo-native checks (blocks never overturned)
                        ▼
              ALLOW / REVIEW / BLOCK
```

## Install and run

```bash
python -m pip install -e '.[api,test]'
python -m pytest -q
bayesian-nemo evaluate --config examples/policy.yaml --phase input --scores '{"injection_risk":0.02,"toxicity_risk":0.02}'
bayesian-nemo evaluate --config examples/corrected_policy.yaml --phase output --scores '{"unsupported_claim_risk":0.99}'
uvicorn bayesian_nemo.api:app --host 0.0.0.0 --port 8000
```

API endpoints are `GET /health`, `GET /config`, and `POST /evaluate`. Example:

```bash
curl http://localhost:8000/evaluate -H 'Content-Type: application/json' \
  -d '{"phase":"input","scores":{"injection_risk":0.02,"toxicity_risk":0.02}}'
```

Optional local UI:

```bash
python -m pip install -e '.[demo]'
python -m bayesian_nemo.demo
```

### Calibrate with your own labeled data

Use the repository's **synthetic demonstration CSV** only to understand the format:

```bash
bayesian-nemo calibrate \
  --csv examples/data/synthetic_input.csv \
  --output my_input_model.json \
  --features injection_risk toxicity_risk \
  --label unsafe \
  --backend corrected_gh \
  --quadrature-order 11 \
  --prior-diagonal 0.3 0.15 0.15
```

The label `unsafe=1` means an independently validated policy violation. Detector scores are bounded to `[0,1]` and must be supplied by a **trusted application**, not by a user message or retrieved document. **Do not update the calibration posterior using unlabeled live conversation events.** Store model snapshots separately from untrusted input. The current calibration CLI uses Gaussian priors but a *nonconjugate logistic likelihood*; the prior precision is configurable.

To switch your runtime from example models, update the YAML `phases.*.model` references to your own calibration JSON snapshots. The YAML also defines the loss matrix, interval gate, and mandatory hard vetoes; no retraining is needed to change decision costs.

### Two real NeMo 0.25.0 adapters

After installing `python -m pip install -e '.[test,nemo]'`, run:

```bash
python examples/legacy_colang/run_local.py
python examples/iorails/run_local.py
python -m pytest -q tests/test_live_nemo_optional.py
```

`LLMRails` supports an optional native Colang action, fed **trusted detector scores** through `ActionResult.context_updates`. `IORails` does **not** support custom Python actions, so the integration wraps its `Guardrails.check_async`/`generate_async` interfaces and explicitly requests `require_iorails=True`. This is **not a native IORails plug-in**.

```python
from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.rails.llm.options import RailType
from bayesian_nemo import BayesianPolicy, IORailsAdapter

rails = Guardrails(RailsConfig.from_path('my_nemo_config'), require_iorails=True)
policy = BayesianPolicy.from_yaml('examples/policy.yaml')
adapter = IORailsAdapter(rails, policy)

# Within an async handler; scores must be produced by trustworthy detectors.
result = await adapter.check_async(
    messages=[{'role':'user','content':'Summarize this document'}],
    scores={'injection_risk': 0.02, 'toxicity_risk': 0.02},
    rail_types=[RailType.INPUT],
)
if result.released:
    safe_checked_text = result.checked_content
else:
    print(result.decision, result.reason)
```

For generation with output protection, supply an `output_scorer(native_response)` that computes detector scores from the **actual generated text and evidence**. If output scoring is configured but the scorer is absent or fails, content is withheld. The adapter does not allow streaming of unchecked tokens. NeMo's configured native input/output rails are still in effect.

## Repository contents

- `src/bayesian_nemo/model.py`: Newton/Laplace, posterior-corrected Gauss–Hermite, predictive uncertainty, snapshot serialization.
- `src/bayesian_nemo/policy.py`: Bayesian decision costs, credible interval gates, hard vetoes, fail-closed evaluation.
- `src/bayesian_nemo/adapters.py`: NeMo 0.25.0 dual-engine boundary.
- `src/bayesian_nemo/calibration.py`: offline labeled CSV calibration.
- `examples/data/`: reproducible **synthetic** calibration datasets.
- `src/bayesian_nemo/models/`: demo calibration snapshots for both deterministic backends.
- `tests/`: numerical, API, adapter, fail-closed, and optional real NeMo tests.
- `.github/workflows/tests.yml`: CI matrix for Python 3.11/3.12 plus an actual NeMo 0.25.0 job.

## Limitations

- Neither deterministic backend is exact under an arbitrary nonconjugate posterior. The Laplace approximation can be unreliable for multimodal, highly skewed, or weakly identified parameter posteriors; corrected GH also has finite-grid error and scales exponentially with coefficient dimension.
- Credible intervals for conditional risk do **not** quantify all possible detector misspecification, distribution shift, data labeling errors, or adversarial robustness.
- The implementation models each phase independently; it does not learn a temporal Bayesian state model, causal dependence, or correlation among repeated query events. Avoid double-counting correlated evidence.
- No production detectors, LLM generation benchmark, or large RAG pipeline are bundled. The Hugging Face Space is a **manual-score inference demo**, not a full AI security product.

**Upstream:** [NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails). Independent open-source software; not affiliated with or endorsed by NVIDIA.
