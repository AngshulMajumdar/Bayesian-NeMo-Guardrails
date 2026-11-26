# Publishing

1. Create a GitHub repository named `Bayesian-NeMo-Guardrails` (recommended path `https://github.com/AngshulMajumdar/Bayesian-NeMo-Guardrails`).
2. Unzip the GitHub archive and upload its **contents**, not an enclosing directory, to the new repository's `main` branch.
3. GitHub Actions will run unit tests and actual NVIDIA NeMo 0.25.0 integration tests automatically; expect `.github/workflows/tests.yml` (include the extension!).
4. Create a Hugging Face Docker Space named `Bayesian-NeMo-Guardrails` under `Angshul`, upload the **separate Hugging Face Space ZIP contents** to its root.
5. Choose CPU basic; the Dockerfile launches Gradio on port 7860. No model API keys are needed for this illustrative detector-score demo.
6. Before describing this as operational safety enforcement, calibrate with labeled real detector data, set audited policy costs, and validate with real NeMo/RAG traffic. Don't present synthetic risk scores as calibrated real-world probabilities.

The GitHub repository and HF Space are distinct packages. Neither modifies the existing Fuzzy NeMo projects.
