"""Gradio visualization of deterministic approximate Bayesian inference.

All detector values are manually supplied; no guardrail detection occurs here.
"""
from __future__ import annotations
from pathlib import Path
import gradio as gr
from .policy import BayesianPolicy

HERE=Path(__file__).resolve().parent
LAPLACE=BayesianPolicy.from_yaml(HERE/'default_policy.yaml')
CORRECTED=BayesianPolicy.from_yaml(HERE/'corrected_policy.yaml')

CSS="""
.gradio-container {max-width: 1050px !important; margin:auto;}
h1 {font-weight: 850 !important; letter-spacing: -.035em;}
"""

def evaluate(backend:str,injection:float,toxicity:float,unsupported:float):
    p=CORRECTED if backend.startswith('Corrected') else LAPLACE
    inp=p.evaluate('input',{'injection_risk':injection,'toxicity_risk':toxicity})
    out=p.evaluate('output',{'unsupported_claim_risk':unsupported})
    order=['allow','review','block']
    aggregate=max((inp,out),key=lambda v:order.index(v.decision)).decision.upper()
    message=(f'## Final decision: **{aggregate}**\n'
             f'**Input posterior predictive risk:** {inp.posterior_predictive_risk:.3f} '
             f'| **Output posterior predictive risk:** {out.posterior_predictive_risk:.3f}\n\n'
             'Posterior risk intervals express model-parameter uncertainty, not a guarantee about safety.')
    return message,inp.to_dict(),out.to_dict()


def make_demo():
    with gr.Blocks(title='Bayesian NeMo Guardrails') as demo:
        gr.Markdown('# Bayesian NeMo Guardrails\n'
                    '### Full Bayesian objective · Laplace approximation · Deterministic quadrature\n'
                    'CPU-only nonconjugate Bayesian decision layer for both **LLMRails** and **IORails**.')
        gr.Markdown('**Demo only:** These sliders simulate detector outputs. Models were calibrated on **synthetic labeled data**; '
                    'the Space does not classify prompts or run an LLM. Production use needs real calibrated detectors.')
        with gr.Row():
            with gr.Column(scale=1):
                backend=gr.Radio(['Laplace + Gauss–Hermite','Corrected Gauss–Hermite'],value='Laplace + Gauss–Hermite',label='Deterministic inference')
                injection=gr.Slider(0,1,value=.05,step=.01,label='Input — prompt-injection detector score')
                toxicity=gr.Slider(0,1,value=.05,step=.01,label='Input — toxicity detector score')
                unsupported=gr.Slider(0,1,value=.05,step=.01,label='Output — unsupported-claim detector score')
                run=gr.Button('Evaluate Bayesian guardrails',variant='primary')
            with gr.Column(scale=2):
                headline=gr.Markdown()
                with gr.Row():
                    inp=gr.JSON(label='Input: probability, interval, expected losses')
                    out=gr.JSON(label='Output: probability, interval, expected losses')
        gr.Markdown('**Source:** [GitHub repository](https://github.com/AngshulMajumdar/Bayesian-NeMo-Guardrails) · '
                    '[NVIDIA NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)')
        controls=[backend,injection,toxicity,unsupported]
        run.click(evaluate,inputs=controls,outputs=[headline,inp,out],api_name='evaluate')
        demo.load(evaluate,inputs=controls,outputs=[headline,inp,out])
    return demo

if __name__=='__main__':make_demo().launch(server_name='0.0.0.0',server_port=7860,
    theme=gr.themes.Soft(primary_hue='orange',secondary_hue='amber'),css=CSS)
