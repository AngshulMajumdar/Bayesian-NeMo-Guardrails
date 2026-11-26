"""Evaluation API. The demo model is synthetic, not a production classifier."""
from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from .policy import BayesianPolicy

DEFAULT_CONFIG=Path(__file__).with_name('default_policy.yaml')
config_path=Path(os.environ.get('BAYESIAN_NEMO_CONFIG',str(DEFAULT_CONFIG)))
policy=BayesianPolicy.from_yaml(config_path)
app=FastAPI(title='Bayesian NeMo Guardrails',version='0.1.0',
            description='Deterministic approximate Bayesian risk inference; demo models trained on synthetic data.')

class EvaluationRequest(BaseModel):
    phase:str
    scores:dict[str,float]=Field(default_factory=dict)

@app.get('/health')
def health():
    return {'status':'ok','package':'bayesian-nemo-guardrails','phases':list(policy.phases)}

@app.get('/config')
def config():
    return {'phases':{name:{'features':list(spec.model.feature_names),
                            'backend':spec.model.backend,
                            'calibration_count':spec.model.calibration_count,
                            'example_data_only':True} for name,spec in policy.phases.items()}}

@app.post('/evaluate')
def evaluate(request:EvaluationRequest):
    if request.phase not in policy.phases:
        raise HTTPException(status_code=400,detail='Unknown evaluation phase')
    return policy.evaluate(request.phase,request.scores).to_dict()
