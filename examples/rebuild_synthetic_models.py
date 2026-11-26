"""Rebuild EXACT demo datasets and snapshots from a fixed RNG seed.

Everything created here is synthetic. Never mistake these for real risk calibration.
Run from repository root: python examples/rebuild_synthetic_models.py
"""
import csv
from pathlib import Path
import numpy as np
from bayesian_nemo.calibration import calibrate
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
RNG=np.random.default_rng(20261126)

for phase,features,n,coefficients in (
    ('input',['injection_risk','toxicity_risk'],850,[-6.0,10.5,3.0]),
    ('output',['unsupported_claim_risk'],600,[-6.0,11.0]),
):
    X=RNG.beta(1.0,1.0,size=(n,len(features)))
    z=np.c_[np.ones(n),X] @ coefficients
    y=(RNG.random(n)<expit(z)).astype(int)
    csv_file=ROOT/'examples/data'/f'synthetic_{phase}.csv'
    csv_file.parent.mkdir(parents=True,exist_ok=True)
    with csv_file.open('w',newline='') as handle:
        writer=csv.writer(handle); writer.writerow([*features,'unsafe'])
        writer.writerows([*r,int(label)] for r,label in zip(X,y))
    for backend,order,suffix in (
        ('laplace',17,'laplace'),('corrected_gh',11,'corrected'),
    ):
        model_file=ROOT/'src/bayesian_nemo/models'/f'{phase}_{suffix}.json'
        fitted=calibrate(csv_file,model_file,features=features,backend=backend,
                         quadrature_order=order,prior_diagonal=[.3]+[.15]*len(features))
        print(phase,backend,'n',fitted.calibration_count,'unsafe',fitted.positive_count)
