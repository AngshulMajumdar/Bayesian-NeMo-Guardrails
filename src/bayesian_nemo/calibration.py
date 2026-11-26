"""Offline, labeled calibration only. Never learn from unreviewed live traffic."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Sequence
import numpy as np
from .model import BayesianLogisticModel


def read_csv(path:str|Path, features:Sequence[str], label:str='unsafe') -> tuple[np.ndarray,np.ndarray]:
    with Path(path).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError('Calibration CSV must contain labeled examples')
    try:
        X=np.asarray([[float(r[name]) for name in features] for r in rows],dtype=float)
        y=np.asarray([float(r[label]) for r in rows],dtype=float)
    except KeyError as e:
        raise ValueError(f'Missing required CSV column: {e}') from e
    return X,y


def calibrate(csv_path:str|Path, output_path:str|Path, *,
              features:Sequence[str], label:str='unsafe',
              backend:str='laplace', quadrature_order:int=17,
              prior_diagonal:Sequence[float]|None=None,
              prior_mean:Sequence[float]|None=None,
              max_iterations:int=150, tolerance:float=1e-9,
              min_per_class:int=5) -> BayesianLogisticModel:
    X,y=read_csv(csv_path,features,label)
    d=len(features)+1
    if prior_diagonal is None: prior_diagonal=[1.0]*d
    prior_diagonal=np.asarray(prior_diagonal,dtype=float)
    if prior_diagonal.shape!=(d,) or not np.all(np.isfinite(prior_diagonal)) or np.any(prior_diagonal<=0):
        raise ValueError('prior_diagonal must contain d strictly positive precision values')
    model=BayesianLogisticModel.fit(
        X,y,feature_names=list(features),prior_mean=prior_mean,
        prior_precision=np.diag(prior_diagonal),backend=backend,
        quadrature_order=quadrature_order,max_iterations=max_iterations,
        tolerance=tolerance,min_per_class=min_per_class)
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    Path(output_path).write_text(json.dumps(model.to_dict(),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return model
