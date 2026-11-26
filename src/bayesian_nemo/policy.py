"""Auditable Bayesian decision policy using calibrated binary-risk predictions."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json
import math
import yaml
from .model import BayesianLogisticModel, Prediction


@dataclass(frozen=True)
class Verdict:
    phase: str
    decision: str
    reason: str
    posterior_predictive_risk: float | None = None
    parameter_risk_interval: tuple[float,float] | None = None
    expected_losses: dict[str,float] = field(default_factory=dict)
    hard_vetoes: tuple[str,...] = ()
    credible_mass: float | None = None
    inference_backend: str | None = None

    def to_dict(self) -> dict:
        return {'phase':self.phase,'decision':self.decision,'reason':self.reason,
                'posterior_predictive_risk':self.posterior_predictive_risk,
                'parameter_risk_interval':list(self.parameter_risk_interval) if self.parameter_risk_interval else None,
                'expected_losses':self.expected_losses,'hard_vetoes':list(self.hard_vetoes),
                'credible_mass':self.credible_mass,'inference_backend':self.inference_backend}


@dataclass
class Phase:
    model: BayesianLogisticModel
    losses: dict[str,tuple[float,float]]
    hard_veto: dict[str,float]
    credible_mass: float
    allow_upper_bound_max: float | None = None
    block_lower_bound_min: float | None = None

    def __post_init__(self):
        if set(self.losses) != {'allow','review','block'}:
            raise ValueError('Each phase must specify allow, review and block losses')
        for name,v in self.losses.items():
            if len(v)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in v):
                raise ValueError(f'Invalid loss for {name}; expected [safe_cost,unsafe_cost]')
        if not 0 < self.credible_mass < 1:
            raise ValueError('credible_mass must be in (0,1)')
        for feat,threshold in self.hard_veto.items():
            if feat not in self.model.feature_names or not 0 <= threshold <= 1:
                raise ValueError('Veto feature or threshold invalid')
        for threshold in (self.allow_upper_bound_max,self.block_lower_bound_min):
            if threshold is not None and not 0<=threshold<=1:
                raise ValueError('Credible-bound thresholds must be in [0,1]')


class BayesianPolicy:
    """Model and cost specification for phases, completely independent of NeMo."""
    def __init__(self, phases: Mapping[str,Phase]):
        self.phases=dict(phases)
        if not self.phases or any(not isinstance(x,Phase) for x in self.phases.values()):
            raise ValueError('At least one valid policy phase required')

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'BayesianPolicy':
        path=Path(path).resolve()
        config=yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(config,dict) or config.get('schema_version') != 1:
            raise ValueError('Policy YAML schema_version must equal 1')
        phases={}
        for name,item in config.get('phases',{}).items():
            model_path=(path.parent/item['model']).resolve()
            model=BayesianLogisticModel.from_dict(json.loads(model_path.read_text(encoding='utf-8')))
            losses={k:tuple(v) for k,v in item['losses'].items()}
            phases[name]=Phase(model,losses,dict(item.get('hard_veto',{})),
                               float(item.get('credible_mass',.95)),
                               item.get('allow_upper_bound_max'),item.get('block_lower_bound_min'))
        return cls(phases)

    def evaluate(self, phase: str, scores: Mapping[str,float] | None) -> Verdict:
        if phase not in self.phases:
            return Verdict(phase,'review','unknown phase; no model configured')
        spec=self.phases[phase]
        if not isinstance(scores, Mapping):
            return Verdict(phase,'review','trusted detector scores missing')
        missing = sorted(set(spec.model.feature_names) - set(scores))
        if missing:
            return Verdict(phase,'review',f'missing mandatory detector scores: {missing}')
        # Reject extra score names rather than silently ignoring them.
        if set(scores)!=set(spec.model.feature_names):
            return Verdict(phase,'review','unexpected detector score keys')
        try:
            numeric={key:float(scores[key]) for key in spec.model.feature_names}
            if any(not math.isfinite(v) or v<0 or v>1 for v in numeric.values()):
                raise ValueError('Detector scores must be finite and in [0,1]')
            prediction=spec.model.predict(numeric,credible_mass=spec.credible_mass)
        except (TypeError,ValueError,OverflowError) as exc:
            return Verdict(phase,'review',f'invalid detector evidence: {exc}')
        p=prediction.posterior_predictive_risk
        expected={a:cost[0]*(1.-p)+cost[1]*p for a,cost in spec.losses.items()}
        decision=min(('allow','review','block'), key=lambda a:(expected[a],('review','block','allow').index(a)))
        rationale='minimum posterior expected decision loss'
        veto=tuple(k for k,threshold in spec.hard_veto.items() if numeric[k]>=threshold)
        if veto:
            decision='block'; rationale='hard safety veto'
        elif decision=='allow' and spec.allow_upper_bound_max is not None and prediction.parameter_risk_upper>spec.allow_upper_bound_max:
            decision='review'; rationale='upper credible risk exceeds allowed bound'
        elif decision=='block' and spec.block_lower_bound_min is not None and prediction.parameter_risk_lower<spec.block_lower_bound_min:
            decision='review'; rationale='lower credible risk insufficient for automatic block'
        return Verdict(phase,decision,rationale,p,
                       (prediction.parameter_risk_lower,prediction.parameter_risk_upper),
                       expected,veto,prediction.credible_mass,prediction.integration_backend)
