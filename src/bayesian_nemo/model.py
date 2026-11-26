"""Deterministic approximate Bayesian logistic inference without VB or MCMC.

Model: Bernoulli logistic likelihood and a proper multivariate Gaussian prior.
The Gaussian prior is *not* conjugate to the logistic likelihood.

Backends:
  laplace: Gaussian posterior from a regularized, damped Newton mode fit;
           sigmoid-normal integral evaluated with Gauss-Hermite quadrature.
  corrected_gh: deterministic Laplace-centered, importance-corrected tensor
           Gauss-Hermite approximation to integrals under the *true*
           unnormalized posterior, practical only in low dimension.

Neither backend computes an exact posterior for the nonconjugate model.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import expit, logsumexp
from scipy.stats import norm


class CalibrationError(ValueError):
    """Invalid or untrustworthy model calibration configuration."""


@lru_cache(maxsize=24)
def _hermite_nodes(order: int) -> tuple[np.ndarray,np.ndarray]:
    return np.polynomial.hermite.hermgauss(order)


def _matrix(a: Any, label: str) -> np.ndarray:
    mat = np.asarray(a, dtype=float)
    if mat.ndim != 2 or not np.all(np.isfinite(mat)):
        raise CalibrationError(f"{label} must be a finite 2D matrix")
    return mat


def _vector(a: Any, label: str) -> np.ndarray:
    v = np.asarray(a, dtype=float)
    if v.ndim != 1 or not np.all(np.isfinite(v)):
        raise CalibrationError(f"{label} must be a finite vector")
    return v


def _log_posterior(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                   mean: np.ndarray, precision: np.ndarray) -> float:
    z = X @ theta
    residual = theta - mean
    return float(np.sum(y*z - np.logaddexp(0.0, z)) - 0.5 * residual @ precision @ residual)


def _log_posterior_batch(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                         mean: np.ndarray, precision: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Batched to prevent O(training_size * nodes) resident matrices."""
    out = np.empty(theta.shape[0], dtype=float)
    for start in range(0, len(theta), chunk_size):
        block = theta[start:start+chunk_size]
        z = X @ block.T
        dev = block - mean
        out[start:start+len(block)] = (
            np.sum(y[:, None]*z - np.logaddexp(0., z), axis=0)
            - 0.5*np.einsum('bi,ij,bj->b', dev, precision, dev)
        )
    return out


def _make_design(features: tuple[str,...], values: Mapping[str, float]) -> np.ndarray:
    if not isinstance(values, Mapping) or set(values) != set(features):
        missing = set(features) - set(values) if isinstance(values, Mapping) else set(features)
        raise ValueError(f"Detector score keys must match trained features; missing={sorted(missing)}")
    x = _vector([values[k] for k in features], 'detector scores')
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError('Detector scores must be in [0,1]')
    return np.r_[1., x]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_vals = values[order]
    cum = np.cumsum(weights[order])
    return float(sorted_vals[min(np.searchsorted(cum, q, side='left'), len(cum)-1)])


@dataclass(frozen=True)
class Prediction:
    posterior_predictive_risk: float
    parameter_risk_lower: float
    parameter_risk_upper: float
    integration_backend: str
    credible_mass: float

    def to_dict(self) -> dict:
        return {
            'posterior_predictive_risk': self.posterior_predictive_risk,
            'parameter_risk_interval': [self.parameter_risk_lower, self.parameter_risk_upper],
            'credible_mass': self.credible_mass,
            'integration_backend': self.integration_backend,
        }


@dataclass
class BayesianLogisticModel:
    feature_names: tuple[str, ...]
    posterior_mode: np.ndarray
    covariance: np.ndarray
    prior_mean: np.ndarray
    prior_precision: np.ndarray
    calibration_count: int
    positive_count: int
    backend: str = 'laplace'
    quadrature_order: int = 17
    corrected_nodes: np.ndarray | None = None
    corrected_weights: np.ndarray | None = None
    diagnostics: dict | None = None

    @classmethod
    def fit(cls, X: Any, y: Any, *, feature_names: list[str] | tuple[str,...],
            prior_mean: Any | None = None, prior_precision: Any | None = None,
            backend: str = 'laplace', quadrature_order: int = 17,
            max_iterations: int = 150, tolerance: float = 1e-9,
            min_per_class: int = 5, max_quadrature_nodes: int = 4096) -> 'BayesianLogisticModel':
        X = _matrix(X, 'X')
        y = _vector(y, 'y')
        names = tuple(feature_names)
        if len(names) == 0 or len(set(names)) != len(names) or not all(isinstance(s,str) and s for s in names):
            raise CalibrationError('Distinct nonempty feature names required')
        n, nf = X.shape
        d = nf + 1
        if nf != len(names) or y.shape != (n,) or n == 0:
            raise CalibrationError('Calibration feature/label shapes do not match')
        if np.any(X < 0) or np.any(X > 1):
            raise CalibrationError('Detector scores must lie in [0,1]')
        if not np.all(np.isin(y, [0, 1])):
            raise CalibrationError('Class labels must be binary 0/1')
        positives = int(y.sum())
        if min(positives, n - positives) < min_per_class:
            raise CalibrationError(f'Need at least {min_per_class} labeled examples of each class')
        if backend not in ('laplace', 'corrected_gh'):
            raise CalibrationError('Unsupported Bayesian inference backend')
        if not isinstance(quadrature_order, int) or quadrature_order < 3 or quadrature_order > 35:
            raise CalibrationError('quadrature_order must be integer in [3,35]')
        if not (0 < tolerance < 1):
            raise CalibrationError('tolerance must lie in (0,1)')
        if backend == 'corrected_gh' and quadrature_order**d > max_quadrature_nodes:
            raise CalibrationError('Corrected Gauss-Hermite grid too large; use laplace or fewer features/nodes')
        if max_iterations < 1:
            raise CalibrationError('max_iterations must be positive')
        mean = np.zeros(d) if prior_mean is None else _vector(prior_mean, 'prior_mean')
        precision = np.eye(d) if prior_precision is None else _matrix(prior_precision, 'prior_precision')
        if mean.shape != (d,) or precision.shape != (d,d) or not np.allclose(precision, precision.T, atol=1e-12):
            raise CalibrationError('Prior dimensions or symmetry invalid')
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as exc:
            raise CalibrationError('Gaussian prior precision must be positive definite') from exc
        design = np.c_[np.ones(n), X]
        theta = mean.copy()
        converged = False
        iterations = 0
        for k in range(max_iterations):
            iterations = k + 1
            z = design @ theta
            p = expit(z)
            gradient = design.T @ (y-p) - precision @ (theta-mean)
            curvature = p*(1-p)
            hessian = design.T @ (curvature[:,None]*design) + precision
            step = cho_solve(cho_factor(hessian, lower=True, check_finite=False), gradient, check_finite=False)
            current = _log_posterior(theta, design, y, mean, precision)
            descent = float(gradient @ step)
            line_scale = 1.0
            # Armijo monotonic backtracking for posterior ascent.
            while line_scale >= 2.0**-35:
                trial = theta + line_scale*step
                if _log_posterior(trial, design, y, mean, precision) >= current + 1e-4*line_scale*descent:
                    break
                line_scale *= .5
            if line_scale < 2.0**-35:
                if float(np.max(np.abs(gradient))) < tolerance*10:
                    converged = True
                    break
                raise CalibrationError('Damped Newton line search could not improve posterior')
            next_theta = theta + line_scale * step
            if float(np.max(np.abs(next_theta-theta))) < tolerance*(1 + float(np.max(np.abs(theta)))):
                theta = next_theta
                converged = True
                break
            theta = next_theta
        if not converged:
            raise CalibrationError('Laplace mode optimization did not converge')
        z = design @ theta
        curv = expit(z)*expit(-z)
        hessian = design.T @ (curv[:,None]*design) + precision
        ch = cho_factor(hessian, lower=True, check_finite=False)
        covariance = cho_solve(ch, np.eye(d), check_finite=False)
        covariance = (covariance + covariance.T)/2
        condition = float(np.linalg.cond(hessian))
        # We do not silently treat a numerically ill-conditioned Laplace mode as reliable.
        if not np.isfinite(condition) or condition > 1e12:
            raise CalibrationError('Posterior Hessian badly conditioned; strengthen prior or scale features')
        model = cls(names, theta, covariance, mean, precision, n, positives,
                    backend=backend, quadrature_order=quadrature_order,
                    diagnostics={'converged':True,'iterations':iterations,
                                 'hessian_condition': condition,
                                 'method':'damped_newton_laplace',
                                 'calibration_class_balance': [n-positives,positives]})
        if backend == 'corrected_gh':
            model._prepare_corrected_grid(design, y, max_quadrature_nodes)
        model.validate()
        return model

    def _prepare_corrected_grid(self, design: np.ndarray, y: np.ndarray,
                                max_quadrature_nodes: int) -> None:
        d = len(self.posterior_mode)
        order = self.quadrature_order
        if order ** d > max_quadrature_nodes:
            raise CalibrationError('Quadrature grid exceeds configured node limit')
        ghx, ghw = _hermite_nodes(order)
        # GH nodes have density proportional to exp(-z^2), while q uses exp(-z^2/2).
        idx = np.stack(np.meshgrid(*([np.arange(order)]*d), indexing='ij'),axis=-1).reshape(-1,d)
        std_z = np.sqrt(2.)*ghx[idx]
        log_base_weights = np.sum(np.log(ghw[idx]), axis=1)
        chol = np.linalg.cholesky(self.covariance)
        nodes = self.posterior_mode[None,:] + std_z @ chol.T
        logpost = _log_posterior_batch(nodes, design, y, self.prior_mean,self.prior_precision)
        mode_logpost = _log_posterior(self.posterior_mode,design,y,self.prior_mean,self.prior_precision)
        log_correction = logpost - mode_logpost + .5*np.sum(std_z**2, axis=1)
        logw = log_base_weights + log_correction
        weights = np.exp(logw - logsumexp(logw))
        if np.any(~np.isfinite(weights)) or not np.isclose(np.sum(weights),1.):
            raise CalibrationError('Nonfinite corrected quadrature weights')
        self.corrected_nodes = nodes
        self.corrected_weights = weights
        effective = float(1./np.sum(weights**2))
        self.diagnostics = dict(self.diagnostics or {}, quadrature_nodes=len(weights),
                                quadrature_effective_nodes=effective,
                                quadrature_warning=(effective < min(10., .01*len(weights))))

    def validate(self) -> None:
        names = self.feature_names
        d = len(names)+1
        if not names or len(set(names))!=len(names):
            raise CalibrationError('Invalid trained feature names')
        self.posterior_mode = _vector(self.posterior_mode,'posterior_mode')
        self.prior_mean = _vector(self.prior_mean,'prior_mean')
        self.covariance = _matrix(self.covariance,'covariance')
        self.prior_precision = _matrix(self.prior_precision,'prior_precision')
        if self.posterior_mode.shape!=(d,) or self.prior_mean.shape!=(d,) or self.covariance.shape!=(d,d) or self.prior_precision.shape!=(d,d):
            raise CalibrationError('Model parameter dimension mismatch')
        for name,mat in (('covariance',self.covariance),('prior_precision',self.prior_precision)):
            if not np.allclose(mat,mat.T,atol=1e-10):
                raise CalibrationError(f'{name} not symmetric')
            try:np.linalg.cholesky(mat)
            except np.linalg.LinAlgError as exc:raise CalibrationError(f'{name} must be SPD') from exc
        if not 0<=self.positive_count<=self.calibration_count or self.calibration_count<1:
            raise CalibrationError('Invalid calibration counts')
        if self.backend not in ('laplace','corrected_gh') or not 3<=self.quadrature_order<=35:
            raise CalibrationError('Invalid posterior backend or order')
        if self.backend == 'corrected_gh':
            if self.corrected_nodes is None or self.corrected_weights is None:
                raise CalibrationError('Missing corrected quadrature support')
            self.corrected_nodes = _matrix(self.corrected_nodes,'corrected_nodes')
            self.corrected_weights = _vector(self.corrected_weights,'corrected_weights')
            if self.corrected_nodes.shape!=(len(self.corrected_weights),d):
                raise CalibrationError('Corrected quadrature shape mismatch')
            if np.any(self.corrected_weights<0) or not np.isclose(self.corrected_weights.sum(),1.,atol=1e-8):
                raise CalibrationError('Invalid quadrature weights')

    def predict(self, scores: Mapping[str,float], *, credible_mass: float=0.95) -> Prediction:
        if not 0 < credible_mass < 1:
            raise ValueError('credible_mass must lie in (0,1)')
        x = _make_design(self.feature_names,scores)
        alpha = (1.-credible_mass)/2.
        if self.backend == 'corrected_gh':
            probs = expit(self.corrected_nodes @ x)
            w = self.corrected_weights
            risk = float(w @ probs)
            lower = _weighted_quantile(probs,w,alpha)
            upper = _weighted_quantile(probs,w,1-alpha)
        else:
            mean = float(x @ self.posterior_mode)
            v = max(0.,float(x @ self.covariance @ x))
            z, w = _hermite_nodes(self.quadrature_order)
            risk = float(w @ expit(mean + np.sqrt(2.*v)*z) / np.sqrt(np.pi))
            radius = float(norm.ppf(1.-alpha)) * np.sqrt(v)
            lower,upper = float(expit(mean-radius)),float(expit(mean+radius))
        return Prediction(float(np.clip(risk,0,1)),float(np.clip(lower,0,1)),
                          float(np.clip(upper,0,1)),self.backend,credible_mass)

    def to_dict(self) -> dict:
        return {
            'schema_version':1, 'model':'nonconjugate_bayesian_logistic',
            'feature_names':list(self.feature_names),
            'posterior_mode':self.posterior_mode.tolist(),
            'covariance':self.covariance.tolist(),
            'prior_mean':self.prior_mean.tolist(),
            'prior_precision':self.prior_precision.tolist(),
            'calibration_count':self.calibration_count,
            'positive_count':self.positive_count,
            'backend':self.backend,
            'quadrature_order':self.quadrature_order,
            'corrected_nodes':None if self.corrected_nodes is None else self.corrected_nodes.tolist(),
            'corrected_weights':None if self.corrected_weights is None else self.corrected_weights.tolist(),
            'diagnostics':self.diagnostics or {},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str,Any]) -> 'BayesianLogisticModel':
        if data.get('schema_version') != 1 or data.get('model') != 'nonconjugate_bayesian_logistic':
            raise CalibrationError('Unsupported/invalid model snapshot')
        m=cls(feature_names=tuple(data['feature_names']),
              posterior_mode=np.asarray(data['posterior_mode'],dtype=float),
              covariance=np.asarray(data['covariance'],dtype=float),
              prior_mean=np.asarray(data['prior_mean'],dtype=float),
              prior_precision=np.asarray(data['prior_precision'],dtype=float),
              calibration_count=int(data['calibration_count']),
              positive_count=int(data['positive_count']),
              backend=data['backend'], quadrature_order=int(data['quadrature_order']),
              corrected_nodes=None if data.get('corrected_nodes') is None else np.asarray(data['corrected_nodes'],dtype=float),
              corrected_weights=None if data.get('corrected_weights') is None else np.asarray(data['corrected_weights'],dtype=float),
              diagnostics=dict(data.get('diagnostics') or {}))
        m.validate()
        return m
