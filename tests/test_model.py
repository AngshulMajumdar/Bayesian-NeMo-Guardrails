import json
import numpy as np
import pytest
from scipy.special import expit
from bayesian_nemo.model import BayesianLogisticModel, CalibrationError


def dataset(n=120,seed=2):
    rng=np.random.default_rng(seed)
    X=rng.uniform(0,1,(n,1))
    y=rng.binomial(1,expit(-4 + 7*X[:,0]))
    return X,y

@pytest.mark.parametrize('backend,order', [('laplace',15),('corrected_gh',11)])
def test_valid_probability_and_intervals(backend,order):
    X,y=dataset()
    m=BayesianLogisticModel.fit(X,y,feature_names=['risk'],backend=backend,quadrature_order=order)
    pred=m.predict({'risk':.7})
    assert 0 <= pred.parameter_risk_lower <= pred.posterior_predictive_risk <= pred.parameter_risk_upper <=1
    assert m.diagnostics['converged']
    restored=BayesianLogisticModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert restored.predict({'risk':.7}).posterior_predictive_risk==pytest.approx(pred.posterior_predictive_risk)


def test_monotonic_on_synthetic():
    X,y=dataset(n=400)
    m=BayesianLogisticModel.fit(X,y,feature_names=['risk'])
    probs=[m.predict({'risk':v}).posterior_predictive_risk for v in (.01,.3,.6,.99)]
    assert probs==sorted(probs)


def test_corrected_quadrature_vs_refinement():
    X,y=dataset(80)
    rough=BayesianLogisticModel.fit(X,y,feature_names=['risk'],backend='corrected_gh',quadrature_order=9)
    fine=BayesianLogisticModel.fit(X,y,feature_names=['risk'],backend='corrected_gh',quadrature_order=19)
    for x in [.1,.25,.7,.9]:
        a=rough.predict({'risk':x}).posterior_predictive_risk
        b=fine.predict({'risk':x}).posterior_predictive_risk
        assert abs(a-b)<0.005


def test_gaussian_vs_posterior_corrected_small_disagreement():
    X,y=dataset(300)
    ordinary=BayesianLogisticModel.fit(X,y,feature_names=['risk'],backend='laplace',quadrature_order=21)
    corrected=BayesianLogisticModel.fit(X,y,feature_names=['risk'],backend='corrected_gh',quadrature_order=19)
    # This is a sanity test, not an assertion that Laplace is always accurate.
    assert abs(ordinary.predict({'risk':.5}).posterior_predictive_risk - corrected.predict({'risk':.5}).posterior_predictive_risk)<.04


def test_prior_regularization_prevents_separation_failure():
    X=np.r_[np.full((20,1),.01),np.full((20,1),.99)]
    y=np.r_[np.zeros(20),np.ones(20)]
    m=BayesianLogisticModel.fit(X,y,feature_names=['risk'],prior_precision=np.eye(2)*.1)
    assert np.all(np.isfinite(m.covariance))


def test_refuse_bad_labels_or_missing_class():
    X,y=dataset()
    with pytest.raises(CalibrationError): BayesianLogisticModel.fit(X,np.full(len(y),0),feature_names=['risk'])
    y[0]=2
    with pytest.raises(CalibrationError): BayesianLogisticModel.fit(X,y,feature_names=['risk'])


def test_refuse_bad_scores():
    X,y=dataset();m=BayesianLogisticModel.fit(X,y,feature_names=['risk'])
    for bad in [{'risk':float('nan')},{'risk':-0.1},{'risk':1.1},{}, {'risk':.5,'other':.2}]:
        with pytest.raises(ValueError):m.predict(bad)


def test_refuse_invalid_precision_and_bad_grids():
    X,y=dataset()
    with pytest.raises(CalibrationError):BayesianLogisticModel.fit(X,y,feature_names=['risk'],prior_precision=np.zeros((2,2)))
    with pytest.raises(CalibrationError):BayesianLogisticModel.fit(X,y,feature_names=['risk'],backend='corrected_gh',quadrature_order=35,max_quadrature_nodes=100)


def test_refuse_unknown_backend():
    X,y=dataset()
    with pytest.raises(CalibrationError):BayesianLogisticModel.fit(X,y,feature_names=['risk'],backend='variational_bayes')


def test_uncertainty_with_more_data():
    X,y=dataset(n=1000)
    small=BayesianLogisticModel.fit(X[:120],y[:120],feature_names=['risk'])
    big=BayesianLogisticModel.fit(X,y,feature_names=['risk'])
    p1=small.predict({'risk':.5});p2=big.predict({'risk':.5})
    assert p2.parameter_risk_upper-p2.parameter_risk_lower < p1.parameter_risk_upper-p1.parameter_risk_lower


def test_finite_serialization_requires_positive_covariance():
    X,y=dataset();m=BayesianLogisticModel.fit(X,y,feature_names=['risk'])
    snap=m.to_dict();snap['covariance']=[[1,2],[2,1]]
    with pytest.raises(CalibrationError):BayesianLogisticModel.from_dict(snap)
