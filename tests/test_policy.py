import pytest
from bayesian_nemo import BayesianPolicy, Phase


def test_all_three_decisions(policy):
    for score,expected in ((.02,'allow'),(.5,'review'),(.8,'block'),(.99,'block')):
        v=policy.evaluate('input',{'injection_risk':score,'toxicity_risk':.02})
        assert v.decision==expected
        assert set(v.expected_losses)=={'allow','review','block'}
        assert v.posterior_predictive_risk is not None


def test_hard_veto(policy):
    v=policy.evaluate('input',{'injection_risk':.99,'toxicity_risk':.02})
    assert v.decision=='block' and v.hard_vetoes==('injection_risk',)

@pytest.mark.parametrize('scores', [None,{}, {'injection_risk':.2},
    {'injection_risk':float('nan'),'toxicity_risk':.2},
    {'injection_risk':1.5,'toxicity_risk':.2},
    {'injection_risk':.2,'toxicity_risk':.2,'unexpected':.4}])
def test_bad_scores_fail_closed(policy,scores):
    assert policy.evaluate('input',scores).decision=='review'


def test_missing_phase_fails_closed(policy):
    assert policy.evaluate('retrieval',{}).decision=='review'


def test_corrected_policy_predicts(corrected_policy):
    v=corrected_policy.evaluate('output',{'unsupported_claim_risk':.9})
    assert v.inference_backend=='corrected_gh'
    assert v.decision=='block'


def test_interval_gate_can_refuse_allow(policy):
    phase=policy.phases['input']
    stricter=Phase(phase.model,phase.losses,phase.hard_veto,phase.credible_mass,allow_upper_bound_max=0.001)
    p=BayesianPolicy({'input':stricter})
    assert p.evaluate('input',{'injection_risk':.02,'toxicity_risk':.02}).decision=='review'


def test_loss_matrix_changes_decision(policy):
    phase=policy.phases['input']
    losses={'allow':(0,1),'review':(4,4),'block':(4,0)}
    p=BayesianPolicy({'input':Phase(phase.model,losses,{},.95)})
    v=p.evaluate('input',{'injection_risk':.5,'toxicity_risk':.02})
    assert v.decision=='allow'


def test_wrong_cost_shape_fails(policy):
    phase=policy.phases['input']
    with pytest.raises(ValueError):Phase(phase.model,{'allow':(1,),'review':(1,1),'block':(1,1)},{},.95)
