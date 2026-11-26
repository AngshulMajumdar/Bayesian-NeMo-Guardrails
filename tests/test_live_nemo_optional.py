"""Actual NeMo 0.25.0 integration tests; optional in local no-NeMo environment."""
from pathlib import Path
import pytest
nemo=pytest.importorskip('nemoguardrails')
from nemoguardrails import Guardrails,RailsConfig
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.rails.llm.options import RailType, RailStatus
from bayesian_nemo import BayesianPolicy,IORailsAdapter,LLMRailsAdapter

ROOT=Path(__file__).resolve().parents[1]
POLICY=BayesianPolicy.from_yaml(ROOT/'examples/policy.yaml')
REGEX_CONFIG={'models':[], 'rails':{'input':{'flows':['regex check input']},
         'config':{'regex_detection':{'input':{'patterns':['SECRET']}}}}}
SAFE={'injection_risk':.02,'toxicity_risk':.02}
HIGH={'injection_risk':.99,'toxicity_risk':.02}

@pytest.mark.parametrize('engine',['legacy','io'])
@pytest.mark.asyncio
async def test_actual_nemo_adapter(engine):
    rails=Guardrails(RailsConfig.from_content(config=REGEX_CONFIG),
          **({'require_iorails':True} if engine=='io' else {'use_iorails':False}))
    adapter=(IORailsAdapter if engine=='io' else LLMRailsAdapter)(rails,POLICY)
    try:
        ok=await adapter.check_async([{'role':'user','content':'hello'}],scores=SAFE,rail_types=[RailType.INPUT])
        assert ok.released and ok.native.status==RailStatus.PASSED
        native_block=await adapter.check_async([{'role':'user','content':'SECRET'}],scores=SAFE,rail_types=[RailType.INPUT])
        assert native_block.decision=='block' and not native_block.released
        bayesian_block=await adapter.check_async([{'role':'user','content':'hello'}],scores=HIGH,rail_types=[RailType.INPUT])
        assert bayesian_block.decision=='block' and not bayesian_block.released
    finally:await rails.shutdown()


@pytest.mark.asyncio
async def test_real_llmrails_colang_action_context():
    config=RailsConfig.from_path(str(ROOT/'examples/legacy_colang'))
    rails=Guardrails(config,use_iorails=False)
    adapter=LLMRailsAdapter(rails,POLICY)
    adapter.register_action()
    scores=dict(SAFE)
    async def trusted_bayesian_detector(context=None):
        return ActionResult(return_value=True,context_updates={'bayesian_nemo_input_scores':dict(scores)})
    rails.register_action(trusted_bayesian_detector,name='trusted_bayesian_detector')
    try:
        for value,status in ((.02,RailStatus.PASSED),(.99,RailStatus.BLOCKED)):
            scores['injection_risk']=value
            checked=await rails.check_async([{'role':'user','content':'hello'}],rail_types=[RailType.INPUT])
            assert checked.status==status
    finally:await rails.shutdown()


def test_iorails_rejects_native_action():
    rails=Guardrails(RailsConfig.from_content(config={'models':[]}),require_iorails=True)
    with pytest.raises(TypeError):LLMRailsAdapter(rails,POLICY)
    with pytest.raises(NotImplementedError):rails.register_action(lambda:True,name='not_supported')

@pytest.mark.parametrize('engine',['legacy','io'])
@pytest.mark.asyncio
async def test_actual_nemo_output_rails(engine):
    output_config={'models':[],'rails':{
        'output':{'flows':['regex check output']},
        'config':{'regex_detection':{'output':{'patterns':['SECRET']}}},
    }}
    rails=Guardrails(RailsConfig.from_content(config=output_config),
          **({'require_iorails':True} if engine=='io' else {'use_iorails':False}))
    adapter=(IORailsAdapter if engine=='io' else LLMRailsAdapter)(rails,POLICY)
    try:
        ok=await adapter.check_async([{'role':'assistant','content':'safe text'}],
            scores={'unsupported_claim_risk':.02},phase='output',rail_types=[RailType.OUTPUT])
        assert ok.released and ok.native.status==RailStatus.PASSED
        blocked_native=await adapter.check_async([{'role':'assistant','content':'SECRET'}],
            scores={'unsupported_claim_risk':.02},phase='output',rail_types=[RailType.OUTPUT])
        assert blocked_native.decision=='block' and blocked_native.native is None
        blocked_bayesian=await adapter.check_async([{'role':'assistant','content':'safe'}],
            scores={'unsupported_claim_risk':.99},phase='output',rail_types=[RailType.OUTPUT])
        assert blocked_bayesian.decision=='block' and blocked_bayesian.native is None
    finally:await rails.shutdown()
