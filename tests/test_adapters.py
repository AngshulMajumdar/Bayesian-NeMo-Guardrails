import asyncio
from dataclasses import dataclass
import pytest
from bayesian_nemo import IORailsAdapter, LLMRailsAdapter

@dataclass
class StubStatus:
    value:str

@dataclass
class StubResponse:
    status:StubStatus
    content:str='safe response'

class FakeGuardrails:
    def __init__(self, io:bool):
        self.use_iorails_engine=io
        self.status='passed'
        self.check_calls=0
        self.generate_calls=0
        self.actions={}
    async def check_async(self, **kwargs):
        self.check_calls+=1
        return StubResponse(StubStatus(self.status),'NeMo checked text')
    async def generate_async(self, **kwargs):
        self.generate_calls+=1
        return {'role':'assistant','content':'test generated'}
    def register_action(self,action,name):self.actions[name]=action

SAFE={'injection_risk':.02,'toxicity_risk':.02}
HIGH={'injection_risk':.99,'toxicity_risk':.02}

@pytest.mark.parametrize('io',[True,False])
@pytest.mark.asyncio
async def test_native_blocks_and_fuzzy_hard_veto(policy,io):
    guard=FakeGuardrails(io)
    adapter=(IORailsAdapter if io else LLMRailsAdapter)(guard,policy)
    ok=await adapter.check_async([{'role':'user','content':'x'}],scores=SAFE,rail_types=['input'])
    assert ok.released and ok.checked_content=='NeMo checked text'
    guard.status='blocked'
    block=await adapter.check_async([{'role':'user','content':'x'}],scores=SAFE,rail_types=['input'])
    assert block.decision=='block' and block.native is None
    previous=guard.check_calls
    blocked=await adapter.check_async([{'role':'user','content':'x'}],scores=HIGH,rail_types=['input'])
    assert blocked.decision=='block' and guard.check_calls==previous


@pytest.mark.asyncio
async def test_native_modify_and_unknown_status(policy):
    guard=FakeGuardrails(io=True)
    adapter=IORailsAdapter(guard,policy)
    guard.status='modified'
    resp=await adapter.check_async([{'role':'user','content':'original'}],scores=SAFE,rail_types=['input'])
    assert resp.released and resp.checked_content=='NeMo checked text'
    guard.status='unknown'
    resp=await adapter.check_async([{'role':'user','content':'x'}],scores=SAFE,rail_types=['input'])
    assert resp.decision=='review' and not resp.released

@pytest.mark.asyncio
async def test_review_without_native_invocation(policy):
    guard=FakeGuardrails(io=True)
    adapter=IORailsAdapter(guard,policy)
    resp=await adapter.check_async([{'role':'user','content':'x'}],scores=None,rail_types=['input'])
    assert resp.decision=='review' and guard.check_calls==0

@pytest.mark.asyncio
async def test_generation_scored_before_release(policy):
    guard=FakeGuardrails(io=True)
    adapter=IORailsAdapter(guard,policy)
    missing=await adapter.generate_async([{'role':'user','content':'ask'}],input_scores=SAFE)
    assert missing.decision=='review' and guard.generate_calls==0
    blocked=await adapter.generate_async([{'role':'user','content':'ask'}],input_scores=SAFE,
                                         output_scorer=lambda _: {'unsupported_claim_risk':.99})
    assert blocked.decision=='block' and blocked.native is None
    ok=await adapter.generate_async([{'role':'user','content':'ask'}],input_scores=SAFE,
                                         output_scorer=lambda _: {'unsupported_claim_risk':.02})
    assert ok.released

@pytest.mark.asyncio
async def test_async_output_scorer_and_exception(policy):
    guard=FakeGuardrails(io=True)
    adapter=IORailsAdapter(guard,policy)
    async def scoring(x):return {'unsupported_claim_risk':.02}
    assert (await adapter.generate_async([{'role':'user','content':'x'}],input_scores=SAFE,output_scorer=scoring)).released
    def broken(x):raise RuntimeError('no ground truth')
    denied=await adapter.generate_async([{'role':'user','content':'x'}],input_scores=SAFE,output_scorer=broken)
    assert denied.decision=='review' and not denied.released

@pytest.mark.asyncio
async def test_legacy_action(policy):
    guard=FakeGuardrails(io=False)
    adapter=LLMRailsAdapter(guard,policy)
    adapter.register_action()
    action=guard.actions['bayesian_nemo_input']
    assert await action(context={'bayesian_nemo_input_scores':SAFE})
    assert not await action(context={'bayesian_nemo_input_scores':HIGH})
    assert not await action(context={})


def test_wrong_engines_are_rejected(policy):
    with pytest.raises(TypeError):LLMRailsAdapter(FakeGuardrails(True),policy)
    with pytest.raises(TypeError):IORailsAdapter(FakeGuardrails(False),policy)

@pytest.mark.asyncio
async def test_multi_phase_must_be_explicit(policy):
    adapter=IORailsAdapter(FakeGuardrails(True),policy)
    with pytest.raises(ValueError):
        await adapter.check_async([{'role':'user','content':'x'},{'role':'assistant','content':'y'}],scores=SAFE)
    with pytest.raises(ValueError):
        await adapter.check_async([{'role':'user','content':'x'}],scores=SAFE,rail_types=['output'])

@pytest.mark.asyncio
async def test_no_streaming(policy):
    adapter=IORailsAdapter(FakeGuardrails(True),policy)
    with pytest.raises(ValueError):
        await adapter.generate_async([{'role':'user','content':'x'}],input_scores=SAFE,stream=True)

@pytest.mark.asyncio
async def test_generation_native_block_cannot_be_overridden(policy):
    class NativeBlocked(FakeGuardrails):
        async def generate_async(self,**kwargs):
            return StubResponse(StubStatus('blocked'),'native refusal')
    guard=NativeBlocked(True)
    adapter=IORailsAdapter(guard,policy)
    result=await adapter.generate_async([{'role':'user','content':'x'}],input_scores=SAFE,
         output_scorer=lambda _: {'unsupported_claim_risk':.02})
    assert result.decision=='block' and result.native is None

@pytest.mark.asyncio
async def test_output_phase_native_and_bayesian_blocks(policy):
    guard=FakeGuardrails(io=True)
    adapter=IORailsAdapter(guard,policy)
    safe=await adapter.check_async([{'role':'assistant','content':'safe'}],
        scores={'unsupported_claim_risk':.02},phase='output',rail_types=['output'])
    assert safe.released
    guard.status='blocked'
    blocked=await adapter.check_async([{'role':'assistant','content':'SECRET'}],
        scores={'unsupported_claim_risk':.02},phase='output',rail_types=['output'])
    assert blocked.decision=='block' and not blocked.released
    blocked_risk=await adapter.check_async([{'role':'assistant','content':'safe'}],
        scores={'unsupported_claim_risk':.99},phase='output',rail_types=['output'])
    assert blocked_risk.decision=='block' and not blocked_risk.released
