"""Real NVIDIA NeMo IORails input check + Bayesian external wrapper."""
import asyncio
from pathlib import Path
from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.rails.llm.options import RailType
from bayesian_nemo import BayesianPolicy,IORailsAdapter

ROOT=Path(__file__).resolve().parents[2]
CONFIG={'models':[], 'rails':{'input':{'flows':['regex check input']},
         'config':{'regex_detection':{'input':{'patterns':['SECRET']}}}}}

async def main():
    rails=Guardrails(RailsConfig.from_content(config=CONFIG),require_iorails=True)
    adapter=IORailsAdapter(rails,BayesianPolicy.from_yaml(ROOT/'examples/policy.yaml'))
    try:
        examples=[('safe','Hello',.02,'allow'),('nemo-veto','SECRET',.02,'block'),
                  ('bayesian-veto','Hello',.99,'block')]
        for label,txt,score,wanted in examples:
            outcome=await adapter.check_async([{'role':'user','content':txt}],
                scores={'injection_risk':score,'toxicity_risk':.02},rail_types=[RailType.INPUT])
            print(label,outcome.decision)
            assert outcome.decision==wanted
    finally:
        await rails.shutdown()

if __name__=='__main__':asyncio.run(main())
