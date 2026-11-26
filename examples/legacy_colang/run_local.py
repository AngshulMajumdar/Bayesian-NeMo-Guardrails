"""Real NVIDIA NeMo 0.25.0 Colang 1.0 action demo, no LLM or GPU.
All detector scores and calibration labels are synthetic (demo only).
"""
import asyncio
from pathlib import Path
from nemoguardrails import Guardrails,RailsConfig
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.rails.llm.options import RailType
from bayesian_nemo import BayesianPolicy, LLMRailsAdapter

ROOT=Path(__file__).resolve().parents[2]
POLICY=BayesianPolicy.from_yaml(ROOT/'examples/policy.yaml')
SCORES={'injection_risk':.02,'toxicity_risk':.02}

async def main():
    config=RailsConfig.from_path(str(Path(__file__).parent))
    rails=Guardrails(config,use_iorails=False)
    adapter=LLMRailsAdapter(rails,POLICY)
    adapter.register_action()
    async def trusted_bayesian_detector(context=None):
        return ActionResult(return_value=True,context_updates={'bayesian_nemo_input_scores':dict(SCORES)})
    rails.register_action(trusted_bayesian_detector,name='trusted_bayesian_detector')
    try:
        for label,score,wanted in [('safe',.02,'passed'),('blocked',.99,'blocked')]:
            SCORES['injection_risk']=score
            result=await rails.check_async([{'role':'user','content':'Hello'}],rail_types=[RailType.INPUT])
            print(label,result.status.value)
            assert result.status.value==wanted
    finally:
        await rails.shutdown()

if __name__=='__main__':asyncio.run(main())
