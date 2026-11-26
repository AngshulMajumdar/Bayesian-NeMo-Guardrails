"""Fail-closed NeMo Guardrails 0.25.0 integration.

Native NeMo checks never get overridden by Bayesian policy.
LLMRails may optionally register a Colang action supplied trusted context scores.
IORails is wrapped externally; it does not support custom Python actions.
"""
from __future__ import annotations
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Callable, Mapping
from .policy import BayesianPolicy, Verdict


@dataclass(frozen=True)
class NativeOutcome:
    decision: str
    phase: str
    bayesian: Verdict
    native: Any = None
    reason: str = ''

    @property
    def released(self) -> bool:
        return self.decision=='allow' and self.native is not None

    @property
    def checked_content(self) -> str | None:
        # For native MODIFIED, expose ONLY the NeMo-checked replacement.
        return getattr(self.native,'content',None) if self.released else None


def _native_status(native: Any) -> str | None:
    value = getattr(native,'status',None)
    value = getattr(value,'value',value)
    return value.lower() if isinstance(value,str) else None


def _engine_name(rails: Any) -> str | None:
    # Guardrails facade or raw engine; actual types validated at the boundary.
    if hasattr(rails,'use_iorails_engine'):
        return 'io' if rails.use_iorails_engine else 'legacy'
    if type(rails).__name__ == 'IORails': return 'io'
    if type(rails).__name__ == 'LLMRails': return 'legacy'
    return None


class BayesianNeMoAdapter:
    def __init__(self, rails:Any, policy: BayesianPolicy):
        if not callable(getattr(rails,'check_async',None)) or not callable(getattr(rails,'generate_async',None)):
            raise TypeError('rails must expose check_async() and generate_async()')
        if not isinstance(policy,BayesianPolicy):
            raise TypeError('policy must be a BayesianPolicy')
        self.rails,self.policy=rails,policy

    async def check_async(self, messages:list[dict], *, scores:Mapping[str,float] | None,
                          phase:str='input', rail_types:list | None=None, **kwargs:Any) -> NativeOutcome:
        if not messages or not all(isinstance(m,dict) and m.get('role') in ('user','assistant','system','tool') for m in messages):
            raise ValueError('messages require well-formed role dictionaries')
        roles={m['role'] for m in messages}
        if rail_types is None:
            if 'assistant' in roles and 'user' in roles:
                raise ValueError('Mixed user/assistant checks require explicit single rail_types')
            expected='output' if 'assistant' in roles else 'input'
            if expected != phase:
                raise ValueError(f'Messages indicate {expected} phase, got {phase}')
        else:
            requested=[getattr(t,'value',t) for t in rail_types]
            if requested != [phase]:
                raise ValueError('rail_types must be exactly the requested fuzzy phase')
        verdict=self.policy.evaluate(phase,scores)
        if verdict.decision!='allow':
            return NativeOutcome(verdict.decision,phase,verdict,reason=verdict.reason)
        native=await self.rails.check_async(messages=messages,rail_types=rail_types,**kwargs)
        status=_native_status(native)
        if status=='blocked':
            return NativeOutcome('block',phase,verdict,reason='blocked by native NeMo')
        if status not in ('passed','modified'):
            return NativeOutcome('review',phase,verdict,reason='unknown native NeMo status; fail closed')
        return NativeOutcome('allow',phase,verdict,native,
                             reason='release only NeMo modified content' if status=='modified' else '')

    async def generate_async(self,messages:list[dict],*,
                             input_scores:Mapping[str,float] | None,
                             output_scorer:Callable[[Any],Any] | None=None,
                             **kwargs:Any)->NativeOutcome:
        """Do not stream; checks take place before releasing generated content.

        Native NeMo generation retains its configured native input/output rails.
        Generated content is withheld if post-generation detector is missing.
        """
        if kwargs.get('stream',False) or kwargs.get('streaming',False):
            raise ValueError('Unchecked streaming is disabled')
        verdict=self.policy.evaluate('input',input_scores)
        if verdict.decision!='allow':
            return NativeOutcome(verdict.decision,'input',verdict,reason=verdict.reason)
        if 'output' in self.policy.phases and output_scorer is None:
            return NativeOutcome('review','output',verdict,reason='trusted output scorer required')
        native=await self.rails.generate_async(messages=messages,**kwargs)
        if native is None:
            return NativeOutcome('review','output',verdict,reason='empty NeMo generation')
        native_status=_native_status(native)
        if native_status=='blocked':
            return NativeOutcome('block','output',verdict,reason='blocked by native NeMo generation rail')
        if native_status is not None and native_status not in ('passed','modified'):
            return NativeOutcome('review','output',verdict,reason='unknown native generation status; fail closed')
        if 'output' not in self.policy.phases:
            return NativeOutcome('allow','input',verdict,native)
        try:
            scores=output_scorer(native)
            if isawaitable(scores): scores=await scores
            output=self.policy.evaluate('output',scores)
        except Exception as exc:
            # A scoring exception is not an authorization to release.
            return NativeOutcome('review','output',verdict,reason=f'output scorer failed: {type(exc).__name__}')
        if output.decision!='allow':
            return NativeOutcome(output.decision,'output',output,reason=output.reason)
        return NativeOutcome('allow','output',output,native)


class LLMRailsAdapter(BayesianNeMoAdapter):
    def __init__(self, rails:Any, policy:BayesianPolicy):
        super().__init__(rails,policy)
        if _engine_name(rails)!='legacy':
            raise TypeError('LLMRailsAdapter requires an LLMRails engine')

    def register_action(self,name:str='bayesian_nemo_input')->None:
        if not callable(getattr(self.rails,'register_action',None)):
            raise TypeError('Engine does not support Colang action registration')
        policy=self.policy
        async def bayesian_action(context:dict|None=None)->bool:
            context=context or {}
            # Only detector-supplied trusted context, never user text.
            return policy.evaluate('input',context.get('bayesian_nemo_input_scores')).decision=='allow'
        self.rails.register_action(bayesian_action,name=name)


class IORailsAdapter(BayesianNeMoAdapter):
    def __init__(self, rails:Any, policy:BayesianPolicy):
        super().__init__(rails,policy)
        if _engine_name(rails)!='io':
            raise TypeError('IORailsAdapter requires an actual IORails engine (no silent fallback)')
