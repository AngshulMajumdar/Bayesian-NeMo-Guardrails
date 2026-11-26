"""Minimal reproducible CLI: offline calibration, then inference."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .calibration import calibrate
from .policy import BayesianPolicy

DEFAULT_CONFIG=Path(__file__).with_name('default_policy.yaml')


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(prog='bayesian-nemo',description='Bayesian inference layer for NeMo Guardrails')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('evaluate',help='Evaluate trusted detector scores')
    p.add_argument('--config',default=str(DEFAULT_CONFIG))
    p.add_argument('--phase',choices=['input','output'],required=True)
    p.add_argument('--scores',required=True,help='JSON mapping of detector outputs')
    c=sub.add_parser('calibrate',help='Fit a nonconjugate Bayesian logistic model offline')
    c.add_argument('--csv',required=True);c.add_argument('--output',required=True)
    c.add_argument('--features',nargs='+',required=True)
    c.add_argument('--label',default='unsafe')
    c.add_argument('--backend',choices=['laplace','corrected_gh'],default='laplace')
    c.add_argument('--quadrature-order',type=int,default=17)
    c.add_argument('--prior-diagonal',nargs='+',type=float)
    c.add_argument('--min-per-class',type=int,default=5)
    args=parser.parse_args(argv)
    try:
        if args.command=='calibrate':
            model=calibrate(args.csv,args.output,features=args.features,label=args.label,
                            backend=args.backend,quadrature_order=args.quadrature_order,
                            prior_diagonal=args.prior_diagonal,min_per_class=args.min_per_class)
            print(json.dumps({'saved':args.output,'n':model.calibration_count,
                              'positive_count':model.positive_count,'backend':model.backend,
                              'diagnostics':model.diagnostics},indent=2))
        else:
            policy=BayesianPolicy.from_yaml(args.config)
            verdict=policy.evaluate(args.phase,json.loads(args.scores))
            print(json.dumps(verdict.to_dict(),indent=2,allow_nan=False))
            return 0 if verdict.decision=='allow' else 2
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
