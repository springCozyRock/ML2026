"""Fit selected models on all regular training rows and save inference pipelines.

The rare Partial-sale expert uses both official training examples. All models,
preprocessing statistics and seeds are saved. This candidate has no independent
OOF score; do not mistake training RMSE for validation performance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from experiment import Features,fit_model,rmse
from transaction_branch import mask
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output'

def predict_saved(bundle,df):
    f=bundle['features'];kind=bundle['kind']
    z=f.native(df) if kind=='cat' else f.matrix(df)
    if kind in ['ridge','lasso','elasticnet','svr','krr']:z=bundle['scaler'].transform(z)
    p=np.asarray(bundle['model'].predict(z),dtype=np.float64)
    if kind=='residual_xgb':p+=bundle['base'].predict(bundle['scaler'].transform(z))
    return p

def main(seeds):
    original=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    train=original.loc[~((original.GrLivArea>4000)&(original.SalePrice<300000))].copy()
    y=np.log(train.SalePrice);specs=json.loads((ROOT/'frozen_config.json').read_text())
    fixed=json.loads((ROOT/'fixed_iterations.json').read_text())
    weights=json.loads((OUT/'optimized.json').read_text())['weights']
    total=np.zeros(len(test));records=[]
    for spec in specs:
        if weights[spec['name']]<0.001:continue
        for seed in seeds:
            dest=ROOT/'models'/spec['name']/('full_'+str(seed));dest.mkdir(parents=True,exist_ok=True)
            if (dest/'pipeline.joblib').exists():
                bundle=joblib.load(dest/'pipeline.joblib');p=predict_saved(bundle,train);q=predict_saved(bundle,test)
            else:
                prep=Features(spec['level'],spec['transform'],spec.get('rare',1)).fit(train)
                p,q,_=fit_model(spec,prep,train,y,train,y,test,seed,dest,fixed.get(spec['name']))
                bundle=joblib.load(dest/'pipeline.joblib')
                assert np.allclose(q,predict_saved(bundle,test),rtol=1e-8,atol=1e-8)
            total+=weights[spec['name']]*q/len(seeds)
            records.append({'name':spec['name'],'seed':seed,'train_rmse_only':rmse(y,p)})
            print('SAVED',spec['name'],seed,flush=True)
    rare=mask(original);rare_test=mask(test)
    total[rare_test]=np.log(original.loc[rare,'SalePrice']).mean()
    submission=pd.DataFrame({'Id':test.Id,'SalePrice':np.exp(total)})
    submission.to_csv(OUT/'submission_fullfit.csv',index=False)
    existing=pd.read_csv(OUT/'submission_transaction_branch.csv')
    merged=submission.copy();merged.SalePrice=np.exp((np.log(submission.SalePrice)+np.log(existing.SalePrice))/2)
    merged.to_csv(OUT/'submission_fullfit_bagged.csv',index=False)
    (OUT/'fullfit_training.json').write_text(json.dumps({'seeds':seeds,'records':records,'weights':weights,
        'rare_transaction_log_mean':float(np.log(original.loc[rare,'SalePrice']).mean()),
        'note':'Training errors are not validation scores. Fullfit + CV bag is a candidate requiring Kaggle evaluation.'},indent=2))
    assert len(merged)==1459 and np.isfinite(merged.SalePrice).all() and (merged.SalePrice>0).all()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',default='42,2026,73');a=p.parse_args();main([int(x) for x in a.seeds.split(',')])
