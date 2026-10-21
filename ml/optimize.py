"""Optuna GPU search; selection CV is explicitly separated from robustness CV."""
import os
os.environ['OMP_NUM_THREADS']='8'
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from experiment import Features,fit_model,rmse

ROOT=Path(__file__).resolve().parent

def main(trials):
    out=ROOT/'experiments/optuna';out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'data/train.csv');train=train.loc[~((train.GrLivArea>4000)&(train.SalePrice<300000))].reset_index(drop=True)
    test=pd.read_csv(ROOT/'data/test.csv');y=np.log(train.SalePrice.to_numpy())
    folds=list(KFold(3,shuffle=True,random_state=631).split(train))
    prepared={}
    for level in ['raw','engineered','extended']:
        for transform in ['none','boxcox']:
            prepared[(level,transform)]=[Features(level,transform,2).fit(train.iloc[ti]) for ti,vi in folds]
    study=optuna.create_study(direction='minimize',study_name='gpu_residual_search',
        storage='sqlite:///'+str(out/'study.sqlite3'),load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=1042),pruner=optuna.pruners.MedianPruner(n_startup_trials=12))
    def objective(trial):
        level=trial.suggest_categorical('level',['raw','engineered','extended'])
        kind=trial.suggest_categorical('kind',['xgb','residual_xgb'])
        transform=trial.suggest_categorical('transform',['none','boxcox']) if kind=='residual_xgb' else 'none'
        params=dict(n_estimators=3500,learning_rate=trial.suggest_float('learning_rate',0.012,0.06,log=True),
            max_depth=trial.suggest_int('max_depth',2,4),min_child_weight=trial.suggest_float('min_child_weight',1,16,log=True),
            subsample=trial.suggest_float('subsample',0.65,1),colsample_bytree=trial.suggest_float('colsample_bytree',0.45,1),
            reg_alpha=trial.suggest_float('reg_alpha',0.001,0.8,log=True),reg_lambda=trial.suggest_float('reg_lambda',0.2,20,log=True),device='cuda:5')
        if kind=='residual_xgb':params['base_alpha']=trial.suggest_float('base_alpha',0.0002,0.001,log=True)
        spec=dict(name='optuna_'+str(trial.number),kind=kind,params=params,level=level,transform=transform,scale='robust',rare=2)
        trial.set_user_attr('spec',spec);scores=[];start=time.time()
        for f,(ti,vi) in enumerate(folds):
            dest=out/('trial_'+str(trial.number))/('fold_'+str(f+1));dest.mkdir(parents=True,exist_ok=True)
            p,q,best=fit_model(spec,prepared[(level,transform)][f],train.iloc[ti],y[ti],train.iloc[vi],y[vi],test,1042+f,dest)
            scores.append(rmse(y[vi],p));trial.report(np.mean(scores),f)
            trial.set_user_attr('fold_scores',scores);trial.set_user_attr('seconds',time.time()-start)
            if trial.should_prune():raise optuna.TrialPruned()
        return float(np.mean(scores))
    study.optimize(objective,n_trials=trials)
    study.trials_dataframe().to_csv(out/'trials.csv',index=False)
    complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    best=sorted(complete,key=lambda t:t.value)[:4]
    (ROOT/'optuna_best_config.json').write_text(json.dumps([t.user_attrs['spec'] for t in best],indent=2))
    print('OPTUNA BEST',[(t.number,t.value,t.user_attrs['spec']) for t in best],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=40);a=p.parse_args();main(a.trials)
