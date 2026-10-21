"""Rare-transaction expert, using official training labels only.

The >4000 sq ft threshold is inherited from the pre-existing baseline. The
expert is triggered by feature values, not test IDs. Full-row OOF predictions
use only the other rows in the corresponding training fold for expert fitting.
Two supporting examples make this highly uncertain; see report limitations.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'

def mask(df,threshold=4000):
    return (df.GrLivArea>threshold)&(df.SaleCondition=='Partial')

def main():
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    oof=pd.read_csv(OUT/'oof_all_rows.csv').set_index('Id').loc[train.Id].reset_index()
    original=oof.prediction_log.to_numpy().copy();pred=original.copy();records=[]
    for f,(ti,vi) in enumerate(KFold(5,shuffle=True,random_state=73).split(train)):
        expert=train.iloc[ti].loc[mask(train.iloc[ti])]
        valid=train.iloc[vi].loc[mask(train.iloc[vi])]
        if len(expert) and len(valid):
            value=float(np.log(expert.SalePrice).mean())
            affected=np.isin(train.iloc[vi].Id,valid.Id)
            pred[vi[affected]]=value
            records.append({'fold':f+1,'expert_training_ids':expert.Id.tolist(),'expert_validation_ids':valid.Id.tolist(),
                            'expert_prediction':float(np.exp(value))})
            assert set(expert.Id).isdisjoint(set(valid.Id))
    y=np.log(train.SalePrice.to_numpy())
    q=oof.copy();q['prediction_log']=pred;q['prediction_price']=np.exp(pred);q['residual_log']=y-pred
    q.to_csv(OUT/'oof_transaction_branch.csv',index=False)
    expert=train.loc[mask(train)];value=float(np.log(expert.SalePrice).mean())
    base=pd.read_csv(OUT/'submission_multiseed.csv')
    assert np.array_equal(base.Id,test.Id)
    affected=mask(test)
    base.loc[affected,'SalePrice']=np.exp(value)
    base.to_csv(OUT/'submission_transaction_branch.csv',index=False)
    sensitivity=[]
    for threshold in [3800,4000,4200,4500]:
        ex=train.loc[mask(train,threshold)];va=test.loc[mask(test,threshold)]
        sensitivity.append({'threshold':threshold,'train_n':len(ex),'train_ids':ex.Id.tolist(),'test_n':len(va),'test_ids':va.Id.tolist(),
                            'estimated_price':float(np.exp(np.log(ex.SalePrice).mean())) if len(ex) else None})
    result={'condition':'GrLivArea > 4000 and SaleCondition == Partial',
        'official_training_n':len(expert),'training_ids':expert.Id.tolist(),'affected_test_n':int(affected.sum()),
        'affected_test_ids':test.loc[affected,'Id'].tolist(),'fitted_price':float(np.exp(value)),
        'all_rows_original_rmse':float(np.sqrt(np.mean((y-original)**2))),
        'all_rows_branch_rmse':float(np.sqrt(np.mean((y-pred)**2))),
        'branch_leave_one_out':records,'threshold_sensitivity':sensitivity,
        'warning':'Only two rare transactions support this branch. This is a hypothesis selected after EDA, not independent confirmation or a reliable general business rule. No hidden test labels or external labels used.'}
    (OUT/'transaction_branch.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert len(base)==1459 and base.Id.is_unique and (base.SalePrice>0).all() and np.isfinite(base.SalePrice).all()

if __name__=='__main__':main()
