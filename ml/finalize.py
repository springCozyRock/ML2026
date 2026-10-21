"""Evaluate frozen weights on new splits; aggregate seed bags; audit all rows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'

def evaluate(folder,weights):
    oofs=[pd.read_csv(folder/name/'oof.csv') for name in weights]
    subs=[pd.read_csv(folder/name/'submission.csv') for name in weights]
    for o in oofs[1:]:assert np.array_equal(o.Id,oofs[0].Id)
    for s in subs[1:]:assert np.array_equal(s.Id,subs[0].Id)
    w=np.array(list(weights.values()));w=w/w.sum()
    result=oofs[0][['Id','actual_log','actual_price']].copy()
    result['prediction_log']=np.column_stack([o.prediction_log for o in oofs])@w
    result['prediction_price']=np.exp(result.prediction_log)
    result['residual_log']=result.actual_log-result.prediction_log
    test=pd.DataFrame({'Id':subs[0].Id,'SalePrice':np.exp(np.column_stack([np.log(s.SalePrice) for s in subs])@w)})
    metrics={'rmse':float(np.sqrt(np.mean(result.residual_log**2))),
             'mae_price':float(mean_absolute_error(result.actual_price,result.prediction_price)),
             'r2_log':float(r2_score(result.actual_log,result.prediction_log)),
             'n_rows':len(result),'weights':weights}
    return result,test,metrics

def calibration(pred,y,degree):
    center=12.0
    def design(x):return np.column_stack([(np.asarray(x)-center)**i for i in range(1,degree+1)])
    X=design(pred);cv=np.zeros(len(y))
    for ti,vi in KFold(5,shuffle=True,random_state=917).split(X):
        model=Ridge(alpha=2.0).fit(X[ti],np.asarray(y)[ti]-np.asarray(pred)[ti])
        cv[vi]=np.asarray(pred)[vi]+model.predict(X[vi])
    model=Ridge(alpha=2.0).fit(X,np.asarray(y)-np.asarray(pred))
    return model,lambda x:np.asarray(x)+model.predict(design(x)),float(np.sqrt(np.mean((np.asarray(y)-cv)**2)))

def main():
    meta=json.loads((OUT/'optimized.json').read_text())
    weights={k:v for k,v in meta['weights'].items() if v>0.001}
    a,ta,ma=evaluate(ROOT/'experiments/combined',weights)
    b,tb,mb=evaluate(ROOT/'experiments/confirmation',weights)
    c,tc,mc=evaluate(ROOT/'experiments/all_rows_audit',weights)
    a=a.sort_values('Id');b=b.sort_values('Id')
    assert np.array_equal(a.Id,b.Id)
    bag=a.copy();bag['prediction_log']=(a.prediction_log.to_numpy()+b.prediction_log.to_numpy())/2
    bag['prediction_price']=np.exp(bag.prediction_log);bag['residual_log']=bag.actual_log-bag.prediction_log
    for name,df in [('optimized',a),('confirmation',b),('all_rows',c),('multiseed',bag)]:
        df.to_csv(OUT/('oof_'+name+'.csv'),index=False)
        raw=pd.read_csv(ROOT/'data/train.csv');joined=df.merge(raw.drop(columns=['SalePrice']),on='Id')
        for col in ['Neighborhood','SaleCondition','OverallQual']:
            rows=[]
            for key,group in joined.groupby(col):
                rows.append({'group':key,'n':len(group),'rmse':float(np.sqrt(np.mean(group.residual_log**2))),
                             'mean_residual':float(group.residual_log.mean())})
            pd.DataFrame(rows).to_csv(OUT/(name+'_errors_by_'+col+'.csv'),index=False)
    for t in [tb,tc]:assert np.array_equal(ta.Id,t.Id)
    multi=ta.copy();multi.SalePrice=np.exp((np.log(ta.SalePrice)+np.log(tb.SalePrice)+np.log(tc.SalePrice))/3)
    multi.to_csv(OUT/'submission_multiseed.csv',index=False)
    # Calibration is assessed by additional cross-fitting, never against hidden test labels.
    candidates=[]
    for degree in [1,2,3]:
        model,apply,cv=calibration(bag.prediction_log.to_numpy(),bag.actual_log.to_numpy(),degree)
        candidates.append({'degree':degree,'crossfit_rmse':cv,'coefficients':model.coef_.tolist(),'intercept':float(model.intercept_)})
    best=min(candidates,key=lambda x:x['crossfit_rmse'])
    model,apply,cv=calibration(bag.prediction_log.to_numpy(),bag.actual_log.to_numpy(),best['degree'])
    calibrated=multi.copy();calibrated.SalePrice=np.exp(apply(np.log(multi.SalePrice)))
    calibrated.to_csv(OUT/'submission_calibrated.csv',index=False)
    count=10
    frac=float(np.sort(bag.residual_log.to_numpy()**2)[-count:].sum()/np.sum(bag.residual_log.to_numpy()**2))
    extra=bag.sort_values('residual_log',key=abs,ascending=False).head(20)
    extra.merge(pd.read_csv(ROOT/'data/train.csv'),on='Id').to_csv(OUT/'largest_errors.csv',index=False)
    summary={'selection':ma,'confirmation':mb,'all_rows_audit':mc,
             'multiseed_oof_rmse':float(np.sqrt(np.mean(bag.residual_log**2))),
             'calibration_trials':candidates,'best_calibration':best,'top10_squared_error_fraction':frac,
             'all_rows_anomalies':c.loc[c.Id.isin([524,1299])].to_dict(orient='records'),
             'note':'Multiple splits reuse the same labels; these are stability diagnostics, not independent test sets. All-row audit retains both anomalies for scoring, removes them only in each fold training set.'}
    (OUT/'final_validation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    for path in ['submission_multiseed.csv','submission_calibrated.csv']:
        s=pd.read_csv(OUT/path);sample=pd.read_csv(ROOT/'data/sample_submission.csv')
        assert len(s)==1459 and list(s)==['Id','SalePrice'] and s.Id.is_unique
        assert np.array_equal(s.Id,sample.Id) and (s.SalePrice>0).all() and np.isfinite(s.SalePrice).all()

if __name__=='__main__':main()
