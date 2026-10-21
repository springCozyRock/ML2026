"""Generate EDA and auditable ensemble diagnostics from actual saved predictions."""
import os
os.environ['MPLBACKEND']='Agg'
os.environ['OMP_NUM_THREADS']='4'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'
FIG=OUT/'figures'
OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})

def save(name):
    plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close()

def eda():
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    missing=pd.DataFrame({'train_count':train.isna().sum(),'test_count':test.isna().sum()}).fillna(0).astype(int)
    missing['train_percent']=missing.train_count/len(train)*100
    missing['test_percent']=missing.test_count/len(test)*100
    missing.sort_values('train_count',ascending=False).to_csv(OUT/'missingness.csv')
    corr=train.select_dtypes(include=np.number).corr().SalePrice.sort_values(ascending=False)
    corr.to_csv(OUT/'correlations.csv',header=['pearson'])
    anomalies=train.loc[(train.GrLivArea>4000)&(train.SalePrice<300000)]
    anomalies.to_csv(OUT/'excluded_rows.csv',index=False)
    stats={'train_shape':list(train.shape),'test_shape':list(test.shape),'numeric_features':len(train.drop(columns=['Id','SalePrice']).select_dtypes(include=np.number).columns),
           'categorical_features':len(train.select_dtypes(exclude=np.number).columns),'sale_price':train.SalePrice.describe().to_dict(),
           'raw_skew':float(train.SalePrice.skew()),'log_skew':float(np.log(train.SalePrice).skew()),
           'excluded_ids':anomalies.Id.tolist(),'duplicate_train_rows':int(train.drop(columns='Id').duplicated().sum())}
    (OUT/'data_stats.json').write_text(json.dumps(stats,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(10,3.4))
    ax[0].hist(train.SalePrice/1000,bins=40,color='#4070a0');ax[0].set(xlabel='SalePrice (thousand USD)',ylabel='Count',title='Original target')
    ax[1].hist(np.log(train.SalePrice),bins=40,color='#328579');ax[1].set(xlabel='ln(SalePrice)',ylabel='Count',title='Logarithmic target')
    save('target_distribution')
    top=missing.sort_values('train_count',ascending=False).head(12).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,4));yy=np.arange(len(top))
    ax.barh(yy-0.18,top.train_percent,height=.36,label='Train',color='#4070a0')
    ax.barh(yy+.18,top.test_percent,height=.36,label='Test',color='#83b8ae')
    ax.set(yticks=yy,yticklabels=top.index,xlabel='Missing values (%)');ax.legend()
    save('missingness')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    normal=~train.Id.isin(anomalies.Id)
    ax[0].scatter(train.loc[normal,'GrLivArea'],train.loc[normal,'SalePrice']/1000,s=10,alpha=.45,color='#4070a0')
    ax[0].scatter(anomalies.GrLivArea,anomalies.SalePrice/1000,c='#bf594c',s=45,label='Excluded anomalies')
    for _,r in anomalies.iterrows():ax[0].annotate('Id '+str(int(r.Id)),(r.GrLivArea,r.SalePrice/1000),fontsize=8)
    ax[0].set(xlabel='GrLivArea (sq ft)',ylabel='Price (thousand USD)');ax[0].legend(fontsize=8)
    data=[np.log(train.loc[train.OverallQual==q,'SalePrice']) for q in sorted(train.OverallQual.unique())]
    ax[1].boxplot(data,labels=sorted(train.OverallQual.unique()),showfliers=False)
    ax[1].set(xlabel='OverallQual',ylabel='ln(SalePrice)')
    save('area_quality')
    cols=corr.drop(['SalePrice','Id'],errors='ignore').head(10).index
    mat=train[list(cols)+['SalePrice']].corr()
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set(xticks=np.arange(len(mat)),yticks=np.arange(len(mat)),xticklabels=mat.columns,yticklabels=mat.columns)
    plt.setp(ax.get_xticklabels(),rotation=55,ha='right');fig.colorbar(im,ax=ax,shrink=.75)
    save('correlation_heatmap')
    for c in ['Neighborhood','OverallQual','YrSold','SaleCondition']:
        group=train.groupby(c).SalePrice.agg(['count','median','mean','std'])
        group.to_csv(OUT/('group_'+c+'.csv'))

def blend(run='screen',names=None,label='ensemble'):
    folder=ROOT/'experiments'/run
    specs=[]
    for f in folder.glob('*/metrics.json'):
        m=json.loads(f.read_text());specs.append(m)
    summary=pd.DataFrame([{k:v for k,v in m.items() if k not in ['spec','folds']} for m in specs]).sort_values('rmse')
    summary.to_csv(folder/'summary.csv',index=False)
    print(summary[['name','rmse','fold_std']].to_string(index=False))
    if not names:
        # One representative of each family avoids overweighting highly correlated variants.
        chosen=[]
        for kind in ['lasso','svr','residual_xgb','xgb','cat','gbr','ridge','lgb']:
            family=[m for m in specs if m['spec']['kind']==kind and m['spec']['level']!='raw']
            if family:chosen.append(min(family,key=lambda m:m['rmse'])['name'])
        names=chosen
    oofs=[pd.read_csv(folder/n/'oof.csv') for n in names]
    subs=[pd.read_csv(folder/n/'submission.csv') for n in names]
    for o in oofs[1:]:assert np.array_equal(o.Id,oofs[0].Id)
    for s in subs[1:]:assert np.array_equal(s.Id,subs[0].Id)
    y=oofs[0].actual_log.to_numpy();P=np.column_stack([o.prediction_log for o in oofs])
    T=np.column_stack([np.log(s.SalePrice) for s in subs])
    equal=np.ones(len(names))/len(names)
    def weights(p,y):
        # Convex weights, gentle shrinkage towards equal weights; no price postprocessing.
        result=minimize(lambda w:np.mean((y-p@w)**2)+0.00005*np.sum((w-equal)**2),equal,
                        bounds=[(0,0.55)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                        method='SLSQP',options={'ftol':1e-12,'maxiter':500})
        if not result.success:raise RuntimeError(result.message)
        return result.x
    w=weights(P,y);pred=P@w;score=float(np.sqrt(np.mean((y-pred)**2)))
    cf=np.zeros(len(y));fold_weights=[]
    for ti,vi in KFold(5,shuffle=True,random_state=817).split(P):
        wi=weights(P[ti],y[ti]);cf[vi]=P[vi]@wi;fold_weights.append(wi.tolist())
    cf_score=float(np.sqrt(np.mean((y-cf)**2)))
    meta={'models':names,'weights':dict(zip(names,w.tolist())),'oof_rmse':score,
          'meta_crossfit_rmse':cf_score,'meta_crossfit_weights':fold_weights,
          'note':'Base-model hyperparameters selected on these CV rows. Meta crossfit is a diagnostic, not unbiased nested validation.',
          'run':run,'label':label}
    (OUT/(label+'.json')).write_text(json.dumps(meta,indent=2))
    s=pd.DataFrame({'Id':subs[0].Id,'SalePrice':np.exp(T@w)})
    assert len(s)==1459 and s.Id.is_unique and np.isfinite(s.SalePrice).all() and (s.SalePrice>0).all()
    s.to_csv(OUT/('submission_'+label+'.csv'),index=False)
    o=oofs[0][['Id','actual_log','actual_price']].copy();o['prediction_log']=pred;o['prediction_price']=np.exp(pred)
    o['residual_log']=y-pred;o.to_csv(OUT/('oof_'+label+'.csv'),index=False)
    o.sort_values('residual_log',key=abs,ascending=False).head(30).merge(pd.read_csv(ROOT/'data/train.csv'),on='Id').to_csv(OUT/('errors_'+label+'.csv'),index=False)
    residual=P-y[:,None];pd.DataFrame(np.corrcoef(residual.T),index=names,columns=names).to_csv(OUT/(label+'_error_correlation.csv'))
    print('BLEND',json.dumps(meta),flush=True)
    fig,ax=plt.subplots(figsize=(8,4));s=summary.head(18).iloc[::-1]
    ax.barh(s.name,s.rmse,color='#4070a0');ax.axvline(score,c='#b45247',ls='--',label='Ensemble OOF')
    ax.set(xlabel='RMSE of natural log price');ax.legend();save('model_comparison')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    ax[0].scatter(y,pred,s=10,alpha=.5,color='#4070a0');lim=[min(y.min(),pred.min()),max(y.max(),pred.max())];ax[0].plot(lim,lim,'--',color='#b45247')
    ax[0].set(xlabel='Observed ln(price)',ylabel='OOF predicted ln(price)')
    ax[1].scatter(pred,y-pred,s=10,alpha=.5,color='#328579');ax[1].axhline(0,c='#b45247',ls='--');ax[1].set(xlabel='OOF predicted ln(price)',ylabel='Observed - predicted')
    save('residual_diagnostics')
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='screen');p.add_argument('--names',default='');p.add_argument('--label',default='ensemble');p.add_argument('--eda-only',action='store_true')
    a=p.parse_args();eda()
    if not a.eda_only:blend(a.run,a.names.split(',') if a.names else None,a.label)
