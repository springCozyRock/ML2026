"""Create final report charts from stored metrics, never from invented scores."""
import os
os.environ['MPLBACKEND']='Agg'
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output';FIG=OUT/'figures'
plt.rcParams.update({'font.size':10,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
def save(name):
    plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close()
def main():
    records=json.loads((OUT/'kaggle_submissions.json').read_text())
    complete=sorted([r for r in records if r['status']=='complete' and r['publicScore']],key=lambda r:r['date'])
    fig,ax=plt.subplots(figsize=(8,3.5));scores=[float(r['publicScore']) for r in complete]
    ax.plot(np.arange(len(scores)),scores,'o-',color='#4070a0',lw=2)
    ax.axhline(.1,color='#b45247',ls='--',label='Target: < 0.1')
    for i,v in enumerate(scores):ax.annotate(f'{v:.5f}',(i,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    ax.set(xticks=np.arange(len(scores)),xticklabels=['v'+str(i+1) for i in range(len(scores))],xlabel='Submission iteration',ylabel='Kaggle public RMSE (log price)')
    ax.legend();save('kaggle_iterations')
    m=json.loads((OUT/'final_validation.json').read_text());br=json.loads((OUT/'transaction_branch.json').read_text())
    labels=['Selection / 1458','New seed / 1458','All rows / 1460','Transaction branch / 1460']
    values=[m['selection']['rmse'],m['confirmation']['rmse'],m['all_rows_audit']['rmse'],br['all_rows_branch_rmse']]
    fig,ax=plt.subplots(figsize=(8,3.2));bars=ax.barh(labels[::-1],values[::-1],color=['#328579','#b45247','#4070a0','#83b8ae'])
    for b,v in zip(bars,values[::-1]):ax.text(v+.0004,b.get_y()+b.get_height()/2,f'{v:.5f}',va='center')
    ax.set(xlim=(.09,.133),xlabel='OOF RMSE (log price)');save('validation_audit')
    o=pd.read_csv(OUT/'oof_transaction_branch.csv');y=o.actual_log;p=o.prediction_log
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    ax[0].scatter(y,p,s=10,alpha=.45,color='#4070a0');lo=min(y.min(),p.min());hi=max(y.max(),p.max());ax[0].plot([lo,hi],[lo,hi],'--',c='#b45247')
    ax[0].set(xlabel='Observed ln(price)',ylabel='Full-row OOF predicted ln(price)')
    ax[1].scatter(p,y-p,s=10,alpha=.45,color='#328579');ax[1].axhline(0,c='#b45247',ls='--');ax[1].set(xlabel='Full-row OOF predicted ln(price)',ylabel='Observed - predicted')
    save('final_residuals')
    series=[]
    for f in (ROOT/'experiments/screen/cat_cpu_d4').glob('fold_*/importance.csv'):
        series.append(pd.read_csv(f,index_col=0).importance)
    if series:
        imp=pd.concat(series,axis=1).mean(axis=1).sort_values(ascending=False)
        imp.to_csv(OUT/'catboost_mean_importance.csv',header=['importance'])
        fig,ax=plt.subplots(figsize=(8,4));s=imp.head(15).iloc[::-1];ax.barh(s.index,s.values,color='#4070a0');ax.set(xlabel='Mean CatBoost feature importance (5 folds)');save('feature_importance')
    summary=[]
    for run in ['screen','tune','optuna_final']:
        for f in (ROOT/'experiments'/run).glob('*/metrics.json'):
            r=json.loads(f.read_text());r['run']=run
            summary.append({k:v for k,v in r.items() if k not in ['folds','spec']})
    pd.DataFrame(summary).sort_values('rmse').to_csv(OUT/'all_experiments.csv',index=False)
    (OUT/'experiment_counts.json').write_text(json.dumps({'five_fold_configurations':len(summary),
        'optuna_trials':len(pd.read_csv(ROOT/'experiments/optuna/trials.csv')),
        'five_fold_model_fits':5*len(summary),'additional_stability_fits':40},indent=2))
if __name__=='__main__':main()
