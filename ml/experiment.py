"""Reproducible House Prices experiments. Only official train labels are used.

Preprocessing is fitted separately inside every training fold. Checkpoints include
OOF predictions, held-out IDs, per-fold parameters, runtime, and data hashes.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '8'
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
warnings.filterwarnings('ignore', category=UserWarning)

QUAL = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ORDINAL = {
    **{c: QUAL for c in ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
                        'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']},
    'BsmtExposure': {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4},
    'BsmtFinType1': {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6},
    'BsmtFinType2': {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6},
    'GarageFinish': {'None':0,'Unf':1,'RFn':2,'Fin':3},
    'Functional': {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,'Mod':5,'Min2':6,'Min1':7,'Typ':8},
    'Fence': {'None':0,'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4},
    'LotShape': {'IR3':1,'IR2':2,'IR1':3,'Reg':4},
    'LandSlope': {'Sev':1,'Mod':2,'Gtl':3}, 'PavedDrive': {'N':0,'P':1,'Y':2},
    'CentralAir': {'N':0,'Y':1},
}
NONE = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC',
        'Fence','MiscFeature','MasVnrType']
ZERO = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
        'BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']

def rmse(y,p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

class Features:
    def __init__(self, level='engineered', transform='boxcox', rare=1):
        self.level, self.transform, self.rare = level, transform, rare

    def base(self, frame):
        z=frame.drop(columns=['Id','SalePrice'], errors='ignore').copy()
        for c in NONE: z[c]=z[c].fillna('None')
        for c in ZERO: z[c]=z[c].fillna(0)
        z['GarageYrBlt']=z['GarageYrBlt'].fillna(z['YearBuilt'])
        z.loc[z.GarageYrBlt>z.YrSold+1,'GarageYrBlt']=z.loc[z.GarageYrBlt>z.YrSold+1,'YearBuilt']
        z['Functional']=z['Functional'].fillna('Typ')
        z['LotFrontage']=z.LotFrontage.fillna(z.Neighborhood.map(self.frontage)).fillna(self.global_frontage)
        for c in ['MSSubClass','MoSold']: z[c]=z[c].astype(str)
        if self.level=='raw': return z
        for c,m in ORDINAL.items():
            z[c+'Ordinal']=z[c].map(m)
        z['TotalSF']=z.TotalBsmtSF+z['1stFlrSF']+z['2ndFlrSF']
        z['FinishedSF']=z.GrLivArea+z.BsmtFinSF1+z.BsmtFinSF2
        z['EffectiveSF']=z['1stFlrSF']+0.8*z['2ndFlrSF']+0.5*(z.BsmtFinSF1+z.BsmtFinSF2)+0.3*z.GarageArea
        z['TotalBath']=z.FullBath+0.5*z.HalfBath+z.BsmtFullBath+0.5*z.BsmtHalfBath
        z['TotalPorchSF']=z.OpenPorchSF+z.EnclosedPorch+z['3SsnPorch']+z.ScreenPorch+z.WoodDeckSF
        z['HouseAge']=(z.YrSold-z.YearBuilt).clip(lower=0)
        z['RemodAge']=(z.YrSold-z.YearRemodAdd).clip(lower=0)
        z['GarageAge']=(z.YrSold-z.GarageYrBlt).clip(lower=0)
        z['Remodeled']=(z.YearRemodAdd>z.YearBuilt).astype(int)
        z['IsNew']=(z.YearBuilt>=z.YrSold-1).astype(int)
        for out,c in [('HasGarage','GarageArea'),('HasBsmt','TotalBsmtSF'),('HasPool','PoolArea'),
                      ('HasFireplace','Fireplaces'),('Has2ndFloor','2ndFlrSF'),('HasPorch','TotalPorchSF')]:
            z[out]=(z[c]>0).astype(int)
        z['QualArea']=z.OverallQual*z.GrLivArea
        z['QualTotalSF']=z.OverallQual*z.TotalSF
        z['QualitySquared']=z.OverallQual**2
        z['QualityCondition']=z.OverallQual*z.OverallCond
        z['BsmtQualitySF']=z.BsmtQualOrdinal*z.TotalBsmtSF
        z['GarageQualityArea']=z.GarageQualOrdinal*z.GarageArea
        z['BathPerRoom']=z.TotalBath/(z.TotRmsAbvGrd+1)
        z['AreaPerRoom']=z.GrLivArea/(z.TotRmsAbvGrd+1)
        z['LotUtilization']=z.GrLivArea/(z.LotArea+1)
        z['LivingLotRatio']=z['1stFlrSF']/(z.LotArea+1)
        z['BsmtFinishedRatio']=(z.BsmtFinSF1+z.BsmtFinSF2)/(z.TotalBsmtSF+1)
        z['TotalQuality']=z.OverallQual+z.ExterQualOrdinal+z.KitchenQualOrdinal+z.BsmtQualOrdinal
        if self.level=='extended':
            for c in ['OverallQual','OverallCond','ExterQualOrdinal','KitchenQualOrdinal']:
                z[c+'AreaLog']=z[c]*np.log1p(z.GrLivArea)
            z['NeighborhoodStyle']=z.Neighborhood.astype(str)+'_'+z.HouseStyle.astype(str)
            z['NeighborhoodQual']=z.Neighborhood.astype(str)+'_'+(z.OverallQual//2).astype(str)
            z['QualityAge']=z.OverallQual*np.log1p(z.HouseAge)
            z['QualityRemodAge']=z.OverallQual*np.log1p(z.RemodAge)
        return z

    def fit(self, frame):
        self.frontage=frame.groupby('Neighborhood').LotFrontage.median()
        self.global_frontage=frame.LotFrontage.median()
        z=self.base(frame)
        self.num=z.select_dtypes(include=np.number).columns.tolist()
        self.cat=[c for c in z.columns if c not in self.num]
        self.median=z[self.num].median().fillna(0)
        self.mode={c: (z[c].mode().iloc[0] if len(z[c].mode()) else 'Unknown') for c in self.cat}
        self.skewed=[c for c in self.num if z[c].min()>=0 and z[c].nunique()>2 and abs(z[c].skew())>0.75]
        self.enc=OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                               min_frequency=self.rare if self.rare>1 else None)
        self.enc.fit(self.clean_cat(z))
        self.feature_names=self.num+list(self.enc.get_feature_names_out(self.cat))
        return self

    def clean_cat(self,z):
        return z[self.cat].fillna(self.mode).astype(str)

    def numeric(self,z):
        z=z[self.num].fillna(self.median).astype(float).copy()
        if self.transform!='none':
            for c in self.skewed:
                v=z[c].clip(lower=0)
                z[c]=np.log1p(v) if self.transform=='log' else boxcox1p(v,0.15)
        return z.to_numpy(dtype=np.float64)

    def matrix(self,frame):
        z=self.base(frame)
        return np.column_stack([self.numeric(z),self.enc.transform(self.clean_cat(z))])

    def native(self,frame):
        z=self.base(frame)
        z[self.num]=z[self.num].fillna(self.median)
        z[self.cat]=self.clean_cat(z)
        return z

def configurations():
    specs=[]
    def add(name,kind,params,level='engineered',transform='boxcox',scale='robust',rare=1):
        specs.append(dict(name=name,kind=kind,params=params,level=level,transform=transform,scale=scale,rare=rare))
    add('raw_ridge','ridge',{'alpha':20},'raw')
    add('raw_xgb','xgb',{'max_depth':3,'learning_rate':0.03,'n_estimators':4000,'min_child_weight':2,'subsample':0.8,'colsample_bytree':0.6,'reg_lambda':1,'reg_alpha':0.1},'raw',transform='none')
    for a in [5,15,35]: add('ridge_'+str(a),'ridge',{'alpha':a})
    for a in [0.0003,0.0005,0.0008]: add('lasso_'+str(a),'lasso',{'alpha':a,'max_iter':15000})
    add('elasticnet','elasticnet',{'alpha':0.0005,'l1_ratio':0.85,'max_iter':15000})
    for c,g in [(20,0.0003),(30,0.0005),(50,0.0002)]:
        add('svr_'+str(c),'svr',{'C':c,'epsilon':0.008,'gamma':g})
    add('kernel_ridge','krr',{'alpha':0.6,'kernel':'polynomial','degree':2,'coef0':2.5})
    add('random_forest','rf',{'n_estimators':600,'max_features':0.7,'min_samples_leaf':2,'n_jobs':8},transform='none')
    for d in [2,3,4]:
        add('xgb_d'+str(d),'xgb',{'n_estimators':5000,'learning_rate':0.02,'max_depth':d,'min_child_weight':2,'subsample':0.8,'colsample_bytree':0.7,'reg_alpha':0.05,'reg_lambda':2},transform='none')
    for leaves in [5,8,12,20]:
        add('lgb_'+str(leaves),'lgb',{'n_estimators':5000,'learning_rate':0.015,'num_leaves':leaves,'max_depth':-1,'min_child_samples':12,'subsample':0.8,'subsample_freq':1,'colsample_bytree':0.7,'reg_alpha':0.05,'reg_lambda':0.2},transform='none')
    add('gbr_huber','gbr',{'n_estimators':3000,'learning_rate':0.025,'max_depth':3,'min_samples_leaf':8,'max_features':0.7,'subsample':0.8,'loss':'huber'})
    for d in [4,5,6]:
        add('cat_cpu_d'+str(d),'cat',{'iterations':5000,'learning_rate':0.025,'depth':d,'l2_leaf_reg':3,'boosting_type':'Ordered','task_type':'CPU'},transform='none')
    add('cat_gpu_d5','cat',{'iterations':5000,'learning_rate':0.025,'depth':5,'l2_leaf_reg':5,'task_type':'GPU','devices':'4'},transform='none')
    return specs

def fit_model(spec,prep,tr,ytr,va,yva,te,seed,fold_dir,fixed_iterations=None):
    kind=spec['kind']; params=spec['params'].copy()
    base_alpha=params.pop('base_alpha',0.0005)
    device=params.pop('device','cuda:4')
    base_va=base_te=0
    if fixed_iterations:
        if kind in ['xgb','residual_xgb','lgb']: params['n_estimators']=int(fixed_iterations)
        if kind=='cat': params['iterations']=int(fixed_iterations)
    if kind=='cat':
        a,b,c=[prep.native(z) for z in [tr,va,te]]
        model=CatBoostRegressor(**params,random_seed=seed,thread_count=8,loss_function='RMSE',verbose=False,allow_writing_files=False)
        kwargs={} if fixed_iterations else {'eval_set':(b,yva),'early_stopping_rounds':250,'use_best_model':True}
        model.fit(a,ytr,cat_features=prep.cat,**kwargs)
        imp=pd.Series(model.get_feature_importance(),index=a.columns)
        best=model.tree_count_
    else:
        a,b,c=[prep.matrix(z) for z in [tr,va,te]]
        if kind in ['ridge','lasso','elasticnet','svr','krr']:
            scaler=RobustScaler() if spec['scale']=='robust' else StandardScaler()
            a=scaler.fit_transform(a); b=scaler.transform(b); c=scaler.transform(c)
        if kind=='ridge': model=Ridge(**params)
        elif kind=='lasso': model=Lasso(**params,random_state=seed)
        elif kind=='elasticnet': model=ElasticNet(**params,random_state=seed)
        elif kind=='svr': model=SVR(**params)
        elif kind=='krr': model=KernelRidge(**params)
        elif kind=='rf': model=RandomForestRegressor(**params,random_state=seed)
        elif kind=='gbr': model=GradientBoostingRegressor(**params,random_state=seed)
        elif kind in ['xgb','residual_xgb']:
            if kind=='residual_xgb':
                scaler=RobustScaler();sa=scaler.fit_transform(a)
                base=Lasso(alpha=base_alpha,max_iter=15000,random_state=seed).fit(sa,ytr)
                base_tr=base.predict(sa);base_va=base.predict(scaler.transform(b));base_te=base.predict(scaler.transform(c))
                ytr=ytr-base_tr;yva=yva-base_va
            model=xgb.XGBRegressor(**params,random_state=seed,n_jobs=8,tree_method='hist',device=device,objective='reg:squarederror',**({} if fixed_iterations else {'early_stopping_rounds':200}))
        elif kind=='lgb': model=lgb.LGBMRegressor(**params,random_state=seed,n_jobs=8,verbosity=-1)
        if kind in ['xgb','residual_xgb']:
            model.fit(a,ytr,**({} if fixed_iterations else {'eval_set':[(b,yva)],'verbose':False}))
        elif kind=='lgb':
            model.fit(a,ytr,**({} if fixed_iterations else {'eval_set':[(b,yva)],'callbacks':[lgb.early_stopping(200,verbose=False)]}))
        else: model.fit(a,ytr)
        best=getattr(model,'best_iteration_',None)
        if kind in ['xgb','residual_xgb']: best=int(model.best_iteration)+1 if not fixed_iterations else fixed_iterations
        if hasattr(model,'feature_importances_'): imp=pd.Series(model.feature_importances_,index=prep.feature_names)
        elif hasattr(model,'coef_'): imp=pd.Series(np.abs(model.coef_),index=prep.feature_names)
        else: imp=None
    if imp is not None:
        imp.sort_values(ascending=False).to_csv(fold_dir/'importance.csv',header=['importance'])
    if fold_dir.name.startswith('full_'):
        import joblib
        joblib.dump({'model':model,'features':prep,'kind':kind,
                     'scaler':locals().get('scaler'),'base':locals().get('base')},fold_dir/'pipeline.joblib',compress=3)
    return model.predict(b)+base_va,model.predict(c)+base_te,best

def run(specs,seed=42,run_name='screen',fixed=None,include_all=False):
    full=pd.read_csv(ROOT/'data/train.csv'); test=pd.read_csv(ROOT/'data/test.csv')
    exclusion=(full.GrLivArea>4000)&(full.SalePrice<300000)
    train=full.copy() if include_all else full.loc[~exclusion].copy()
    train=train.reset_index(drop=True); y=np.log(train.SalePrice.to_numpy())
    out=ROOT/'experiments'/run_name; out.mkdir(parents=True,exist_ok=True)
    meta={'seed':seed,'n_splits':5,'metric':'RMSE(log(y), log(prediction))','rows':len(train),
          'excluded_ids':[] if include_all else full.loc[exclusion,'Id'].tolist(),
          'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.csv')},
          'fold_preprocessing':True,'specs':specs}
    (out/'run_config.json').write_text(json.dumps(meta,indent=2))
    folds=list(KFold(5,shuffle=True,random_state=seed).split(train))
    for spec in specs:
        dest=out/spec['name']; dest.mkdir(exist_ok=True)
        if (dest/'metrics.json').exists():
            print('SKIP completed',spec['name'],flush=True);continue
        start=time.time(); oof=np.zeros(len(train)); pred=np.zeros(len(test)); records=[]
        try:
            for f,(ti,vi) in enumerate(folds):
                fold_dir=dest/('fold_'+str(f+1));fold_dir.mkdir(exist_ok=True)
                checkpoint=fold_dir/'predictions.npz'
                if checkpoint.exists():
                    z=np.load(checkpoint);p,q=z['valid'],z['test']; record=json.loads((fold_dir/'metrics.json').read_text())
                else:
                    t=time.time(); tr,va=train.iloc[ti].copy(),train.iloc[vi].copy()
                    # When evaluating all 1460 rows, anomalous rows are excluded only from fold training.
                    if include_all:
                        keep=~((tr.GrLivArea>4000)&(tr.SalePrice<300000)); tr=tr.loc[keep];ytr=y[ti][keep]
                    else:ytr=y[ti]
                    prep=Features(spec['level'],spec['transform'],spec.get('rare',1)).fit(tr)
                    p,q,best=fit_model(spec,prep,tr,ytr,va,y[vi],test,seed+f,fold_dir,(fixed or {}).get(spec['name']))
                    record={'fold':f+1,'rmse':rmse(y[vi],p),'seconds':time.time()-t,'iterations':best,
                            'train_rows':len(tr),'valid_rows':len(vi),'encoded_features':len(prep.feature_names)}
                    np.savez_compressed(checkpoint,valid=p,test=q,valid_ids=va.Id.to_numpy(),train_ids=tr.Id.to_numpy())
                    (fold_dir/'metrics.json').write_text(json.dumps(record,indent=2))
                oof[vi]=p;pred+=q/5;records.append(record)
                print(spec['name'],'fold',f+1,'rmse',round(record['rmse'],6),'seconds',round(record['seconds'],1),'iterations',record['iterations'],flush=True)
            metrics={'name':spec['name'],'rmse':rmse(y,oof),'fold_mean':float(np.mean([r['rmse'] for r in records])),
                     'fold_std':float(np.std([r['rmse'] for r in records])), 'seconds':time.time()-start,
                     'mae_price':float(mean_absolute_error(train.SalePrice,np.exp(oof))),
                     'r2_log':float(r2_score(y,oof)),'folds':records,'spec':spec}
            pd.DataFrame({'Id':train.Id,'actual_log':y,'prediction_log':oof,'actual_price':train.SalePrice,
                          'prediction_price':np.exp(oof)}).to_csv(dest/'oof.csv',index=False)
            pd.DataFrame({'Id':test.Id,'SalePrice':np.exp(pred)}).to_csv(dest/'submission.csv',index=False)
            (dest/'metrics.json').write_text(json.dumps(metrics,indent=2))
            print('RESULT',spec['name'],round(metrics['rmse'],6),flush=True)
        except Exception as exc:
            import traceback;traceback.print_exc()
            (dest/'error.txt').write_text(traceback.format_exc())
    results=[json.loads(p.read_text()) for p in out.glob('*/metrics.json')]
    pd.DataFrame([{k:v for k,v in m.items() if k not in ['folds','spec']} for m in results]).sort_values('rmse').to_csv(out/'summary.csv',index=False)
    print('RUN COMPLETE',run_name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='screen');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--names',default='');p.add_argument('--config');p.add_argument('--fixed');p.add_argument('--all-rows',action='store_true')
    args=p.parse_args();specs=json.loads(Path(args.config).read_text()) if args.config else configurations()
    if args.names: specs=[s for s in specs if s['name'] in args.names.split(',')]
    run(specs,args.seed,args.run,json.loads(Path(args.fixed).read_text()) if args.fixed else None,args.all_rows)
