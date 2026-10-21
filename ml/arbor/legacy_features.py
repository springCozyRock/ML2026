"""Protected legacy fold-fitted feature helper, extracted from ml/experiment.py.
Only numpy, pandas and sklearn required; no training or file access at import.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def boxcox1p(value, lam):
    return np.expm1(lam * np.log1p(value)) / lam if lam else np.log1p(value)

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

