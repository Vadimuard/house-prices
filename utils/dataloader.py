import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        dropped = ['Alley', 'PoolQC', 'MiscFeature',
                   'Fence', 'FireplaceQu', 'Utilities']
        self.dataset.drop(dropped, axis=1, inplace=True)

        # Numerical features
        self.dataset['LotFrontage'].fillna(
            value=self.dataset['LotFrontage'].median(), inplace=True)
        self.dataset['MasVnrArea'].fillna(value=self.dataset['MasVnrArea'].median(), inplace=True)
        self.dataset['BsmtFullBath'].fillna(
            value=self.dataset['BsmtFullBath'].median(), inplace=True)
        self.dataset['GarageYrBlt'].fillna(
            value=self.dataset['GarageYrBlt'].median(), inplace=True)
        self.dataset['BsmtFinSF1'].fillna(
            value=self.dataset['BsmtFinSF1'].median(), inplace=True)
        self.dataset['BsmtFinSF2'].fillna(value=self.dataset['BsmtFinSF2'].median(), inplace=True)
        self.dataset['BsmtUnfSF'].fillna(value=self.dataset['BsmtUnfSF'].median(), inplace=True)
        self.dataset['TotalBsmtSF'].fillna(
            value=self.dataset['TotalBsmtSF'].median(), inplace=True)
        self.dataset['BsmtHalfBath'].fillna(value=self.dataset['BsmtHalfBath'].median(), inplace=True)
        self.dataset['GarageCars'].fillna(
            value=self.dataset['GarageCars'].median(), inplace=True)
        self.dataset['GarageArea'].fillna(
            value=self.dataset['GarageArea'].median(), inplace=True)

        # Categorical features
        self.dataset['MasVnrType'].fillna(value=self.dataset['MasVnrType'].mode(), inplace=True)
        self.dataset['BsmtCond'].fillna(value=self.dataset['BsmtCond'].mode(), inplace=True)
        self.dataset['BsmtExposure'].fillna(value=self.dataset['BsmtExposure'].mode(), inplace=True)
        self.dataset['Electrical'].fillna(value=self.dataset['Electrical'].mode(), inplace=True)
        self.dataset['BsmtFinType2'].fillna(value=self.dataset['BsmtFinType2'].mode(), inplace=True)
        self.dataset['GarageType'].fillna(value=self.dataset['GarageType'].mode(), inplace=True)
        self.dataset['GarageFinish'].fillna(value=self.dataset['GarageFinish'].mode(), inplace=True)
        self.dataset['GarageQual'].fillna(value=self.dataset['GarageQual'].mode(), inplace=True)
        self.dataset['GarageCond'].fillna(value=self.dataset['GarageCond'].mode(), inplace=True)
        self.dataset['BsmtFinType1'].fillna(value=self.dataset['BsmtFinType1'].mode(), inplace=True)
        self.dataset['BsmtQual'].fillna(value=self.dataset['BsmtQual'].mode(), inplace=True)

        le = LabelEncoder()
        self.dataset['MSZoning'] = le.fit_transform(self.dataset['MSZoning'])
        self.dataset['Exterior1st'] = le.fit_transform(
            self.dataset['Exterior1st'])
        self.dataset['Exterior2nd'] = le.fit_transform(
            self.dataset['Exterior2nd'])
        self.dataset['KitchenQual'] = le.fit_transform(
            self.dataset['KitchenQual'])
        self.dataset['Functional'] = le.fit_transform(
            self.dataset['Functional'])
        self.dataset['SaleType'] = le.fit_transform(
            self.dataset['SaleType'])
        self.dataset['Street'] = le.fit_transform(self.dataset['Street'])
        self.dataset['LotShape'] = le.fit_transform(self.dataset['LotShape'])
        self.dataset['LandContour'] = le.fit_transform(
            self.dataset['LandContour'])
        self.dataset['LotConfig'] = le.fit_transform(self.dataset['LotConfig'])
        self.dataset['LandSlope'] = le.fit_transform(self.dataset['LandSlope'])
        self.dataset['Neighborhood'] = le.fit_transform(
            self.dataset['Neighborhood'])
        self.dataset['Condition1'] = le.fit_transform(
            self.dataset['Condition1'])
        self.dataset['Condition2'] = le.fit_transform(
            self.dataset['Condition2'])
        self.dataset['BldgType'] = le.fit_transform(self.dataset['BldgType'])
        self.dataset['HouseStyle'] = le.fit_transform(
            self.dataset['HouseStyle'])
        self.dataset['RoofStyle'] = le.fit_transform(self.dataset['RoofStyle'])
        self.dataset['RoofMatl'] = le.fit_transform(self.dataset['RoofMatl'])
        self.dataset['MasVnrType'] = le.fit_transform(
            self.dataset['MasVnrType'])
        self.dataset['ExterQual'] = le.fit_transform(self.dataset['ExterQual'])
        self.dataset['ExterCond'] = le.fit_transform(self.dataset['ExterCond'])
        self.dataset['Foundation'] = le.fit_transform(
            self.dataset['Foundation'])
        self.dataset['BsmtQual'] = le.fit_transform(self.dataset['BsmtQual'])
        self.dataset['BsmtCond'] = le.fit_transform(self.dataset['BsmtCond'])
        self.dataset['BsmtExposure'] = le.fit_transform(
            self.dataset['BsmtExposure'])
        self.dataset['BsmtFinType1'] = le.fit_transform(
            self.dataset['BsmtFinType1'])
        self.dataset['BsmtFinType2'] = le.fit_transform(
            self.dataset['BsmtFinType2'])
        self.dataset['Heating'] = le.fit_transform(self.dataset['Heating'])
        self.dataset['HeatingQC'] = le.fit_transform(self.dataset['HeatingQC'])
        self.dataset['CentralAir'] = le.fit_transform(
            self.dataset['CentralAir'])
        self.dataset['Electrical'] = le.fit_transform(
            self.dataset['Electrical'])
        self.dataset['GarageType'] = le.fit_transform(
            self.dataset['GarageType'])
        self.dataset['GarageFinish'] = le.fit_transform(
            self.dataset['GarageFinish'])
        self.dataset['GarageQual'] = le.fit_transform(
            self.dataset['GarageQual'])
        self.dataset['GarageCond'] = le.fit_transform(
            self.dataset['GarageCond'])
        self.dataset['PavedDrive'] = le.fit_transform(
            self.dataset['PavedDrive'])
        self.dataset['SaleCondition'] = le.fit_transform(
            self.dataset['SaleCondition'])

        scaler = StandardScaler()
        self.dataset = pd.DataFrame(scaler.fit_transform(self.dataset))
        return self.dataset
