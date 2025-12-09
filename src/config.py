cols_for_fe = ['TotalBsmtSF','1stFlrSF', '2ndFlrSF','HalfBath','FullBath', 'BsmtFullBath','BsmtHalfBath',
                'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch', 'YrSold','YearBuilt',
                'GarageArea','Fireplaces','OverallQual','OverallCond']

numeric_features_median = ['LotFrontage','LotArea', 'LowQualFinSF', 'GrLivArea','BedroomAbvGr',
 'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','WoodDeckSF','PoolArea','MiscVal','MoSold','TotalSf']


numeric_features_zero = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF',
 'TotalBsmtSF','GarageArea','GarageCars']

categorical_features_ordinal = ['ExterCond',
 'ExterQual',
 'Functional',
 'BsmtQual',
 'BsmtCond',
 'HeatingQC',
 'KitchenQual',
 'FireplaceQu',
 'GarageQual',
 'GarageCond',
 'PoolQC',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'GarageFinish',
 'LandSlope',
 'LotShape',
 'PavedDrive',
 'Utilities',
 'OverallCond',
 'OverallQual'] 

categorical_features_nominal = ['MSZoning',
 'Street',
 'Alley',
 'LandContour',
 'LotConfig',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'Foundation',
 'Heating',
 'CentralAir',
 'Electrical',
 'GarageType',
 'Fence',
 'MiscFeature',
 'SaleType',
 'SaleCondition',
 'MSSubClass']

quality_scale = ['Missing', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
ExterCond_order = quality_scale
ExterQual_order = quality_scale
Functional_order = ['Missing', 'Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']
BsmtQual_order = quality_scale
BsmtCond_order = quality_scale
HeatingQC_order = quality_scale
KitchenQual_order = quality_scale
FireplaceQu_order = quality_scale
GarageQual_order = quality_scale
GarageCond_order = quality_scale
PoolQC_order = ['Missing', 'Fa', 'TA', 'Gd', 'Ex']
BsmtExposure_order = ['Missing', 'No', 'Mn', 'Av', 'Gd']
BsmtFinType1_order = ['Missing', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
BsmtFinType2_order = ['Missing', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
GarageFinish_order = ['Missing', 'Unf', 'RFn', 'Fin']
LandSlope_order = ['Gtl', 'Mod', 'Sev']
LotShape_order = ['Reg', 'IR1', 'IR2', 'IR3']
PavedDrive_order = ['N', 'P', 'Y']
Utilities_order = ['Missing', 'ELO', 'NoSeWa', 'NoSewr', 'AllPub']
OverallCond_order = list(range(1, 11))
OverallQual_order = list(range(1, 11))

list_of_orders = [ExterCond_order, ExterQual_order, Functional_order, BsmtQual_order, BsmtCond_order, HeatingQC_order, KitchenQual_order, FireplaceQu_order,
                 GarageQual_order, GarageCond_order, PoolQC_order, BsmtExposure_order, BsmtFinType1_order, BsmtFinType2_order, GarageFinish_order, LandSlope_order,
                 LotShape_order, PavedDrive_order, Utilities_order, OverallCond_order, OverallQual_order]

