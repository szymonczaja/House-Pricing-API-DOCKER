from sklearn.base import BaseEstimator, TransformerMixin

class FeatureTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X_copy = X.copy()
    X_copy['TotalSf'] = (X_copy['TotalBsmtSF'] + X_copy['1stFlrSF'] + X_copy['2ndFlrSF'])
    X_copy['TotalBath'] = (X_copy['HalfBath'] * 0.5 + X_copy['FullBath'] + X_copy['BsmtFullBath'] + X_copy['BsmtHalfBath'] * 0.5)
    X_copy['TotalPorchSf'] = (X_copy['OpenPorchSF'] + X_copy['EnclosedPorch'] + X_copy['3SsnPorch'] + X_copy['ScreenPorch'])
    X_copy['Age_at_sell'] = (X_copy['YrSold'] - X_copy['YearBuilt'])
    X_copy['YrSinceRemod'] = (X_copy['YrSold'] - X_copy['YearRemodAdd'])
    X_copy['IsNew'] = (X_copy['YearBuilt'] == X_copy['YrSold']).astype(int)
    X_copy['HasGarage'] = (X_copy['GarageArea'] > 1).astype(int)
    X_copy['HasFireplace'] = (X_copy['Fireplaces'] > 0).astype(int)
    X_copy['TotalQuality'] = (X_copy['OverallQual'] + X_copy['OverallCond'])
    cols_to_drop = ['Id', '1stFlrSF', '2ndFlrSF', 'HalfBath', 'FullBath', 'BsmtFullBath', 'BsmtHalfBath',  'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                'ScreenPorch',  'YearBuilt', 'YrSold', 'YearRemodAdd']
    X_copy = X_copy.drop(columns=cols_to_drop, axis=1)
    return X_copy