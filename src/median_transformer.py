from sklearn.base import BaseEstimator, TransformerMixin

class GroupedMedianTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, X, y=None):
    X_copy = X.copy()
    self.global_median = X_copy.groupby('Neighborhood')['LotFrontage'].median()
    self.global_median2 = X_copy['LotFrontage'].median()
    return self

  def transform(self, X):
    X_copy = X.copy()
    X_copy['LotFrontage'] = X_copy['LotFrontage'].fillna(X_copy['Neighborhood'].map(self.global_median))
    X_copy['LotFrontage'] = X_copy['LotFrontage'].fillna(self.global_median2)
    return X_copy