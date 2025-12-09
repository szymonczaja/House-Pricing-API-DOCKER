from sklearn.base import BaseEstimator, TransformerMixin

class ZeroImputerTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X_copy = X.copy()
    for col in self.cols:
      X_copy[col] = X_copy[col].fillna(0)
    return X_copy