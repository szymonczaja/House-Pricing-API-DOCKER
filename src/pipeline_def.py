import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from zero_transformer import ZeroImputerTransformer
from median_transformer import GroupedMedianTransformer
from fe_transformer import FeatureTransformer
from sklearn.base import clone
from config import (
    cols_for_fe,
    numeric_features_median,
    numeric_features_zero,
    categorical_features_ordinal,
    categorical_features_nominal,
    list_of_orders
)
def build_fe_stage():
    preprocessing_stage = Pipeline([
    ('feature_transformer', FeatureTransformer()),
    ('median_imputer',GroupedMedianTransformer())])
    return preprocessing_stage

def build_small_pipelines():
    num_median_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    num_zero_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())])

    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OrdinalEncoder(categories=list_of_orders, handle_unknown='use_encoded_value', unknown_value=-1))])

    nominal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return num_median_pipeline, num_zero_pipeline, ordinal_pipeline, nominal_pipeline

def build_final_preprocessor():
    num_median_pipeline, num_zero_pipeline, ordinal_pipeline, nominal_pipeline = build_small_pipelines()
    preprocessing_stage = build_fe_stage()

    preprocessor = ColumnTransformer([
    ('num_median', num_median_pipeline, numeric_features_median),
    ('num_zero', num_zero_pipeline, numeric_features_zero ),
    ('ord_feature', ordinal_pipeline, categorical_features_ordinal),
    ('nom_feature', nominal_pipeline, categorical_features_nominal)])

    final_preprocessing_pipeline = Pipeline([
        ('pre_imputer', ZeroImputerTransformer(cols=cols_for_fe)),
        ('preprocessing_stage', preprocessing_stage),
        ('preprocessor', preprocessor)])

    return final_preprocessing_pipeline

def build_final_model_pipeline(xgb_model, ridge_model, weights=[0.4, 0.6]):
    final_preprocessor = build_final_preprocessor()
    ridge_regressor = clone(ridge_model)
    xgb_regressor = clone(xgb_model)
    
    voting_regressor = VotingRegressor([
        ('ridge', ridge_regressor),
        ('xgb', xgb_regressor)
    ], weights=weights)

    voting_pipeline = Pipeline([
        ('final_preprocessor', final_preprocessor),
        ('voting_regressor', voting_regressor)])

    log1p_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)
    
    final_mlops_pipeline = TransformedTargetRegressor(
        regressor=voting_pipeline,
        transformer=log1p_transformer)

    return final_mlops_pipeline