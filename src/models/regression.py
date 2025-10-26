"""
Modelos de Regressão para predição de desempenho no ENEM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import streamlit as st


class RegressionModel:
    """Classe base para modelos de regressão"""
    
    def __init__(self, model_type: str = 'linear', random_state: int = 42):
        """
        Args:
            model_type: Tipo do modelo ('linear', 'ridge', 'lasso', 'elasticnet', 
                        'random_forest', 'gradient_boosting', 'xgboost')
            random_state: Seed para reprodutibilidade
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o modelo baseado no tipo"""
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elasticnet': ElasticNet(random_state=self.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        self.model = models.get(self.model_type, LinearRegression())
    
    @st.cache_data(ttl=3600)
    def train(_self, X_train: pd.DataFrame, y_train: pd.Series, 
              scale_features: bool = True):
        """
        Treina o modelo
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            scale_features: Se True, normaliza as features
        """
        _self.feature_names = X_train.columns.tolist()
        
        if scale_features and _self.model_type in ['ridge', 'lasso', 'elasticnet']:
            X_train_scaled = _self.scaler.fit_transform(X_train)
            _self.model.fit(X_train_scaled, y_train)
        else:
            _self.model.fit(X_train, y_train)
        
        return _self
    
    def predict(self, X_test: pd.DataFrame, scale_features: bool = True):
        """Faz predições"""
        if scale_features and self.model_type in ['ridge', 'lasso', 'elasticnet']:
            X_test_scaled = self.scaler.transform(X_test)
            return self.model.predict(X_test_scaled)
        else:
            return self.model.predict(X_test)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, 
                 scale_features: bool = True):
        """
        Avalia o modelo e calcula métricas
        
        Returns:
            dict com métricas (R², RMSE, MAE, MAPE)
        """
        y_pred = self.predict(X_test, scale_features)
        
        self.metrics = {
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return self.metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5,
                        scale_features: bool = True):
        """
        Validação cruzada
        
        Returns:
            dict com médias e desvios das métricas
        """
        if scale_features and self.model_type in ['ridge', 'lasso', 'elasticnet']:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        r2_scores = cross_val_score(self.model, X_scaled, y, cv=cv, 
                                     scoring='r2', n_jobs=-1)
        neg_mse_scores = cross_val_score(self.model, X_scaled, y, cv=cv,
                                          scoring='neg_mean_squared_error', n_jobs=-1)
        
        return {
            'R²_mean': r2_scores.mean(),
            'R²_std': r2_scores.std(),
            'RMSE_mean': np.sqrt(-neg_mse_scores.mean()),
            'RMSE_std': np.sqrt(neg_mse_scores.std())
        }
    
    def get_feature_importance(self):
        """
        Retorna importância das features (quando disponível)
        
        Returns:
            DataFrame com features e importâncias
        """
        if self.model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            importances = self.model.feature_importances_
            
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        elif self.model_type in ['linear', 'ridge', 'lasso', 'elasticnet']:
            coefs = self.model.coef_
            
            return pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefs,
                'abs_coefficient': np.abs(coefs)
            }).sort_values('abs_coefficient', ascending=False)
        
        return None
    
    def get_model_params(self):
        """Retorna parâmetros do modelo"""
        return self.model.get_params()
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                               param_grid: dict, cv: int = 5):
        """
        Otimização de hiperparâmetros com GridSearch
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            param_grid: Dicionário com hiperparâmetros para testar
            cv: Número de folds para validação cruzada
        
        Returns:
            Melhores parâmetros encontrados
        """
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }


@st.cache_data(ttl=3600)
def prepare_regression_data(df: pd.DataFrame, target_col: str, 
                             feature_cols: list, test_size: float = 0.2,
                             random_state: int = 42):
    """
    Prepara dados para regressão
    
    Args:
        df: DataFrame com os dados
        target_col: Coluna alvo (ex: 'nota_media_5_notas')
        feature_cols: Lista de colunas de features
        test_size: Proporção do conjunto de teste
        random_state: Seed para reprodutibilidade
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    df_clean = df[feature_cols + [target_col]].dropna()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def compare_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series,
                    models_to_compare: list = None):
    """
    Compara múltiplos modelos de regressão
    
    Args:
        X_train, X_test, y_train, y_test: Dados de treino e teste
        models_to_compare: Lista de tipos de modelo para comparar
    
    Returns:
        DataFrame com métricas de todos os modelos
    """
    if models_to_compare is None:
        models_to_compare = ['linear', 'ridge', 'lasso', 'random_forest', 
                             'gradient_boosting', 'xgboost']
    
    results = []
    
    for model_type in models_to_compare:
        model = RegressionModel(model_type=model_type)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        results.append({
            'Modelo': model_type,
            'R²': metrics['R²'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE']
        })
    
    return pd.DataFrame(results).sort_values('R²', ascending=False)


def get_residuals_analysis(model: RegressionModel, X_test: pd.DataFrame, 
                             y_test: pd.Series):
    """
    Análise de resíduos do modelo
    
    Returns:
        DataFrame com valores reais, preditos e resíduos
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    return pd.DataFrame({
        'real': y_test,
        'predicted': y_pred,
        'residual': residuals,
        'abs_residual': np.abs(residuals),
        'squared_residual': residuals ** 2
    })
