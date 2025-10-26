"""
Modelos de Classificação para categorizar desempenho no ENEM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import streamlit as st


class ClassificationModel:
    """Classe base para modelos de classificação"""
    
    def __init__(self, model_type: str = 'logistic', random_state: int = 42):
        """
        Args:
            model_type: Tipo do modelo ('logistic', 'random_forest', 
                        'gradient_boosting', 'svm', 'xgboost')
            random_state: Seed para reprodutibilidade
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        self.metrics = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o modelo baseado no tipo"""
        models = {
            'logistic': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        self.model = models.get(self.model_type, LogisticRegression())
    
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
        _self.class_names = np.unique(y_train).tolist()
        
        y_train_encoded = _self.label_encoder.fit_transform(y_train)
        
        if scale_features and _self.model_type in ['logistic', 'svm']:
            X_train_scaled = _self.scaler.fit_transform(X_train)
            _self.model.fit(X_train_scaled, y_train_encoded)
        else:
            _self.model.fit(X_train, y_train_encoded)
        
        return _self
    
    def predict(self, X_test: pd.DataFrame, scale_features: bool = True):
        """Faz predições"""
        if scale_features and self.model_type in ['logistic', 'svm']:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred_encoded = self.model.predict(X_test_scaled)
        else:
            y_pred_encoded = self.model.predict(X_test)
        
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X_test: pd.DataFrame, scale_features: bool = True):
        """Retorna probabilidades das classes"""
        if scale_features and self.model_type in ['logistic', 'svm']:
            X_test_scaled = self.scaler.transform(X_test)
            return self.model.predict_proba(X_test_scaled)
        else:
            return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 scale_features: bool = True, average: str = 'weighted'):
        """
        Avalia o modelo e calcula métricas
        
        Args:
            average: Tipo de média para métricas multiclass ('weighted', 'macro', 'micro')
        
        Returns:
            dict com métricas
        """
        y_pred = self.predict(X_test, scale_features)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        self.metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test_encoded, y_pred_encoded, average=average, zero_division=0),
            'Recall': recall_score(y_test_encoded, y_pred_encoded, average=average, zero_division=0),
            'F1-Score': f1_score(y_test_encoded, y_pred_encoded, average=average, zero_division=0)
        }
        
        if len(self.class_names) == 2:
            y_proba = self.predict_proba(X_test, scale_features)[:, 1]
            self.metrics['ROC-AUC'] = roc_auc_score(y_test_encoded, y_proba)
        
        return self.metrics
    
    def get_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series,
                              scale_features: bool = True):
        """Retorna matriz de confusão"""
        y_pred = self.predict(X_test, scale_features)
        return confusion_matrix(y_test, y_pred, labels=self.class_names)
    
    def get_classification_report(self, X_test: pd.DataFrame, y_test: pd.Series,
                                    scale_features: bool = True):
        """Retorna relatório de classificação"""
        y_pred = self.predict(X_test, scale_features)
        return classification_report(y_test, y_pred, labels=self.class_names, 
                                       output_dict=True)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5,
                        scale_features: bool = True):
        """
        Validação cruzada estratificada
        
        Returns:
            dict com médias e desvios das métricas
        """
        y_encoded = self.label_encoder.fit_transform(y)
        
        if scale_features and self.model_type in ['logistic', 'svm']:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        accuracy_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=skf,
                                           scoring='accuracy', n_jobs=-1)
        f1_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=skf,
                                     scoring='f1_weighted', n_jobs=-1)
        
        return {
            'Accuracy_mean': accuracy_scores.mean(),
            'Accuracy_std': accuracy_scores.std(),
            'F1_mean': f1_scores.mean(),
            'F1_std': f1_scores.std()
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
        
        elif self.model_type == 'logistic':
            if len(self.class_names) == 2:
                coefs = self.model.coef_[0]
            else:
                coefs = np.mean(np.abs(self.model.coef_), axis=0)
            
            return pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefs,
                'abs_coefficient': np.abs(coefs)
            }).sort_values('abs_coefficient', ascending=False)
        
        return None
    
    def get_roc_curve_data(self, X_test: pd.DataFrame, y_test: pd.Series,
                            scale_features: bool = True):
        """
        Calcula dados para curva ROC (apenas classificação binária)
        
        Returns:
            dict com fpr, tpr, thresholds e auc
        """
        if len(self.class_names) != 2:
            return None
        
        y_test_encoded = self.label_encoder.transform(y_test)
        y_proba = self.predict_proba(X_test, scale_features)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_test_encoded, y_proba)
        auc = roc_auc_score(y_test_encoded, y_proba)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        }


@st.cache_data(ttl=3600)
def prepare_classification_data(df: pd.DataFrame, target_col: str,
                                  feature_cols: list, n_classes: int = 3,
                                  test_size: float = 0.2, random_state: int = 42):
    """
    Prepara dados para classificação
    
    Args:
        df: DataFrame com os dados
        target_col: Coluna alvo (ex: 'nota_media_5_notas')
        feature_cols: Lista de colunas de features
        n_classes: Número de classes para categorizar (padrão: 3 - Baixo/Médio/Alto)
        test_size: Proporção do conjunto de teste
        random_state: Seed para reprodutibilidade
    
    Returns:
        X_train, X_test, y_train, y_test, class_labels
    """
    df_clean = df[feature_cols + [target_col]].dropna()
    
    percentiles = np.linspace(0, 100, n_classes + 1)
    bins = [df_clean[target_col].quantile(p/100) for p in percentiles]
    
    if n_classes == 3:
        labels = ['Baixo', 'Médio', 'Alto']
    elif n_classes == 2:
        labels = ['Baixo', 'Alto']
    else:
        labels = [f'Classe_{i+1}' for i in range(n_classes)]
    
    df_clean['target_class'] = pd.cut(df_clean[target_col], bins=bins, 
                                        labels=labels, include_lowest=True)
    
    X = df_clean[feature_cols]
    y = df_clean['target_class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, labels


def compare_classifiers(X_train: pd.DataFrame, X_test: pd.DataFrame,
                         y_train: pd.Series, y_test: pd.Series,
                         models_to_compare: list = None):
    """
    Compara múltiplos modelos de classificação
    
    Args:
        X_train, X_test, y_train, y_test: Dados de treino e teste
        models_to_compare: Lista de tipos de modelo para comparar
    
    Returns:
        DataFrame com métricas de todos os modelos
    """
    if models_to_compare is None:
        models_to_compare = ['logistic', 'random_forest', 'gradient_boosting', 'xgboost']
    
    results = []
    
    for model_type in models_to_compare:
        model = ClassificationModel(model_type=model_type)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        results.append({
            'Modelo': model_type,
            **metrics
        })
    
    return pd.DataFrame(results).sort_values('F1-Score', ascending=False)
