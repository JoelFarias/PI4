"""
Módulo de processamento e limpeza de dados.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging

from ..utils.constants import (
    ESCOLARIDADE_PAI,
    ESCOLARIDADE_MAE,
    OCUPACAO_PAI,
    OCUPACAO_MAE,
    FAIXA_RENDA,
    SEXO,
    COR_RACA,
    ESTADO_CIVIL,
    get_escolaridade_nivel,
    get_grupo_ocupacional
)


logger = logging.getLogger(__name__)


def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
    """Remove outliers de colunas específicas."""
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
        
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & 
                (df_clean[col] <= upper_bound)
            ]
        
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < 3]
    
    return df_clean


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'drop',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Trata valores ausentes."""
    
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    
    elif strategy == 'mean':
        for col in columns:
            if col in df_clean.columns and df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    elif strategy == 'median':
        for col in columns:
            if col in df_clean.columns and df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    elif strategy == 'mode':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    return df_clean


def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variáveis categóricas para análise."""
    
    df_encoded = df.copy()
    
    if 'q001' in df_encoded.columns:
        df_encoded['escolaridade_pai_nivel'] = df_encoded['q001'].apply(get_escolaridade_nivel)
    
    if 'q002' in df_encoded.columns:
        df_encoded['escolaridade_mae_nivel'] = df_encoded['q002'].apply(get_escolaridade_nivel)
    
    if 'q003' in df_encoded.columns:
        df_encoded['ocupacao_pai_grupo'] = df_encoded['q003'].apply(get_grupo_ocupacional)
    
    if 'q004' in df_encoded.columns:
        df_encoded['ocupacao_mae_grupo'] = df_encoded['q004'].apply(get_grupo_ocupacional)
    
    if 'tp_sexo' in df_encoded.columns:
        df_encoded['sexo_encoded'] = df_encoded['tp_sexo'].map({'M': 1, 'F': 0})
    
    return df_encoded


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas para análise."""
    
    df_enhanced = df.copy()
    
    if all(col in df_enhanced.columns for col in ['escolaridade_pai_nivel', 'escolaridade_mae_nivel']):
        df_enhanced['escolaridade_pais_media'] = (
            df_enhanced['escolaridade_pai_nivel'] + 
            df_enhanced['escolaridade_mae_nivel']
        ) / 2
        
        df_enhanced['escolaridade_pais_max'] = df_enhanced[
            ['escolaridade_pai_nivel', 'escolaridade_mae_nivel']
        ].max(axis=1)
    
    if all(col in df_enhanced.columns for col in ['ocupacao_pai_grupo', 'ocupacao_mae_grupo']):
        df_enhanced['ocupacao_pais_media'] = (
            df_enhanced['ocupacao_pai_grupo'] + 
            df_enhanced['ocupacao_mae_grupo']
        ) / 2
    
    return df_enhanced


def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normaliza dados numéricos (0-1)."""
    
    df_norm = df.copy()
    
    for col in columns:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            
            if max_val > min_val:
                df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
    
    return df_norm


def standardize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Padroniza dados (z-score)."""
    
    df_std = df.copy()
    
    for col in columns:
        if col in df_std.columns:
            mean_val = df_std[col].mean()
            std_val = df_std[col].std()
            
            if std_val > 0:
                df_std[f'{col}_std'] = (df_std[col] - mean_val) / std_val
    
    return df_std


def prepare_for_ml(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara dados para machine learning."""
    
    df_clean = df.dropna(subset=[target_column])
    
    df_encoded = encode_categorical_variables(df_clean)
    df_enhanced = create_derived_features(df_encoded)
    
    feature_columns = [
        col for col in df_enhanced.columns 
        if col not in [target_column] and df_enhanced[col].dtype in ['int64', 'float64']
    ]
    
    X = df_enhanced[feature_columns].fillna(0)
    y = df_enhanced[target_column]
    
    return X, y
