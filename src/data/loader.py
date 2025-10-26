"""
Módulo de carregamento de dados do ENEM 2024.
"""

import pandas as pd
import streamlit as st
from typing import Optional, List, Dict, Any
import logging

from ..database.queries import (
    get_participantes_sample,
    get_notas_estatisticas,
    get_distribuicao_por_campo,
    execute_custom_query
)
from ..utils.config import Config
from ..utils.constants import COLUNAS_NOTAS


logger = logging.getLogger(__name__)


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando dados...")
def load_ifdm_data() -> pd.DataFrame:
    """Carrega dados do CSV do IFDM."""
    try:
        df = pd.read_csv(
            Config.IFDM_CSV_PATH,
            sep=';',
            skiprows=1,
            encoding='utf-8',
            decimal=','
        )
        
        df.columns = ['sigla', 'codigo', 'municipio', '2019', '2020', '2021', '2022', '2023']
        
        for year in ['2019', '2020', '2021', '2022', '2023']:
            df[year] = pd.to_numeric(df[year], errors='coerce')
        
        return df
    
    except Exception as e:
        logger.error(f"Erro ao carregar IFDM: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando amostra de dados...")
def load_sample_data(
    n_samples: int = 10000,
    columns: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Carrega amostra de dados dos participantes."""
    
    where_clause = ""
    if filters:
        conditions = []
        for key, value in filters.items():
            if isinstance(value, (list, tuple)):
                values_str = "', '".join(str(v) for v in value)
                conditions.append(f"{key} IN ('{values_str}')")
            else:
                conditions.append(f"{key} = '{value}'")
        where_clause = " AND ".join(conditions)
    
    return get_participantes_sample(
        limit=n_samples,
        columns=columns,
        where_clause=where_clause
    )


@st.cache_data(ttl=Config.CACHE_TTL)
def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Gera relatório de qualidade dos dados."""
    
    total_rows = len(df)
    
    missing_report = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        missing_report[col] = {
            'count': missing_count,
            'percentage': missing_pct
        }
    
    duplicates = df.duplicated().sum()
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    outliers_report = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        outliers_report[col] = {
            'count': outliers,
            'percentage': (outliers / total_rows) * 100
        }
    
    return {
        'total_rows': total_rows,
        'total_columns': len(df.columns),
        'missing_values': missing_report,
        'duplicates': duplicates,
        'outliers': outliers_report,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }
