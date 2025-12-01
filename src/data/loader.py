"""
Módulo de carregamento de dados do ENEM 2024.

IMPORTANTE: As tabelas ed_enem_2024_participantes (socioeconômico) e 
ed_enem_2024_resultados (notas) não têm relação direta por participante.
A análise é realizada no nível MUNICIPAL (análise ecológica).
"""

import pandas as pd
import streamlit as st
from typing import Optional, List, Dict, Any
import logging

from ..database.queries import (
    get_participantes_sample,
    get_notas_estatisticas,
    get_distribuicao_por_campo,
    execute_custom_query,
    get_dados_municipio_completo,
    get_participantes_aggregated,
    get_resultados_aggregated,
    get_media_por_municipio,
    get_media_por_uf,
    get_media_por_regiao
)
from ..utils.config import Config
from ..utils.constants import COLUNAS_NOTAS


logger = logging.getLogger(__name__)


# ==============================================================================
# FUNÇÕES DE CARREGAMENTO - NÍVEL MUNICIPAL (RECOMENDADO)
# ==============================================================================

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando dados municipais...")
def load_municipio_data(min_participantes: int = 30) -> pd.DataFrame:
    """
    Carrega dados completos agregados por município.
    
    Esta é a função principal para carregar dados para análise.
    Retorna um DataFrame com uma linha por município contendo:
    - Dados socioeconômicos agregados (percentuais)
    - Médias de desempenho
    - Informações geográficas
    
    Args:
        min_participantes: Mínimo de participantes por município (padrão: 30)
        
    Returns:
        DataFrame com dados municipais (~5.570 municípios)
    """
    try:
        logger.info(f"Iniciando carregamento de dados municipais (min_participantes={min_participantes})")
        df = get_dados_municipio_completo(min_participantes)
        
        if df.empty:
            logger.warning(f"⚠️ DataFrame vazio retornado! Nenhum município com >= {min_participantes} participantes")
        else:
            logger.info(f"✅ Sucesso: {len(df)} municípios carregados")
            logger.debug(f"Colunas disponíveis: {df.columns.tolist()}")
            logger.debug(f"Primeiros municípios: {df['municipio'].head(3).tolist() if 'municipio' in df.columns else 'N/A'}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados municipais: {e}", exc_info=True)
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL)
def load_municipio_features_for_model() -> tuple:
    """
    Carrega features municipais preparadas para machine learning.
    
    Returns:
        Tupla (DataFrame, lista de nomes de features)
    """
    try:
        df = load_municipio_data()
        
        if df.empty:
            return pd.DataFrame(), []
        
        # Features para modelos
        feature_cols = [
            'perc_pai_ensino_superior',
            'perc_mae_ensino_superior',
            'perc_pais_ensino_superior',
            'perc_pai_ocupacao_qualificada',
            'perc_mae_ocupacao_qualificada',
            'perc_renda_alta',
            'perc_renda_baixa',
            'media_pessoas_residencia',
            'perc_escola_privada',
            'perc_escola_publica',
            'perc_feminino',
            'idade_media',
            'total_participantes'
        ]
        
        # Filtrar apenas features disponíveis
        available_features = [col for col in feature_cols if col in df.columns]
        
        return df, available_features
        
    except Exception as e:
        logger.error(f"Erro ao preparar features municipais: {e}")
        return pd.DataFrame(), []


@st.cache_data(ttl=Config.CACHE_TTL)
def load_municipio_by_region(regiao: str = None, uf: str = None) -> pd.DataFrame:
    """
    Carrega dados municipais filtrados por região ou UF.
    
    Args:
        regiao: Nome da região (Norte, Nordeste, Sudeste, Sul, Centro-Oeste)
        uf: Sigla da UF (SP, RJ, MG, etc.)
        
    Returns:
        DataFrame com municípios filtrados
    """
    try:
        df = load_municipio_data()
        
        if df.empty:
            return df
        
        if uf:
            df = df[df['uf'] == uf]
        elif regiao:
            df = df[df['regiao'] == regiao]
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao filtrar municípios: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL)
def load_top_municipios(n: int = 100, order: str = 'DESC') -> pd.DataFrame:
    """
    Carrega top N municípios por desempenho.
    
    Args:
        n: Número de municípios
        order: 'DESC' para melhores, 'ASC' para piores
        
    Returns:
        DataFrame com top municípios
    """
    return get_media_por_municipio(top_n=n, order=order)


@st.cache_data(ttl=Config.CACHE_TTL)
def get_municipio_statistics(df: pd.DataFrame) -> Dict:
    """
    Calcula estatísticas descritivas dos dados municipais.
    
    Args:
        df: DataFrame com dados municipais
        
    Returns:
        Dicionário com estatísticas
    """
    if df.empty:
        return {}
    
    stats = {
        'total_municipios': len(df),
        'total_participantes': int(df['total_participantes'].sum()) if 'total_participantes' in df.columns else 0,
        'media_geral_brasil': round(df['media_geral'].mean(), 2) if 'media_geral' in df.columns else 0,
        'melhor_municipio': df.iloc[0]['municipio'] if len(df) > 0 and 'municipio' in df.columns else 'N/A',
        'melhor_nota': round(df.iloc[0]['media_geral'], 2) if len(df) > 0 and 'media_geral' in df.columns else 0,
        'regioes': df['regiao'].nunique() if 'regiao' in df.columns else 0,
        'ufs': df['uf'].nunique() if 'uf' in df.columns else 0
    }
    
    return stats


# ==============================================================================
# FUNÇÕES DE CARREGAMENTO - OUTRAS FONTES
# ==============================================================================

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
