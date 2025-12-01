"""
Módulo de configuração centralizada do dashboard.

Este módulo gerencia todas as configurações da aplicação,
incluindo acesso ao banco de dados, cache, e outras variáveis.

Prioridade de configuração:
1. Streamlit Secrets (produção no Streamlit Cloud)
2. Variáveis de ambiente (.env para desenvolvimento local)
3. Valores padrão (fallback)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import streamlit as st
from dotenv import load_dotenv


# Carregar variáveis de ambiente do arquivo .env (se existir)
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Classe de configuração centralizada."""
    
    # ========================================================================
    # BANCO DE DADOS POSTGRESQL
    # ========================================================================
    
    @staticmethod
    def get_database_config() -> Dict[str, Any]:
        """
        Retorna configurações do banco de dados.
        
        Prioridade:
        1. Streamlit secrets
        2. Variáveis de ambiente
        3. Valores padrão
        
        Returns:
            Dict com configurações do banco
        """
        # Tentar pegar do Streamlit Secrets primeiro
        try:
            if hasattr(st, 'secrets') and 'database' in st.secrets:
                db_secrets = st.secrets['database']
                return {
                    'host': db_secrets.get('host', 'bigdata.dataiesb.com'),
                    'port': int(db_secrets.get('port', 5432)),
                    'database': db_secrets.get('database', 'iesb'),
                    'user': db_secrets.get('user', 'data_iesb'),
                    'password': db_secrets.get('password', 'iesb'),
                    'schema': db_secrets.get('schema', 'public'),
                    'pool_min_size': int(db_secrets.get('pool_min_size', 1)),
                    'pool_max_size': int(db_secrets.get('pool_max_size', 10)),
                    'pool_timeout': int(db_secrets.get('pool_timeout', 30)),
                }
        except Exception:
            pass
        
        # Fallback para variáveis de ambiente
        return {
            'host': os.getenv('DB_HOST', 'bigdata.dataiesb.com'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'iesb'),
            'user': os.getenv('DB_USER', 'data_iesb'),
            'password': os.getenv('DB_PASSWORD', 'iesb'),
            'schema': os.getenv('DB_SCHEMA', 'public'),
            'pool_min_size': int(os.getenv('DB_POOL_MIN_SIZE', 1)),
            'pool_max_size': int(os.getenv('DB_POOL_MAX_SIZE', 10)),
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', 30)),
        }
    
    # ========================================================================
    # APLICAÇÃO
    # ========================================================================
    
    APP_TITLE = os.getenv('APP_TITLE', 'Dashboard ENEM 2024 - Análise de Fatores Familiares')
    APP_ICON = os.getenv('APP_ICON', '📊')
    APP_LAYOUT = os.getenv('APP_LAYOUT', 'wide')
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    
    # ========================================================================
    # CACHE E PERFORMANCE
    # ========================================================================
    
    CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))  # 1 hora
    ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'True').lower() == 'true'
    CACHE_MAX_ENTRIES = int(os.getenv('CACHE_MAX_ENTRIES', 100))
    
    # ========================================================================
    # LIMITES DE DADOS
    # ========================================================================
    
    MAX_ROWS_DISPLAY = int(os.getenv('MAX_ROWS_DISPLAY', 10000))
    SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', 0))  # 0 = usar todos os dados
    DEFAULT_QUERY_LIMIT = int(os.getenv('DEFAULT_QUERY_LIMIT', 100000))
    
    # ========================================================================
    # MACHINE LEARNING
    # ========================================================================
    
    TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', 0.8))
    CV_FOLDS = int(os.getenv('CV_FOLDS', 5))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
    N_JOBS = int(os.getenv('N_JOBS', -1))  # -1 = usar todos os cores
    
    # ========================================================================
    # VISUALIZAÇÃO
    # ========================================================================
    
    CHART_THEME = os.getenv('CHART_THEME', 'plotly')
    DEFAULT_CHART_HEIGHT = int(os.getenv('DEFAULT_CHART_HEIGHT', 500))
    EXPORT_DPI = int(os.getenv('EXPORT_DPI', 300))
    
    # Paleta de cores
    COLOR_PRIMARY = '#1f77b4'      # Azul
    COLOR_SECONDARY = '#ff7f0e'    # Laranja
    COLOR_SUCCESS = '#2ca02c'      # Verde
    COLOR_WARNING = '#d62728'      # Vermelho
    COLOR_NEUTRAL = '#7f7f7f'      # Cinza
    
    COLOR_PALETTE = [
        '#1f77b4',  # Azul
        '#ff7f0e',  # Laranja
        '#2ca02c',  # Verde
        '#d62728',  # Vermelho
        '#9467bd',  # Roxo
        '#8c564b',  # Marrom
        '#e377c2',  # Rosa
        '#7f7f7f',  # Cinza
        '#bcbd22',  # Verde-amarelo
        '#17becf',  # Ciano
    ]
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    
    # ========================================================================
    # FEATURES FLAGS
    # ========================================================================
    
    ENABLE_GEOSPATIAL_ANALYSIS = os.getenv('ENABLE_GEOSPATIAL_ANALYSIS', 'True').lower() == 'true'
    ENABLE_ML_MODELS = os.getenv('ENABLE_ML_MODELS', 'True').lower() == 'true'
    ENABLE_EXPORT_DATA = os.getenv('ENABLE_EXPORT_DATA', 'True').lower() == 'true'
    ENABLE_ADMIN_FEATURES = os.getenv('ENABLE_ADMIN_FEATURES', 'False').lower() == 'true'
    
    # ========================================================================
    # CAMINHOS
    # ========================================================================
    
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    ASSETS_DIR = BASE_DIR / 'assets'
    
    # CSV do IFDM
    IFDM_CSV_PATH = DATA_DIR / 'ipeadata[29-09-2025-03-52].csv'
    
    # ========================================================================
    # MÉTODOS AUXILIARES
    # ========================================================================
    
    @classmethod
    def get_connection_string(cls) -> str:
        """
        Retorna string de conexão PostgreSQL.
        
        Returns:
            String de conexão no formato SQLAlchemy
        """
        db_config = cls.get_database_config()
        return (
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
    
    @classmethod
    def get_psycopg2_params(cls) -> Dict[str, Any]:
        """
        Retorna parâmetros para conexão psycopg2.
        
        Returns:
            Dict com parâmetros de conexão
        """
        db_config = cls.get_database_config()
        return {
            'host': db_config['host'],
            'port': db_config['port'],
            'database': db_config['database'],
            'user': db_config['user'],
            'password': db_config['password'],
        }
    
    @classmethod
    def is_production(cls) -> bool:
        """
        Verifica se está em ambiente de produção.
        
        Returns:
            True se em produção (Streamlit Cloud)
        """
        try:
            return hasattr(st, 'secrets') and 'database' in st.secrets
        except Exception:
            # Se st.secrets não existe ou gera erro, não está em produção
            return False
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """
        Retorna todas as configurações.
        
        Returns:
            Dict com todas as configurações
        """
        return {
            'app': {
                'title': cls.APP_TITLE,
                'icon': cls.APP_ICON,
                'layout': cls.APP_LAYOUT,
                'debug': cls.DEBUG_MODE,
            },
            'database': cls.get_database_config(),
            'cache': {
                'ttl': cls.CACHE_TTL,
                'enabled': cls.ENABLE_CACHE,
                'max_entries': cls.CACHE_MAX_ENTRIES,
            },
            'limits': {
                'max_rows_display': cls.MAX_ROWS_DISPLAY,
                'sample_size': cls.SAMPLE_SIZE,
                'default_query_limit': cls.DEFAULT_QUERY_LIMIT,
            },
            'ml': {
                'train_test_split': cls.TRAIN_TEST_SPLIT,
                'cv_folds': cls.CV_FOLDS,
                'random_state': cls.RANDOM_STATE,
                'n_jobs': cls.N_JOBS,
            },
            'visualization': {
                'theme': cls.CHART_THEME,
                'chart_height': cls.DEFAULT_CHART_HEIGHT,
                'export_dpi': cls.EXPORT_DPI,
                'color_palette': cls.COLOR_PALETTE,
            },
            'features': {
                'geospatial': cls.ENABLE_GEOSPATIAL_ANALYSIS,
                'ml_models': cls.ENABLE_ML_MODELS,
                'export': cls.ENABLE_EXPORT_DATA,
                'admin': cls.ENABLE_ADMIN_FEATURES,
            },
            'paths': {
                'base_dir': str(cls.BASE_DIR),
                'data_dir': str(cls.DATA_DIR),
                'assets_dir': str(cls.ASSETS_DIR),
                'ifdm_csv': str(cls.IFDM_CSV_PATH),
            },
            'environment': 'production' if cls.is_production() else 'development',
        }


# Instância global de configuração (singleton)
config = Config()
