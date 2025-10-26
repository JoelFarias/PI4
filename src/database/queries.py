"""
Módulo de queries SQL otimizadas para o banco de dados ENEM.

Este módulo centraliza todas as queries SQL utilizadas no dashboard,
facilitando manutenção e reutilização.
"""

import pandas as pd
import streamlit as st
from typing import Optional, List, Dict, Any
import logging

from .connection import DatabaseConnection
from ..utils.config import Config
from ..utils.constants import TABLE_PARTICIPANTES, TABLE_RESULTADOS, TABLE_MUNICIPIOS, COLUNAS_NOTAS


logger = logging.getLogger(__name__)


# ==============================================================================
# QUERIES BÁSICAS
# ==============================================================================

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando dados...")
def get_participantes_sample(
    limit: int = 1000,
    columns: Optional[List[str]] = None,
    where_clause: str = "",
    order_by: str = None
) -> pd.DataFrame:
    """
    Retorna amostra de participantes COM NOTAS (JOIN entre tabelas).
    NOTA: Faz JOIN entre PARTICIPANTES (dados socioeconômicos/demográficos) 
    e RESULTADOS (notas). Usa TABLESAMPLE para melhor performance.
    
    Args:
        limit: Número máximo de registros
        columns: Lista de colunas a selecionar (None = colunas principais)
        where_clause: Cláusula WHERE (sem o WHERE)
        order_by: Cláusula ORDER BY (None = sem ordenação para melhor performance)
        
    Returns:
        DataFrame com dados dos participantes + notas
    """
    try:
        # Colunas principais se não especificadas
        if columns is None:
            cols = """
                p.nu_ano, p.nu_inscricao, p.co_municipio_prova,
                p.tp_sexo, p.tp_faixa_etaria, p.idade_calculada,
                p.tp_cor_raca, p.tp_estado_civil,
                p.q001, p.q002, p.q003, p.q004, p.q005, p.q006, p.q007,
                p.tp_dependencia_adm_esc, p.tp_localizacao_esc, p.tp_st_conclusao,
                p.sg_uf_prova, p.regiao_nome_prova,
                r.nota_cn_ciencias_da_natureza,
                r.nota_ch_ciencias_humanas,
                r.nota_lc_linguagens_e_codigos,
                r.nota_mt_matematica,
                r.nota_redacao,
                r.nota_media_5_notas
            """
        else:
            # Mapeamento de aliases para colunas reais do banco
            COLUMN_ALIASES = {
                'escolaridade_pai': 'q001',
                'escolaridade_mae': 'q002',
                'ocupacao_pai': 'q003',
                'ocupacao_mae': 'q004',
                'pessoas_residencia': 'q005',
                'faixa_renda': 'q006',
                'municipio_nome_prova': 'no_municipio_prova',
                'uf_nome_prova': 'nome_uf_prova'
            }
            
            # Prefixar colunas com p. ou r. conforme necessário
            cols_prefixed = []
            for col in columns:
                # Mapear alias para coluna real
                real_col = COLUMN_ALIASES.get(col, col)
                
                if real_col.startswith('nota_'):
                    # SELECT com alias se diferente
                    if col != real_col:
                        cols_prefixed.append(f"r.{real_col} as {col}")
                    else:
                        cols_prefixed.append(f"r.{real_col}")
                else:
                    # SELECT com alias se diferente  
                    if col != real_col:
                        cols_prefixed.append(f"p.{real_col} as {col}")
                    else:
                        cols_prefixed.append(f"p.{real_col}")
            cols = ", ".join(cols_prefixed)
        
        # Montar query com JOIN - usar DISTINCT para evitar duplicatas
        query = f"""
            SELECT DISTINCT {cols}
            FROM {TABLE_PARTICIPANTES} p
            INNER JOIN {TABLE_RESULTADOS} r 
                ON p.nu_ano = r.nu_ano 
                AND p.co_municipio_prova = r.co_municipio_prova
        """
        
        # Adicionar WHERE se fornecido
        if where_clause:
            # Prefixar colunas no where_clause com r. se for nota_, senão com p.
            where_fixed = where_clause
            # Substituir referências a colunas de nota sem prefixo
            # Usar word boundaries para evitar substituições parciais
            import re
            
            # Primeiro, substituir aliases por colunas reais
            COLUMN_ALIASES = {
                'escolaridade_pai': 'q001',
                'escolaridade_mae': 'q002',
                'ocupacao_pai': 'q003',
                'ocupacao_mae': 'q004',
                'pessoas_residencia': 'q005',
                'faixa_renda': 'q006'
            }
            
            for alias, real_col in COLUMN_ALIASES.items():
                pattern = r'\b' + re.escape(alias) + r'\b'
                where_fixed = re.sub(pattern, real_col, where_fixed)
            
            # Depois, prefixar colunas de notas
            for col in COLUNAS_NOTAS:
                # Criar padrão que captura a coluna sem prefixo
                pattern = r'\b' + re.escape(col) + r'\b'
                # Verificar se existe sem prefixo
                if re.search(pattern, where_fixed):
                    # Substituir apenas se não tiver r. ou p. antes
                    where_fixed = re.sub(r'(?<!r\.)(?<!p\.)' + pattern, f'r.{col}', where_fixed)
            
            # Por fim, prefixar colunas q00X que não foram prefixadas
            for i in range(1, 8):
                q_col = f'q00{i}'
                pattern = r'\b' + re.escape(q_col) + r'\b'
                if re.search(pattern, where_fixed):
                    where_fixed = re.sub(r'(?<!r\.)(?<!p\.)' + pattern, f'p.{q_col}', where_fixed)
                    
            query += f" WHERE {where_fixed}"
        else:
            # Garantir que há notas
            query += " WHERE r.nota_media_5_notas IS NOT NULL"
        
        # LIMIT primeiro para reduzir dados antes de ordenar
        query += f" LIMIT {limit}"
        
        if order_by:
            # Envolver em subquery para ordenar apenas o subset
            query = f"SELECT * FROM ({query}) AS sample ORDER BY {order_by}"
        
        # Executar query
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        return pd.DataFrame(data, columns=columns_names)
        
    except Exception as e:
        logger.error(f"Erro ao carregar amostra de participantes: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando dados de notas...")
def get_resultados_sample(
    limit: int = 1000,
    columns: Optional[List[str]] = None,
    where_clause: str = "",
    order_by: str = None
) -> pd.DataFrame:
    """
    Retorna amostra de RESULTADOS (apenas notas, SEM JOIN com participantes).
    Use esta função quando precisar apenas de dados de notas/desempenho.
    
    Args:
        limit: Número máximo de registros
        columns: Lista de colunas a selecionar (None = todas as notas)
        where_clause: Cláusula WHERE (sem o WHERE)
        order_by: Cláusula ORDER BY
        
    Returns:
        DataFrame com dados de notas
    """
    try:
        # Colunas padrão se não especificadas
        if columns is None:
            cols = """
                nu_inscricao,
                nota_cn_ciencias_da_natureza,
                nota_ch_ciencias_humanas,
                nota_lc_linguagens_e_codigos,
                nota_mt_matematica,
                nota_redacao,
                nota_media_5_notas
            """
        else:
            cols = ", ".join(columns)
        
        # Montar query
        query = f"""
            SELECT {cols}
            FROM {TABLE_RESULTADOS}
        """
        
        # Adicionar WHERE se fornecido
        if where_clause:
            query += f" WHERE {where_clause}"
        else:
            # Garantir que há notas válidas
            query += " WHERE nota_media_5_notas IS NOT NULL"
        
        # Adicionar LIMIT
        query += f" LIMIT {limit}"
        
        # Adicionar ORDER BY se fornecido
        if order_by:
            query = f"SELECT * FROM ({query}) AS sample ORDER BY {order_by}"
        
        # Executar query
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        return pd.DataFrame(data, columns=columns_names)
        
    except Exception as e:
        logger.error(f"Erro ao carregar amostra de resultados: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando estatísticas...")
def get_notas_estatisticas() -> pd.DataFrame:
    """
    Retorna estatísticas descritivas das notas.
    NOTA: Usa tabela ed_enem_2024_resultados pois ed_enem_2024_participantes 
    não possui notas preenchidas.
    
    Returns:
        DataFrame com média, mediana, desvio padrão, min, max por prova
    """
    try:
        query = f"""
            SELECT
                COUNT(*) as total_participantes,
                -- Ciências da Natureza
                AVG(nota_cn_ciencias_da_natureza) as cn_media,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY nota_cn_ciencias_da_natureza) as cn_mediana,
                STDDEV(nota_cn_ciencias_da_natureza) as cn_desvio,
                MIN(nota_cn_ciencias_da_natureza) as cn_min,
                MAX(nota_cn_ciencias_da_natureza) as cn_max,
                -- Ciências Humanas
                AVG(nota_ch_ciencias_humanas) as ch_media,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY nota_ch_ciencias_humanas) as ch_mediana,
                STDDEV(nota_ch_ciencias_humanas) as ch_desvio,
                MIN(nota_ch_ciencias_humanas) as ch_min,
                MAX(nota_ch_ciencias_humanas) as ch_max,
                -- Linguagens e Códigos
                AVG(nota_lc_linguagens_e_codigos) as lc_media,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY nota_lc_linguagens_e_codigos) as lc_mediana,
                STDDEV(nota_lc_linguagens_e_codigos) as lc_desvio,
                MIN(nota_lc_linguagens_e_codigos) as lc_min,
                MAX(nota_lc_linguagens_e_codigos) as lc_max,
                -- Matemática
                AVG(nota_mt_matematica) as mt_media,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY nota_mt_matematica) as mt_mediana,
                STDDEV(nota_mt_matematica) as mt_desvio,
                MIN(nota_mt_matematica) as mt_min,
                MAX(nota_mt_matematica) as mt_max,
                -- Redação
                AVG(nota_redacao) as red_media,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY nota_redacao) as red_mediana,
                STDDEV(nota_redacao) as red_desvio,
                MIN(nota_redacao) as red_min,
                MAX(nota_redacao) as red_max,
                -- Média Geral
                AVG(nota_media_5_notas) as media_geral,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY nota_media_5_notas) as media_mediana,
                STDDEV(nota_media_5_notas) as media_desvio,
                MIN(nota_media_5_notas) as media_min,
                MAX(nota_media_5_notas) as media_max
            FROM {TABLE_RESULTADOS}
            WHERE nota_media_5_notas IS NOT NULL
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        return pd.DataFrame(data, columns=columns_names)
        
    except Exception as e:
        logger.error(f"Erro ao carregar estatísticas de notas: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando distribuição...")
def get_distribuicao_por_campo(campo: str, limit: int = 50) -> pd.DataFrame:
    """
    Retorna distribuição de frequência por campo.
    
    Args:
        campo: Nome do campo
        limit: Número máximo de categorias
        
    Returns:
        DataFrame com campo, quantidade e percentual
    """
    try:
        query = f"""
            SELECT
                {campo},
                COUNT(*) as quantidade,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentual
            FROM {TABLE_PARTICIPANTES}
            WHERE {campo} IS NOT NULL
            GROUP BY {campo}
            ORDER BY quantidade DESC
            LIMIT {limit}
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        return pd.DataFrame(data, columns=columns_names)
        
    except Exception as e:
        logger.error(f"Erro ao carregar distribuição de {campo}: {e}")
        return pd.DataFrame()


# ==============================================================================
# QUERIES DE CORRELAÇÃO
# ==============================================================================

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Analisando correlação...")
def get_media_por_escolaridade(tipo: str = 'pai') -> pd.DataFrame:
    """
    Retorna média de notas por escolaridade dos pais.
    NOTA: Busca APENAS da tabela RESULTADOS (não faz JOIN).
    
    Args:
        tipo: 'pai' ou 'mae'
        
    Returns:
        DataFrame com escolaridade e médias por prova
    """
    try:
        campo = 'q001' if tipo == 'pai' else 'q002'
        col_name = 'escolaridade_pai' if tipo == 'pai' else 'escolaridade_mae'
        
        query = f"""
            SELECT
                {campo} as {col_name},
                COUNT(*) as quantidade,
                ROUND(AVG(nota_cn_ciencias_da_natureza), 2) as cn_media,
                ROUND(AVG(nota_ch_ciencias_humanas), 2) as ch_media,
                ROUND(AVG(nota_lc_linguagens_e_codigos), 2) as lc_media,
                ROUND(AVG(nota_mt_matematica), 2) as mt_media,
                ROUND(AVG(nota_redacao), 2) as red_media,
                ROUND(AVG(nota_media_5_notas), 2) as media_geral
            FROM {TABLE_RESULTADOS}
            WHERE {campo} IS NOT NULL
                AND nota_media_5_notas IS NOT NULL
            GROUP BY {campo}
            ORDER BY {campo}
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        return pd.DataFrame(data, columns=columns_names)
        
    except Exception as e:
        logger.error(f"Erro ao carregar média por escolaridade do {tipo}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Analisando correlação...")
def get_media_por_ocupacao(tipo: str = 'pai') -> pd.DataFrame:
    """
    Retorna média de notas por ocupação dos pais.
    NOTA: Faz JOIN entre PARTICIPANTES (dados socioeconômicos) e RESULTADOS (notas).
    
    Args:
        tipo: 'pai' ou 'mae'
        
    Returns:
        DataFrame com ocupação e médias por prova
    """
    try:
        campo = 'q003' if tipo == 'pai' else 'q004'
        col_name = 'ocupacao_pai' if tipo == 'pai' else 'ocupacao_mae'
        
        query = f"""
            SELECT
                p.{campo} as {col_name},
                COUNT(*) as quantidade,
                ROUND(AVG(r.nota_cn_ciencias_da_natureza), 2) as cn_media,
                ROUND(AVG(r.nota_ch_ciencias_humanas), 2) as ch_media,
                ROUND(AVG(r.nota_lc_linguagens_e_codigos), 2) as lc_media,
                ROUND(AVG(r.nota_mt_matematica), 2) as mt_media,
                ROUND(AVG(r.nota_redacao), 2) as red_media,
                ROUND(AVG(r.nota_media_5_notas), 2) as media_geral
            FROM {TABLE_PARTICIPANTES} p
            INNER JOIN {TABLE_RESULTADOS} r 
                ON p.nu_ano = r.nu_ano 
                AND p.co_municipio_prova = r.co_municipio_prova
            WHERE p.{campo} IS NOT NULL
                AND r.nota_media_5_notas IS NOT NULL
            GROUP BY p.{campo}
            ORDER BY p.{campo}
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        return pd.DataFrame(data, columns=columns_names)
        
    except Exception as e:
        logger.error(f"Erro ao carregar média por ocupação do {tipo}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Analisando correlação...")
def get_media_por_renda() -> pd.DataFrame:
    """
    Retorna média de notas por faixa de renda familiar.
    NOTA: Faz JOIN entre PARTICIPANTES (dados socioeconômicos) e RESULTADOS (notas).
    
    Returns:
        DataFrame com renda e médias por prova
    """
    try:
        query = f"""
            SELECT
                p.q006 as faixa_renda,
                COUNT(*) as quantidade,
                ROUND(AVG(r.nota_cn_ciencias_da_natureza), 2) as cn_media,
                ROUND(AVG(r.nota_ch_ciencias_humanas), 2) as ch_media,
                ROUND(AVG(r.nota_lc_linguagens_e_codigos), 2) as lc_media,
                ROUND(AVG(r.nota_mt_matematica), 2) as mt_media,
                ROUND(AVG(r.nota_redacao), 2) as red_media,
                ROUND(AVG(r.nota_media_5_notas), 2) as media_geral
            FROM {TABLE_PARTICIPANTES} p
            INNER JOIN {TABLE_RESULTADOS} r 
                ON p.nu_ano = r.nu_ano 
                AND p.co_municipio_prova = r.co_municipio_prova
            WHERE p.q006 IS NOT NULL
                AND r.nota_media_5_notas IS NOT NULL
            GROUP BY p.q006
            ORDER BY p.q006
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        return pd.DataFrame(data, columns=columns_names)
        
    except Exception as e:
        logger.error(f"Erro ao carregar média por renda: {e}")
        return pd.DataFrame()


# ==============================================================================
# QUERIES GEOESPACIAIS
# ==============================================================================

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando dados geográficos...")
def get_media_por_municipio(top_n: int = 100, order: str = 'DESC') -> pd.DataFrame:
    """
    Retorna média de notas por município.
    NOTA: Usa tabela RESULTADOS para as notas.
    
    Args:
        top_n: Número de municípios a retornar
        order: 'DESC' para melhores, 'ASC' para piores
        
    Returns:
        DataFrame com município, UF, médias e localização
    """
    try:
        query = f"""
            SELECT
                r.no_municipio_prova as municipio,
                r.sg_uf_prova as uf,
                r.regiao_nome_prova as regiao,
                COUNT(*) as quantidade_participantes,
                ROUND(AVG(r.nota_media_5_notas), 2) as media_geral,
                ROUND(AVG(r.nota_cn_ciencias_da_natureza), 2) as media_cn,
                ROUND(AVG(r.nota_ch_ciencias_humanas), 2) as media_ch,
                ROUND(AVG(r.nota_lc_linguagens_e_codigos), 2) as media_lc,
                ROUND(AVG(r.nota_mt_matematica), 2) as media_mt,
                ROUND(AVG(r.nota_redacao), 2) as media_redacao,
                m.latitude,
                m.longitude
            FROM {TABLE_RESULTADOS} r
            LEFT JOIN {TABLE_MUNICIPIOS} m ON r.co_municipio_prova = m.codigo_municipio_dv
            WHERE r.nota_media_5_notas IS NOT NULL
            GROUP BY r.no_municipio_prova, r.sg_uf_prova, r.regiao_nome_prova, m.latitude, m.longitude
            HAVING COUNT(*) >= 30
            ORDER BY media_geral {order}
            LIMIT {top_n}
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        df = pd.DataFrame(data, columns=columns_names)
        
        # Converter colunas numéricas de Decimal para float
        numeric_cols = ['quantidade_participantes', 'media_geral', 'media_cn', 'media_ch', 
                       'media_lc', 'media_mt', 'media_redacao', 'latitude', 'longitude']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar média por município: {e}")
        return pd.DataFrame()


# ==============================================================================
# QUERIES PERSONALIZADAS
# ==============================================================================

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Executando query personalizada...")
def execute_custom_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """
    Executa query personalizada com segurança.
    
    Args:
        query: Query SQL
        params: Parâmetros da query (para evitar SQL injection)
        
    Returns:
        DataFrame com resultados
    """
    try:
        with DatabaseConnection.get_cursor() as cur:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        return pd.DataFrame(data, columns=columns_names)
        
    except Exception as e:
        logger.error(f"Erro ao executar query personalizada: {e}")
        return pd.DataFrame()


# ==============================================================================
# QUERIES PARA ANÁLISE GEOGRÁFICA
# ==============================================================================

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando dados por UF...")
def get_media_por_uf() -> pd.DataFrame:
    """
    Retorna média de notas por UF.
    NOTA: Usa tabela RESULTADOS para as notas.
    
    Returns:
        DataFrame com estatísticas por UF
    """
    try:
        query = f"""
            SELECT 
                sg_uf_prova as uf,
                COUNT(*) as total_participantes,
                AVG(nota_cn_ciencias_da_natureza) as cn_media,
                AVG(nota_ch_ciencias_humanas) as ch_media,
                AVG(nota_lc_linguagens_e_codigos) as lc_media,
                AVG(nota_mt_matematica) as mt_media,
                AVG(nota_redacao) as red_media,
                AVG(nota_media_5_notas) as media_geral
            FROM {TABLE_RESULTADOS}
            WHERE nota_media_5_notas IS NOT NULL
            GROUP BY sg_uf_prova
            ORDER BY media_geral DESC
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        df = pd.DataFrame(data, columns=columns_names)
        
        # Converter colunas numéricas de Decimal para float
        numeric_cols = ['total_participantes', 'cn_media', 'ch_media', 'lc_media', 
                       'mt_media', 'red_media', 'media_geral']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar média por UF: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando dados por região...")
def get_media_por_regiao() -> pd.DataFrame:
    """
    Retorna média de notas por região.
    NOTA: Usa tabela RESULTADOS para as notas.
    
    Returns:
        DataFrame com estatísticas por região
    """
    try:
        query = f"""
            SELECT 
                regiao_nome_prova as regiao,
                COUNT(*) as total_participantes,
                AVG(nota_cn_ciencias_da_natureza) as cn_media,
                AVG(nota_ch_ciencias_humanas) as ch_media,
                AVG(nota_lc_linguagens_e_codigos) as lc_media,
                AVG(nota_mt_matematica) as mt_media,
                AVG(nota_redacao) as red_media,
                AVG(nota_media_5_notas) as media_geral
            FROM {TABLE_RESULTADOS}
            WHERE nota_media_5_notas IS NOT NULL
            GROUP BY regiao_nome_prova
            ORDER BY media_geral DESC
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        df = pd.DataFrame(data, columns=columns_names)
        
        # Converter colunas numéricas de Decimal para float
        numeric_cols = ['total_participantes', 'cn_media', 'ch_media', 'lc_media', 
                       'mt_media', 'red_media', 'media_geral']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar média por região: {e}")
        return pd.DataFrame()


# ==============================================================================
# ALIASES PARA COMPATIBILIDADE
# ==============================================================================

def get_media_por_escolaridade_pai() -> pd.DataFrame:
    """Wrapper para get_media_por_escolaridade com tipo='pai'"""
    return get_media_por_escolaridade(tipo='pai')


def get_media_por_escolaridade_mae() -> pd.DataFrame:
    """Wrapper para get_media_por_escolaridade com tipo='mae'"""
    return get_media_por_escolaridade(tipo='mae')


def get_media_por_ocupacao_pai() -> pd.DataFrame:
    """Wrapper para get_media_por_ocupacao com tipo='pai'"""
    return get_media_por_ocupacao(tipo='pai')


def get_media_por_ocupacao_mae() -> pd.DataFrame:
    """Wrapper para get_media_por_ocupacao com tipo='mae'"""
    return get_media_por_ocupacao(tipo='mae')
