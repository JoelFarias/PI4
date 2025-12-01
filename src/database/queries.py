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
    [DEPRECADO] Esta função está obsoleta. Use get_dados_municipio_completo() 
    para dados agregados por município.
    
    AVISO: JOIN entre PARTICIPANTES e RESULTADOS por município gera duplicatas
    pois não há relação direta por participante individual (nu_inscricao).
    As tabelas só podem ser unidas após agregação por município.
    
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
# QUERIES DE AGREGAÇÃO MUNICIPAL
# ==============================================================================

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando dados municipais completos...")
def get_dados_municipio_completo(min_participantes: int = 30) -> pd.DataFrame:
    """
    Retorna dados completos agregados por município.
    
    IMPORTANTE: Esta função agrega corretamente as tabelas PARTICIPANTES 
    (dados socioeconômicos) e RESULTADOS (notas) SEPARADAMENTE por município,
    e depois faz JOIN. Isso é necessário porque as tabelas não têm relação
    direta por participante individual.
    
    Args:
        min_participantes: Número mínimo de participantes por município
        
    Returns:
        DataFrame com dados agregados por município incluindo:
        - co_municipio_prova, no_municipio, sg_uf, regiao
        - total_participantes
        - Percentuais de características socioeconômicas
        - Médias de notas por prova
    """
    try:
        query = f"""
        WITH socio_agg AS (
            SELECT 
                co_municipio_prova,
                COUNT(*) as total_participantes_socio,
                -- Escolaridade (valores exatos do banco - SEM períodos finais)
                ROUND(100.0 * SUM(CASE WHEN q001 IN ('Completou a Faculdade, mas não completou a Pós-graduação', 'Completou a Pós-graduação') THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as perc_pai_ensino_superior,
                ROUND(100.0 * SUM(CASE WHEN q002 IN ('Completou a Faculdade, mas não completou a Pós-graduação', 'Completou a Pós-graduação') THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as perc_mae_ensino_superior,
                ROUND(100.0 * SUM(CASE WHEN q001 IN ('Completou a Faculdade, mas não completou a Pós-graduação', 'Completou a Pós-graduação') OR q002 IN ('Completou a Faculdade, mas não completou a Pós-graduação', 'Completou a Pós-graduação') THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as perc_pais_ensino_superior,
                -- Ocupação (Grupos 4 e 5 = qualificados)
                ROUND(100.0 * SUM(CASE WHEN q003 LIKE 'Grupo 4%' OR q003 LIKE 'Grupo 5%' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as perc_pai_ocupacao_qualificada,
                ROUND(100.0 * SUM(CASE WHEN q004 LIKE 'Grupo 4%' OR q004 LIKE 'Grupo 5%' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as perc_mae_ocupacao_qualificada,
                -- Renda (campo q007 - faixas altas: acima de R$ 14.120,00)
                ROUND(100.0 * SUM(CASE WHEN q007 IN (
                    'De R$ 14.120,01 até R$ 16.944,00',
                    'De R$ 16.944,01 até R$ 21.180,00',
                    'De R$ 21.180,01 até R$ 28.240,00',
                    'Acima de R$ 28.240,00'
                ) THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as perc_renda_alta,
                ROUND(100.0 * SUM(CASE WHEN q007 IN (
                    'Nenhuma renda',
                    'Até R$ 1.412,00',
                    'De R$ 1.412,01 até R$ 2.118,00',
                    'De R$ 2.118,01 até R$ 2.824,00'
                ) THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as perc_renda_baixa,
                -- Média de pessoas na residência
                ROUND(AVG(CASE WHEN q005 ~ '^[0-9]$' THEN q005::INTEGER ELSE NULL END), 2) as media_pessoas_residencia,
                -- Escola (tp_dependencia_adm_esc está NULL - não usamos filtros)
                0.0 as perc_escola_privada,
                0.0 as perc_escola_publica,
                -- Demografia
                ROUND(100.0 * SUM(CASE WHEN tp_sexo = 'F' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as perc_feminino,
                ROUND(AVG(idade_calculada), 1) as idade_media
            FROM {TABLE_PARTICIPANTES}
            WHERE co_municipio_prova IS NOT NULL
            GROUP BY co_municipio_prova
        ),
        desempenho_agg AS (
            SELECT 
                co_municipio_prova,
                no_municipio_prova,
                sg_uf_prova,
                regiao_nome_prova,
                COUNT(*) as total_participantes_notas,
                ROUND(AVG(nota_cn_ciencias_da_natureza), 2) as media_cn,
                ROUND(AVG(nota_ch_ciencias_humanas), 2) as media_ch,
                ROUND(AVG(nota_lc_linguagens_e_codigos), 2) as media_lc,
                ROUND(AVG(nota_mt_matematica), 2) as media_mt,
                ROUND(AVG(nota_redacao), 2) as media_redacao,
                ROUND(AVG(nota_media_5_notas), 2) as media_geral
            FROM {TABLE_RESULTADOS}
            WHERE co_municipio_prova IS NOT NULL
                AND nota_media_5_notas IS NOT NULL
            GROUP BY co_municipio_prova, no_municipio_prova, sg_uf_prova, regiao_nome_prova
        )
        SELECT 
            d.co_municipio_prova,
            d.no_municipio_prova as municipio,
            d.sg_uf_prova as uf,
            d.regiao_nome_prova as regiao,
            GREATEST(COALESCE(s.total_participantes_socio, 0), d.total_participantes_notas) as total_participantes,
            -- Dados socioeconômicos (com COALESCE para valores NULL)
            COALESCE(s.perc_pai_ensino_superior, 0.0) as perc_pai_ensino_superior,
            COALESCE(s.perc_mae_ensino_superior, 0.0) as perc_mae_ensino_superior,
            COALESCE(s.perc_pais_ensino_superior, 0.0) as perc_pais_ensino_superior,
            COALESCE(s.perc_pai_ocupacao_qualificada, 0.0) as perc_pai_ocupacao_qualificada,
            COALESCE(s.perc_mae_ocupacao_qualificada, 0.0) as perc_mae_ocupacao_qualificada,
            COALESCE(s.perc_renda_alta, 0.0) as perc_renda_alta,
            COALESCE(s.perc_renda_baixa, 0.0) as perc_renda_baixa,
            COALESCE(s.media_pessoas_residencia, 0.0) as media_pessoas_residencia,
            COALESCE(s.perc_escola_privada, 0.0) as perc_escola_privada,
            COALESCE(s.perc_escola_publica, 0.0) as perc_escola_publica,
            COALESCE(s.perc_feminino, 0.0) as perc_feminino,
            COALESCE(s.idade_media, 0.0) as idade_media,
            -- Dados de desempenho
            d.media_cn,
            d.media_ch,
            d.media_lc,
            d.media_mt,
            d.media_redacao,
            d.media_geral,
            -- Coordenadas
            m.latitude,
            m.longitude
        FROM desempenho_agg d
        LEFT JOIN socio_agg s ON d.co_municipio_prova = s.co_municipio_prova
        LEFT JOIN {TABLE_MUNICIPIOS} m ON d.co_municipio_prova = m.codigo_municipio_dv
        WHERE d.total_participantes_notas >= {min_participantes}
        ORDER BY d.media_geral DESC
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        logger.info(f"Query executada: {len(data)} linhas retornadas")
        
        df = pd.DataFrame(data, columns=columns_names)
        
        # Converter colunas numéricas
        numeric_cols = [col for col in df.columns if col not in ['co_municipio_prova', 'municipio', 'uf', 'regiao']]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Carregados {len(df)} municípios com dados completos (min_participantes={min_participantes})")
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados municipais completos: {e}", exc_info=True)
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando dados socioeconômicos agregados...")
def get_participantes_aggregated(
    min_participantes: int = 30,
    group_by: List[str] = None
) -> pd.DataFrame:
    """
    Retorna dados socioeconômicos agregados por município ou outras dimensões.
    
    Args:
        min_participantes: Número mínimo de participantes
        group_by: Lista de campos para agrupar (padrão: ['co_municipio_prova'])
        
    Returns:
        DataFrame com dados agregados
    """
    try:
        if group_by is None:
            group_by = ['co_municipio_prova', 'sg_uf_prova', 'regiao_nome_prova']
        
        group_cols = ', '.join(group_by)
        
        query = f"""
        SELECT 
            {group_cols},
            COUNT(*) as total_participantes,
            -- Escolaridade (usando textos completos)
            SUM(CASE WHEN q001 IN ('Completou a Faculdade, mas não completou a Pós-graduação', 'Completou a Pós-graduação') THEN 1 ELSE 0 END) as pai_superior_count,
            SUM(CASE WHEN q002 IN ('Completou a Faculdade, mas não completou a Pós-graduação', 'Completou a Pós-graduação') THEN 1 ELSE 0 END) as mae_superior_count,
            ROUND(100.0 * SUM(CASE WHEN q001 IN ('Completou a Faculdade, mas não completou a Pós-graduação', 'Completou a Pós-graduação') THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_pai_superior,
            ROUND(100.0 * SUM(CASE WHEN q002 IN ('Completou a Faculdade, mas não completou a Pós-graduação', 'Completou a Pós-graduação') THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_mae_superior,
            -- Ocupação (usando textos completos)
            ROUND(100.0 * SUM(CASE WHEN q003 LIKE 'Grupo 4%' OR q003 LIKE 'Grupo 5%' THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_pai_qualificado,
            ROUND(100.0 * SUM(CASE WHEN q004 LIKE 'Grupo 4%' OR q004 LIKE 'Grupo 5%' THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_mae_qualificado,
            -- Renda (campo q007 - faixas altas)
            ROUND(100.0 * SUM(CASE WHEN q007 IN (
                'De R$ 16.944,01 até R$ 21.180,00',
                'De R$ 21.180,01 até R$ 28.240,00',
                'Acima de R$ 28.240,00'
            ) THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_renda_alta,
            -- Escola (tp_dependencia_adm_esc como string)
            ROUND(100.0 * SUM(CASE WHEN tp_dependencia_adm_esc = '4' THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_escola_privada,
            -- Demografia
            ROUND(100.0 * SUM(CASE WHEN tp_sexo = 'F' THEN 1 ELSE 0 END) / COUNT(*), 2) as perc_feminino,
            ROUND(AVG(idade_calculada), 1) as idade_media
        FROM {TABLE_PARTICIPANTES}
        WHERE co_municipio_prova IS NOT NULL
        GROUP BY {group_cols}
        HAVING COUNT(*) >= {min_participantes}
        ORDER BY total_participantes DESC
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        df = pd.DataFrame(data, columns=columns_names)
        
        # Converter numéricas
        for col in df.columns:
            if col not in group_by:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar participantes agregados: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Carregando resultados agregados...")
def get_resultados_aggregated(
    min_participantes: int = 30,
    group_by: List[str] = None
) -> pd.DataFrame:
    """
    Retorna dados de desempenho agregados por município ou outras dimensões.
    
    Args:
        min_participantes: Número mínimo de participantes
        group_by: Lista de campos para agrupar (padrão: ['co_municipio_prova'])
        
    Returns:
        DataFrame com médias de notas agregadas
    """
    try:
        if group_by is None:
            group_by = ['co_municipio_prova', 'no_municipio_prova', 'sg_uf_prova', 'regiao_nome_prova']
        
        group_cols = ', '.join(group_by)
        
        query = f"""
        SELECT 
            {group_cols},
            COUNT(*) as total_participantes,
            ROUND(AVG(nota_cn_ciencias_da_natureza), 2) as media_cn,
            ROUND(AVG(nota_ch_ciencias_humanas), 2) as media_ch,
            ROUND(AVG(nota_lc_linguagens_e_codigos), 2) as media_lc,
            ROUND(AVG(nota_mt_matematica), 2) as media_mt,
            ROUND(AVG(nota_redacao), 2) as media_redacao,
            ROUND(AVG(nota_media_5_notas), 2) as media_geral,
            ROUND(STDDEV(nota_media_5_notas), 2) as desvio_padrao_geral,
            ROUND(MIN(nota_media_5_notas), 2) as min_nota_geral,
            ROUND(MAX(nota_media_5_notas), 2) as max_nota_geral
        FROM {TABLE_RESULTADOS}
        WHERE co_municipio_prova IS NOT NULL
            AND nota_media_5_notas IS NOT NULL
        GROUP BY {group_cols}
        HAVING COUNT(*) >= {min_participantes}
        ORDER BY media_geral DESC
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            columns_names = [desc[0] for desc in cur.description]
            data = cur.fetchall()
        
        df = pd.DataFrame(data, columns=columns_names)
        
        # Converter numéricas
        for col in df.columns:
            if col not in group_by:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar resultados agregados: {e}")
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
