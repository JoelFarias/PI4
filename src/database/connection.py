"""
Módulo de conexão com banco de dados PostgreSQL.

Este módulo gerencia a conexão com o banco de dados do IESB,
implementando pool de conexões e tratamento de erros.
"""

import psycopg2
from psycopg2 import pool, OperationalError, DatabaseError
from typing import Optional, Dict, Any, List, Tuple
import streamlit as st
from contextlib import contextmanager
import logging

from ..utils.config import Config


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Classe para gerenciar conexões com PostgreSQL usando pool de conexões.
    """
    
    _pool: Optional[pool.SimpleConnectionPool] = None
    
    @classmethod
    def get_pool(cls) -> pool.SimpleConnectionPool:
        """
        Retorna ou cria um pool de conexões.
        
        Returns:
            Pool de conexões PostgreSQL
        """
        if cls._pool is None or cls._pool.closed:
            try:
                db_config = Config.get_database_config()
                
                cls._pool = pool.SimpleConnectionPool(
                    minconn=db_config['pool_min_size'],
                    maxconn=db_config['pool_max_size'],
                    host=db_config['host'],
                    port=db_config['port'],
                    database=db_config['database'],
                    user=db_config['user'],
                    password=db_config['password'],
                    connect_timeout=db_config['pool_timeout'],
                    options=f"-c search_path={db_config['schema']}"
                )
                
                logger.info("Pool de conexões criado com sucesso")
                
            except Exception as e:
                logger.error(f"Erro ao criar pool de conexões: {e}")
                raise
        
        return cls._pool
    
    @classmethod
    @contextmanager
    def get_connection(cls):
        """
        Context manager para obter conexão do pool.
        
        Uso:
            with DatabaseConnection.get_connection() as conn:
                # usar conexão
        
        Yields:
            Conexão psycopg2
        """
        conn = None
        try:
            pool_instance = cls.get_pool()
            conn = pool_instance.getconn()
            
            if conn:
                yield conn
            else:
                raise OperationalError("Não foi possível obter conexão do pool")
                
        except Exception as e:
            logger.error(f"Erro ao obter conexão: {e}")
            raise
            
        finally:
            if conn:
                pool_instance = cls.get_pool()
                pool_instance.putconn(conn)
    
    @classmethod
    @contextmanager
    def get_cursor(cls, commit: bool = False):
        """
        Context manager para obter cursor de conexão.
        
        Args:
            commit: Se True, faz commit automático ao final
        
        Uso:
            with DatabaseConnection.get_cursor() as cur:
                cur.execute("SELECT * FROM tabela")
                results = cur.fetchall()
        
        Yields:
            Cursor psycopg2
        """
        with cls.get_connection() as conn:
            cur = conn.cursor()
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Erro ao executar query: {e}")
                raise
            finally:
                cur.close()
    
    @classmethod
    def test_connection(cls) -> Tuple[bool, str]:
        """
        Testa a conexão com o banco de dados.
        
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            with cls.get_cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                
                if result and result[0] == 1:
                    return True, "Conexão estabelecida com sucesso!"
                else:
                    return False, "Conexão estabelecida, mas teste falhou"
                    
        except OperationalError as e:
            return False, f"Erro de conexão: {str(e)}"
        except DatabaseError as e:
            return False, f"Erro de banco de dados: {str(e)}"
        except Exception as e:
            return False, f"Erro desconhecido: {str(e)}"
    
    @classmethod
    def get_table_info(cls, table_name: str) -> List[Tuple]:
        """
        Retorna informações sobre as colunas de uma tabela.
        
        Args:
            table_name: Nome da tabela
            
        Returns:
            Lista de tuplas com (nome_coluna, tipo_dados)
        """
        try:
            query = """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
            
            with cls.get_cursor() as cur:
                cur.execute(query, (table_name,))
                return cur.fetchall()
                
        except Exception as e:
            logger.error(f"Erro ao obter informações da tabela {table_name}: {e}")
            return []
    
    @classmethod
    def get_table_count(cls, table_name: str, where_clause: str = "") -> int:
        """
        Retorna o número de registros em uma tabela.
        
        Args:
            table_name: Nome da tabela
            where_clause: Cláusula WHERE opcional (sem o WHERE)
            
        Returns:
            Número de registros
        """
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            
            with cls.get_cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()
                return result[0] if result else 0
                
        except Exception as e:
            logger.error(f"Erro ao contar registros de {table_name}: {e}")
            return 0
    
    @classmethod
    def close_all_connections(cls):
        """
        Fecha todas as conexões do pool.
        """
        if cls._pool and not cls._pool.closed:
            cls._pool.closeall()
            logger.info("Pool de conexões fechado")


# ==============================================================================
# FUNÇÕES DE CACHE PARA STREAMLIT
# ==============================================================================

@st.cache_resource
def get_database_connection():
    """
    Retorna instância de DatabaseConnection com cache do Streamlit.
    
    Returns:
        Instância de DatabaseConnection
    """
    return DatabaseConnection()


@st.cache_data(ttl=Config.CACHE_TTL, show_spinner="Testando conexão...")
def test_database_connection() -> Tuple[bool, str]:
    """
    Testa conexão com cache.
    
    Returns:
        Tupla (sucesso, mensagem)
    """
    return DatabaseConnection.test_connection()


@st.cache_data(ttl=Config.CACHE_TTL)
def get_cached_table_info(table_name: str) -> List[Tuple]:
    """
    Retorna informações da tabela com cache.
    
    Args:
        table_name: Nome da tabela
        
    Returns:
        Lista de tuplas com informações das colunas
    """
    return DatabaseConnection.get_table_info(table_name)


@st.cache_data(ttl=Config.CACHE_TTL)
def get_cached_table_count(table_name: str, where_clause: str = "") -> int:
    """
    Retorna contagem de registros com cache.
    
    Args:
        table_name: Nome da tabela
        where_clause: Cláusula WHERE opcional
        
    Returns:
        Número de registros
    """
    return DatabaseConnection.get_table_count(table_name, where_clause)
