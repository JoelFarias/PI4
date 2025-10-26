"""
Script de teste para verificar a configuração do ambiente e conexão com o banco de dados.

Execute este script antes de iniciar o dashboard para garantir que tudo está funcionando.

Uso:
    python tests/test_connection.py
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import DatabaseConnection
from src.utils.config import Config
from src.utils.constants import TABLE_PARTICIPANTES, TABLE_MUNICIPIOS


def print_header(text: str):
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def test_environment():
    """Testa se as variáveis de ambiente foram carregadas."""
    print_header("🔧 TESTE 1: Variáveis de Ambiente")
    
    print("\n📋 Configurações carregadas:")
    print(f"  • Título da aplicação: {Config.APP_TITLE}")
    print(f"  • Layout: {Config.APP_LAYOUT}")
    print(f"  • Debug Mode: {Config.DEBUG_MODE}")
    print(f"  • Cache TTL: {Config.CACHE_TTL}s")
    print(f"  • Ambiente: {'Produção' if Config.is_production() else 'Desenvolvimento'}")
    
    print("\n✅ Variáveis de ambiente carregadas com sucesso!")
    return True


def test_database_config():
    """Testa se as configurações do banco estão corretas."""
    print_header("🗄️ TESTE 2: Configurações do Banco de Dados")
    
    db_config = Config.get_database_config()
    
    print("\n📋 Configurações do PostgreSQL:")
    print(f"  • Host: {db_config['host']}")
    print(f"  • Porta: {db_config['port']}")
    print(f"  • Banco: {db_config['database']}")
    print(f"  • Usuário: {db_config['user']}")
    print(f"  • Senha: {'*' * len(db_config['password'])}")
    print(f"  • Schema: {db_config['schema']}")
    print(f"  • Pool Min: {db_config['pool_min_size']}")
    print(f"  • Pool Max: {db_config['pool_max_size']}")
    
    print("\n✅ Configurações do banco carregadas com sucesso!")
    return True


def test_connection():
    """Testa conexão com o banco de dados."""
    print_header("🔌 TESTE 3: Conexão com Banco de Dados")
    
    print("\n🔄 Tentando conectar ao PostgreSQL...")
    
    try:
        sucesso, mensagem = DatabaseConnection.test_connection()
        
        if sucesso:
            print(f"✅ {mensagem}")
            return True
        else:
            print(f"❌ {mensagem}")
            return False
    
    except Exception as e:
        print(f"❌ Erro ao testar conexão: {str(e)}")
        return False


def test_tables():
    """Testa acesso às tabelas."""
    print_header("📊 TESTE 4: Acesso às Tabelas")
    
    tabelas = [
        (TABLE_PARTICIPANTES, "Participantes do ENEM 2024"),
        (TABLE_MUNICIPIOS, "Municípios"),
    ]
    
    resultados = []
    
    for tabela, descricao in tabelas:
        print(f"\n🔍 Testando tabela: {tabela} ({descricao})")
        
        try:
            # Obter informações da tabela
            colunas = DatabaseConnection.get_table_info(tabela)
            
            if colunas:
                print(f"  ✅ Tabela encontrada com {len(colunas)} colunas")
                print(f"  📋 Primeiras 5 colunas:")
                for i, (nome, tipo) in enumerate(colunas[:5], 1):
                    print(f"     {i}. {nome} ({tipo})")
                
                # Contar registros
                count = DatabaseConnection.get_table_count(tabela)
                print(f"  📊 Total de registros: {count:,}".replace(",", "."))
                
                resultados.append(True)
            else:
                print(f"  ❌ Tabela não encontrada ou sem permissão de acesso")
                resultados.append(False)
        
        except Exception as e:
            print(f"  ❌ Erro ao acessar tabela: {str(e)}")
            resultados.append(False)
    
    if all(resultados):
        print("\n✅ Todas as tabelas estão acessíveis!")
        return True
    else:
        print("\n⚠️ Algumas tabelas não estão acessíveis!")
        return False


def test_query():
    """Testa execução de uma query simples."""
    print_header("🔍 TESTE 5: Execução de Query")
    
    print("\n🔄 Executando query de teste...")
    
    try:
        query = f"""
            SELECT COUNT(*) as total
            FROM {TABLE_PARTICIPANTES}
            LIMIT 1
        """
        
        with DatabaseConnection.get_cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
        
        if result:
            total = result[0]
            print(f"✅ Query executada com sucesso!")
            print(f"📊 Total de participantes no banco: {total:,}".replace(",", "."))
            return True
        else:
            print("❌ Query executada, mas sem resultados")
            return False
    
    except Exception as e:
        print(f"❌ Erro ao executar query: {str(e)}")
        return False


def main():
    """Função principal que executa todos os testes."""
    print("\n" + "=" * 70)
    print("  🧪 TESTES DE CONFIGURAÇÃO E CONEXÃO - Dashboard ENEM 2024")
    print("=" * 70)
    
    testes = [
        ("Variáveis de Ambiente", test_environment),
        ("Configurações do Banco", test_database_config),
        ("Conexão com PostgreSQL", test_connection),
        ("Acesso às Tabelas", test_tables),
        ("Execução de Query", test_query),
    ]
    
    resultados = {}
    
    for nome, teste_func in testes:
        try:
            resultado = teste_func()
            resultados[nome] = resultado
        except Exception as e:
            print(f"\n❌ Erro crítico no teste '{nome}': {str(e)}")
            resultados[nome] = False
    
    # Resumo final
    print_header("📊 RESUMO DOS TESTES")
    
    print("\n📋 Resultados:")
    for nome, resultado in resultados.items():
        status = "✅ PASSOU" if resultado else "❌ FALHOU"
        print(f"  • {nome}: {status}")
    
    total_testes = len(resultados)
    testes_ok = sum(resultados.values())
    taxa_sucesso = (testes_ok / total_testes) * 100
    
    print(f"\n📈 Taxa de Sucesso: {testes_ok}/{total_testes} ({taxa_sucesso:.1f}%)")
    
    if all(resultados.values()):
        print("\n🎉 TODOS OS TESTES PASSARAM! Sistema pronto para uso.")
        print("\n🚀 Próximo passo: Execute o dashboard com:")
        print("   streamlit run Home.py")
        return 0
    else:
        print("\n⚠️ ALGUNS TESTES FALHARAM! Verifique a configuração.")
        print("\n🔧 Possíveis soluções:")
        print("   1. Verifique as credenciais no arquivo .env")
        print("   2. Confirme que o banco de dados está acessível")
        print("   3. Verifique as permissões do usuário no banco")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
