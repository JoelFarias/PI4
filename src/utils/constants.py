"""
Constantes e dicionários de mapeamento utilizados no dashboard.

Este módulo centraliza todos os valores fixos e mapeamentos
de códigos para descrições legíveis das variáveis do ENEM.
"""

from typing import Dict, List


# ==============================================================================
# TABELAS DO BANCO DE DADOS
# ==============================================================================

TABLE_PARTICIPANTES = 'ed_enem_2024_participantes'
TABLE_RESULTADOS = 'ed_enem_2024_resultados'
TABLE_MUNICIPIOS = 'municipio'


# ==============================================================================
# COLUNAS DE NOTAS
# ==============================================================================

COLUNAS_NOTAS = [
    'nota_cn_ciencias_da_natureza',
    'nota_ch_ciencias_humanas',
    'nota_lc_linguagens_e_codigos',
    'nota_mt_matematica',
    'nota_redacao',
    'nota_media_5_notas'
]

NOMES_PROVAS = {
    'nota_cn_ciencias_da_natureza': 'Ciências da Natureza',
    'nota_ch_ciencias_humanas': 'Ciências Humanas',
    'nota_lc_linguagens_e_codigos': 'Linguagens e Códigos',
    'nota_mt_matematica': 'Matemática',
    'nota_redacao': 'Redação',
    'nota_media_5_notas': 'Média Geral'
}

SIGLAS_PROVAS = {
    'nota_cn_ciencias_da_natureza': 'CN',
    'nota_ch_ciencias_humanas': 'CH',
    'nota_lc_linguagens_e_codigos': 'LC',
    'nota_mt_matematica': 'MT',
    'nota_redacao': 'RED',
    'nota_media_5_notas': 'MÉDIA'
}


# ==============================================================================
# VARIÁVEIS SOCIOECONÔMICAS - QUESTIONÁRIO
# ==============================================================================

# Q001 - Escolaridade do Pai
ESCOLARIDADE_PAI = {
    'A': 'Nunca estudou',
    'B': 'Não completou a 4ª série/5º ano do Ensino Fundamental',
    'C': 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental',
    'D': 'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio',
    'E': 'Completou o Ensino Médio, mas não completou a Faculdade',
    'F': 'Completou a Faculdade, mas não completou a Pós-graduação',
    'G': 'Completou a Pós-graduação',
    'H': 'Não sei',
}

# Q002 - Escolaridade da Mãe
ESCOLARIDADE_MAE = {
    'A': 'Nunca estudou',
    'B': 'Não completou a 4ª série/5º ano do Ensino Fundamental',
    'C': 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental',
    'D': 'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio',
    'E': 'Completou o Ensino Médio, mas não completou a Faculdade',
    'F': 'Completou a Faculdade, mas não completou a Pós-graduação',
    'G': 'Completou a Pós-graduação',
    'H': 'Não sei',
}

# Níveis de escolaridade (ordenados)
NIVEL_ESCOLARIDADE_ORDEM = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Q003 - Ocupação do Pai
OCUPACAO_PAI = {
    'A': 'Grupo 1: Lavrador, agricultor sem empregados, bóia fria, criador de animais, pescador, lenhador, seringueiro, extrativista',
    'B': 'Grupo 2: Diarista, empregado doméstico, cuidador de idosos, babá, cozinheiro (em casas particulares), motorista particular, jardineiro, faxineiro de empresas e prédios, vigilante, porteiro, carteiro, office-boy, vendedor, caixa, atendente de loja, auxiliar administrativo, recepcionista, servente de pedreiro, repositor de mercadoria',
    'C': 'Grupo 3: Padeiro, cozinheiro industrial ou em restaurantes, sapateiro, costureiro, joalheiro, torneiro mecânico, operador de máquinas, soldador, operário de fábrica, trabalhador da mineração, pedreiro, pintor, eletricista, encanador, motorista, caminhoneiro, taxista',
    'D': 'Grupo 4: Professor (de ensino fundamental ou médio, idioma, música, artes, etc.), técnico (de enfermagem, contabilidade, eletrônica, etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretor de imóveis, supervisor de vendas, gerente, mestre de obras, pastor, microempresário (proprietário de empresa com menos de 10 empregados), pequeno comerciante, pequeno proprietário de terras, trabalhador autônomo ou por conta própria',
    'E': 'Grupo 5: Médico, engenheiro, dentista, psicólogo, economista, advogado, juiz, promotor, defensor, delegado, tenente, capitão, coronel, professor universitário, diretor em empresas públicas ou privadas, político, proprietário de empresas com mais de 10 empregados',
    'F': 'Não sei',
}

# Q004 - Ocupação da Mãe
OCUPACAO_MAE = {
    'A': 'Grupo 1: Lavradora, agricultora sem empregados, bóia fria, criadora de animais, pescadora, lenhadora, seringueira, extrativista',
    'B': 'Grupo 2: Diarista, empregada doméstica, cuidadora de idosos, babá, cozinheira (em casas particulares), motorista particular, jardineira, faxineira de empresas e prédios, vigilante, porteira, carteira, office-boy, vendedora, caixa, atendente de loja, auxiliar administrativa, recepcionista, servente de pedreiro, repositora de mercadoria',
    'C': 'Grupo 3: Padeira, cozinheira industrial ou em restaurantes, sapateira, costureira, joalheira, torneira mecânica, operadora de máquinas, soldadora, operária de fábrica, trabalhadora da mineração, pedreira, pintora, eletricista, encanadora, motorista, caminhoneira, taxista',
    'D': 'Grupo 4: Professora (de ensino fundamental ou médio, idioma, música, artes, etc.), técnica (de enfermagem, contabilidade, eletrônica, etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretora de imóveis, supervisora de vendas, gerente, mestre de obras, pastora, microempresária (proprietária de empresa com menos de 10 empregados), pequena comerciante, pequena proprietária de terras, trabalhadora autônoma ou por conta própria',
    'E': 'Grupo 5: Médica, engenheira, dentista, psicóloga, economista, advogada, juíza, promotora, defensora, delegada, tenente, capitã, coronel, professora universitária, diretora em empresas públicas ou privadas, política, proprietária de empresas com mais de 10 empregados',
    'F': 'Não sei',
}

# Grupos ocupacionais simplificados
GRUPOS_OCUPACIONAIS = {
    'A': 'Grupo 1 - Agricultura e Extrativismo',
    'B': 'Grupo 2 - Serviços Básicos',
    'C': 'Grupo 3 - Operacional e Técnico',
    'D': 'Grupo 4 - Profissionais de Nível Médio/Superior',
    'E': 'Grupo 5 - Alta Qualificação',
    'F': 'Não sei',
}

# Q005 - Quantidade de pessoas na residência
PESSOAS_RESIDENCIA = {
    '0': 'Nenhuma',
    '1': 'Uma',
    '2': 'Duas',
    '3': 'Três',
    '4': 'Quatro',
    '5': 'Cinco',
    '6': 'Seis',
    '7': 'Sete',
    '8': 'Oito ou mais',
}

# Q006 - Renda familiar mensal
FAIXA_RENDA = {
    'A': 'Nenhuma renda',
    'B': 'Até R$ 1.412,00',
    'C': 'De R$ 1.412,01 até R$ 2.824,00',
    'D': 'De R$ 2.824,01 até R$ 4.236,00',
    'E': 'De R$ 4.236,01 até R$ 5.648,00',
    'F': 'De R$ 5.648,01 até R$ 7.060,00',
    'G': 'De R$ 7.060,01 até R$ 8.472,00',
    'H': 'De R$ 8.472,01 até R$ 9.884,00',
    'I': 'De R$ 9.884,01 até R$ 11.296,00',
    'J': 'De R$ 11.296,01 até R$ 12.708,00',
    'K': 'De R$ 12.708,01 até R$ 14.120,00',
    'L': 'De R$ 14.120,01 até R$ 16.956,00',
    'M': 'De R$ 16.956,01 até R$ 21.204,00',
    'N': 'De R$ 21.204,01 até R$ 28.240,00',
    'O': 'De R$ 28.240,01 até R$ 42.360,00',
    'P': 'De R$ 42.360,01 até R$ 56.480,00',
    'Q': 'Mais de R$ 56.480,00',
}


# ==============================================================================
# VARIÁVEIS DEMOGRÁFICAS
# ==============================================================================

# Faixa Etária
FAIXA_ETARIA = {
    1: 'Menor de 17 anos',
    2: '17 anos',
    3: '18 anos',
    4: '19 anos',
    5: '20 anos',
    6: '21 anos',
    7: '22 anos',
    8: '23 anos',
    9: '24 anos',
    10: '25 anos',
    11: 'Entre 26 e 30 anos',
    12: 'Entre 31 e 35 anos',
    13: 'Entre 36 e 40 anos',
    14: 'Entre 41 e 45 anos',
    15: 'Entre 46 e 50 anos',
    16: 'Entre 51 e 55 anos',
    17: 'Entre 56 e 60 anos',
    18: 'Entre 61 e 65 anos',
    19: 'Entre 66 e 70 anos',
    20: 'Maior de 70 anos',
}

# Sexo
SEXO = {
    'M': 'Masculino',
    'F': 'Feminino',
}

# Cor/Raça
COR_RACA = {
    0: 'Não declarado',
    1: 'Branca',
    2: 'Preta',
    3: 'Parda',
    4: 'Amarela',
    5: 'Indígena',
}

# Estado Civil
ESTADO_CIVIL = {
    0: 'Não informado',
    1: 'Solteiro(a)',
    2: 'Casado(a)/União estável',
    3: 'Divorciado(a)/Desquitado(a)/Separado(a)',
    4: 'Viúvo(a)',
}


# ==============================================================================
# VARIÁVEIS EDUCACIONAIS
# ==============================================================================

# Tipo de Escola
TIPO_ESCOLA = {
    1: 'Escola Pública',
    2: 'Escola Privada',
    3: 'Não informado',
}

# Dependência Administrativa da Escola
DEPENDENCIA_ADMINISTRATIVA = {
    1: 'Federal',
    2: 'Estadual',
    3: 'Municipal',
    4: 'Privada',
}

# Localização da Escola
LOCALIZACAO_ESCOLA = {
    1: 'Urbana',
    2: 'Rural',
}

# Situação de Conclusão
SITUACAO_CONCLUSAO = {
    1: 'Já concluí o Ensino Médio',
    2: 'Estou cursando e concluirei o Ensino Médio em 2024',
    3: 'Estou cursando e concluirei o Ensino Médio após 2024',
    4: 'Não concluí e não estou cursando o Ensino Médio',
}

# Tipo de Ensino
TIPO_ENSINO = {
    1: 'Ensino Regular',
    2: 'Educação de Jovens e Adultos (EJA)',
}


# ==============================================================================
# REGIÕES DO BRASIL
# ==============================================================================

REGIOES = {
    1: 'Norte',
    2: 'Nordeste',
    3: 'Sudeste',
    4: 'Sul',
    5: 'Centro-Oeste',
}

SIGLAS_UF = [
    'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
    'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
    'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
]

UF_REGIAO = {
    # Norte
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 
    'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    # Nordeste
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste',
    'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    # Sudeste
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    # Sul
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul',
    # Centro-Oeste
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'MT': 'Centro-Oeste',
}


# ==============================================================================
# CATEGORIAS DE DESEMPENHO
# ==============================================================================

CATEGORIAS_DESEMPENHO = {
    'baixo': 'Baixo Desempenho',
    'medio': 'Médio Desempenho',
    'alto': 'Alto Desempenho',
}

# Percentis para classificação
PERCENTIL_BAIXO = 33
PERCENTIL_ALTO = 67


# ==============================================================================
# MENSAGENS E TEXTOS
# ==============================================================================

TEXTO_CONTEXTUALIZACAO = """
Este dashboard apresenta uma análise exploratória e preditiva do **ENEM 2024**, 
com foco especial na correlação entre **fatores socioeconômicos familiares** 
(escolaridade e ocupação dos pais) e o **desempenho acadêmico** dos estudantes.

A análise utiliza dados de **4,3 milhões de participantes** do ENEM 2024, 
armazenados em um banco de dados PostgreSQL, e aplica técnicas de estatística, 
visualização de dados e machine learning para identificar padrões e insights relevantes.
"""

TEXTO_METODOLOGIA = """
### Metodologia de Análise

1. **Análise Exploratória de Dados (EDA)**
   - Estatísticas descritivas
   - Distribuições e outliers
   - Visualizações univariadas e bivariadas

2. **Análise de Correlação**
   - Correlação de Pearson (variáveis contínuas)
   - Correlação de Spearman (variáveis ordinais)
   - Testes de hipóteses (ANOVA, Qui-quadrado)

3. **Modelos Preditivos**
   - Regressão: Linear, Ridge, Lasso, Random Forest, XGBoost
   - Classificação: Logística, SVM, Random Forest, XGBoost
   - Clustering: K-Means, DBSCAN

4. **Validação de Modelos**
   - Train/Test Split (80/20)
   - K-Fold Cross-Validation (k=5)
   - Métricas: R², RMSE, MAE, Accuracy, Precision, Recall, F1
"""


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def get_escolaridade_nivel(codigo: str) -> int:
    """
    Converte código de escolaridade para nível numérico.
    
    Args:
        codigo: Código da escolaridade (A-H)
        
    Returns:
        Nível numérico (0-7)
    """
    return NIVEL_ESCOLARIDADE_ORDEM.index(codigo) if codigo in NIVEL_ESCOLARIDADE_ORDEM else -1


def get_grupo_ocupacional(codigo: str) -> int:
    """
    Converte código de ocupação para grupo numérico.
    
    Args:
        codigo: Código da ocupação (A-F)
        
    Returns:
        Grupo numérico (1-6)
    """
    grupos = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    return grupos.get(codigo, 0)


def categorizar_desempenho(nota: float, percentil_33: float, percentil_67: float) -> str:
    """
    Categoriza desempenho baseado em percentis.
    
    Args:
        nota: Nota do estudante
        percentil_33: Valor do percentil 33
        percentil_67: Valor do percentil 67
        
    Returns:
        Categoria: 'baixo', 'medio' ou 'alto'
    """
    if nota < percentil_33:
        return 'baixo'
    elif nota < percentil_67:
        return 'medio'
    else:
        return 'alto'
