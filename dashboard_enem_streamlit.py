import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import numpy as np

# -------------------- Configurações --------------------
st.set_page_config(page_title="ENEM 2024 - Dashboard (Agregado Municipal)", layout="wide")

DB_CONFIG = {
    'host': 'bigdata.dataiesb.com',
    'database': 'iesb',
    'port': 5432,
    'user': 'data_iesb',
    'password': 'iesb'
}

Q005_MAP = {
    'A': 'Nenhuma Renda', 'B': 'Até R$ 1.320,00', 'C': 'De R$ 1.320,01 a R$ 1.980,00',
    'D': 'De R$ 1.980,01 a R$ 2.640,00', 'E': 'De R$ 2.640,01 a R$ 3.300,00',
    'F': 'De R$ 3.300,01 a R$ 3.960,00', 'G': 'De R$ 3.960,01 a R$ 5.280,00',
    'H': 'De R$ 5.280,01 a R$ 6.600,00', 'I': 'De R$ 6.600,01 a R$ 7.920,00',
    'J': 'De R$ 7.920,01 a R$ 9.240,00', 'K': 'De R$ 9.240,01 a R$ 10.560,00',
    'L': 'De R$ 10.560,01 a R$ 11.880,00', 'M': 'De R$ 11.880,01 a R$ 13.200,00',
    'N': 'De R$ 13.200,01 a R$ 15.840,00', 'O': 'De R$ 15.840,01 a R$ 19.800,00',
    'P': 'Mais de R$ 19.800,00', 'Q': 'Não Declarado'
}

escolaridade_map = {
    'A': 'A - Nenhuma/Incompleto', 'B': 'B - Fund. Completo',
    'C': 'C - Médio Incompleto', 'D': 'D - Médio Completo',
    'E': 'E - Superior Incompleto', 'F': 'F - Superior Completo',
    'G': 'G - Pós-Graduação', 'H': 'H - Não Sabe'
}

# -------------------- Função de carregamento (agregado municipal, cap 50k padrão) --------------------
@st.cache_data(show_spinner="Conectando e carregando amostra do ENEM 2024 (agregado municipal)...")
def load_data(sample_size=50000, randomize=False, quick_check_rows=2000):
    connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    engine = None
    try:
        engine = create_engine(connection_string, pool_pre_ping=True)
        inspector = inspect(engine)

        needed_tables = [
            'ed_enem_2024_participantes',
            'ed_enem_2024_resultados_amos_per',
            'municipio'
        ]

        missing = [t for t in needed_tables if not inspector.has_table(t)]
        if missing:
            st.error(f"Tabelas faltando no banco: {missing}. Verifique nomes/permissões.")
            return pd.DataFrame()

        order_clause = "ORDER BY RANDOM()" if randomize else ""
        # aplica limite máximo para proteger o servidor (cap em 500.000)
        if sample_size is not None:
            sample_size = int(min(sample_size, 500000))
            limit_clause = f"LIMIT {sample_size}"
        else:
            limit_clause = ""

        # Consulta: traz participantes e as médias agregadas de resultados por município
        query = f"""
            SELECT
                p.nu_inscricao,
                p.q001                       AS escolaridade_pai,
                p.q002                       AS escolaridade_mae,
                p.q005                       AS faixa_renda,
                p.tp_sexo                    AS sexo,
                p.tp_cor_raca                AS cor_raca,
                p.idade_calculada            AS idade,
                p.co_municipio_prova,
                p.sg_uf_prova                AS uf,
                p.regiao_nome_prova          AS regiao,
                m.nome_municipio,
                r_agg.nota_cn_media_mun,
                r_agg.nota_ch_media_mun,
                r_agg.nota_lc_media_mun,
                r_agg.nota_mt_media_mun,
                r_agg.nota_redacao_media_mun,
                r_agg.nota_media_5_notas_media_mun,
                r_agg.resultados_count_mun
            FROM ed_enem_2024_participantes p
            LEFT JOIN (
                SELECT
                    co_municipio_prova,
                    AVG(nota_cn_ciencias_da_natureza)                AS nota_cn_media_mun,
                    AVG(nota_ch_ciencias_humanas)                    AS nota_ch_media_mun,
                    AVG(nota_lc_linguagens_e_codigos)                AS nota_lc_media_mun,
                    AVG(nota_mt_matematica)                          AS nota_mt_media_mun,
                    AVG(nota_redacao)                                AS nota_redacao_media_mun,
                    AVG(nota_media_5_notas)                          AS nota_media_5_notas_media_mun,
                    COUNT(*)                                         AS resultados_count_mun
                FROM ed_enem_2024_resultados_amos_per
                GROUP BY co_municipio_prova
            ) r_agg
              ON r_agg.co_municipio_prova = p.co_municipio_prova
            LEFT JOIN municipio m
              ON p.co_municipio_prova = m.codigo_municipio_dv
            WHERE p.q001 IS NOT NULL
              AND p.q002 IS NOT NULL
              AND p.q005 IS NOT NULL
            {order_clause}
            {limit_clause};
        """

        df = pd.read_sql(query, engine)
        if df is not None and len(df) > 0:
            return df

        # caso a query principal retorne vazia, tenta fallback por amostras
        p_sample = pd.read_sql(f"SELECT nu_inscricao, q001, q002, q005, tp_sexo, tp_cor_raca, idade_calculada, co_municipio_prova, sg_uf_prova, regiao_nome_prova FROM ed_enem_2024_participantes LIMIT {quick_check_rows};", engine)
        r_sample = pd.read_sql(f"SELECT co_municipio_prova, nota_cn_ciencias_da_natureza, nota_ch_ciencias_humanas, nota_lc_linguagens_e_codigos, nota_mt_matematica, nota_redacao, nota_media_5_notas FROM ed_enem_2024_resultados_amos_per LIMIT {quick_check_rows};", engine)

        if p_sample.empty and r_sample.empty:
            return pd.DataFrame()

        if not r_sample.empty:
            # agrega resultados por município
            r_sample['co_municipio_prova'] = r_sample['co_municipio_prova'].astype(str).str.strip()
            r_agg = r_sample.groupby('co_municipio_prova').agg({
                'nota_cn_ciencias_da_natureza': 'mean',
                'nota_ch_ciencias_humanas': 'mean',
                'nota_lc_linguagens_e_codigos': 'mean',
                'nota_mt_matematica': 'mean',
                'nota_redacao': 'mean',
                'nota_media_5_notas': 'mean'
            }).reset_index().rename(columns={
                'nota_cn_ciencias_da_natureza': 'nota_cn_media_mun',
                'nota_ch_ciencias_humanas': 'nota_ch_media_mun',
                'nota_lc_linguagens_e_codigos': 'nota_lc_media_mun',
                'nota_mt_matematica': 'nota_mt_media_mun',
                'nota_redacao': 'nota_redacao_media_mun',
                'nota_media_5_notas': 'nota_media_5_notas_media_mun'
            })
        else:
            r_agg = pd.DataFrame()

        if not p_sample.empty and not r_agg.empty:
            p_sample['co_municipio_prova'] = p_sample['co_municipio_prova'].astype(str).str.strip()
            merged = p_sample.merge(r_agg, on='co_municipio_prova', how='left')
            return merged

        if not p_sample.empty:
            return p_sample

        if not r_sample.empty:
            return r_sample

        return pd.DataFrame()

    except SQLAlchemyError:
        st.error("Erro ao consultar o banco de dados.")
        return pd.DataFrame()

    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass


# -------------------- Decodificadores e pós-processamento --------------------
def decode_enem_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # renomeia colunas de fallback quando necessário
    col_map = {}
    if 'q005' in df.columns and 'faixa_renda' not in df.columns:
        col_map['q005'] = 'faixa_renda'
    if 'q001' in df.columns and 'escolaridade_pai' not in df.columns:
        col_map['q001'] = 'escolaridade_pai'
    if 'q002' in df.columns and 'escolaridade_mae' not in df.columns:
        col_map['q002'] = 'escolaridade_mae'
    if 'tp_sexo' in df.columns and 'sexo' not in df.columns:
        col_map['tp_sexo'] = 'sexo'
    if 'tp_cor_raca' in df.columns and 'cor_raca' not in df.columns:
        col_map['tp_cor_raca'] = 'cor_raca'
    if 'idade_calculada' in df.columns and 'idade' not in df.columns:
        col_map['idade_calculada'] = 'idade'
    if 'sg_uf_prova' in df.columns and 'uf' not in df.columns:
        col_map['sg_uf_prova'] = 'uf'
    if 'regiao_nome_prova' in df.columns and 'regiao' not in df.columns:
        col_map['regiao_nome_prova'] = 'regiao'
    if col_map:
        df.rename(columns=col_map, inplace=True)

    # garante colunas essenciais
    essential_cols = ['faixa_renda', 'sexo', 'escolaridade_pai', 'escolaridade_mae']
    for c in essential_cols:
        if c not in df.columns:
            df[c] = np.nan

    df['faixa_renda_legivel'] = df['faixa_renda'].map(Q005_MAP) if 'faixa_renda' in df.columns else pd.Series(['Desconhecido'] * len(df))
    df['faixa_renda_legivel'] = df['faixa_renda_legivel'].fillna('Desconhecido')

    if 'sexo' in df.columns:
        df['sexo'] = df['sexo'].replace({'M': 'Masculino', 'F': 'Feminino'})
        df['sexo'] = df['sexo'].fillna('Desconhecido')
    else:
        df['sexo'] = 'Desconhecido'

    if 'escolaridade_pai' in df.columns:
        df['escolaridade_pai'] = df['escolaridade_pai'].replace(escolaridade_map)
        df['escolaridade_pai'] = df['escolaridade_pai'].fillna('Desconhecido')
    else:
        df['escolaridade_pai'] = 'Desconhecido'

    if 'escolaridade_mae' in df.columns:
        df['escolaridade_mae'] = df['escolaridade_mae'].replace(escolaridade_map)
        df['escolaridade_mae'] = df['escolaridade_mae'].fillna('Desconhecido')
    else:
        df['escolaridade_mae'] = 'Desconhecido'

    # garante colunas individuais de nota existam
    notes_cols = [
        'nota_cn_ciencias_da_natureza', 'nota_ch_ciencias_humanas',
        'nota_lc_linguagens_e_codigos', 'nota_mt_matematica', 'nota_redacao',
        'nota_media_5_notas'
    ]
    for nc in notes_cols:
        if nc not in df.columns:
            df[nc] = np.nan

    # Se existirem colunas municipais agregadas, preenche NA das notas individuais
    nota_cols_ind = notes_cols
    nota_cols_mun = [
        'nota_cn_media_mun', 'nota_ch_media_mun', 'nota_lc_media_mun',
        'nota_mt_media_mun', 'nota_redacao_media_mun', 'nota_media_5_notas_media_mun'
    ]
    for ind, mun in zip(nota_cols_ind, nota_cols_mun):
        if mun in df.columns:
            df[ind] = df[ind].fillna(df[mun])

    return df


# -------------------- Novas visualizações para casos com muitos 'Desconhecido' --------------------

def create_declaration_vs_score_scatter(df, min_records=10, top_n=30):
    # Para municípios com pelo menos min_records, plota taxa de declaração (x) vs nota média (y)
    grp = df.groupby('nome_municipio').agg(
        declarados=('faixa_renda_legivel', lambda s: (s != 'Desconhecido').sum()),
        total=('faixa_renda_legivel', 'count'),
        nota_media=('nota_media_5_notas', 'mean')
    ).reset_index()
    grp['tx_declaracao'] = grp['declarados'] / grp['total']
    grp = grp[grp['total'] >= min_records]
    if grp.empty:
        return px.bar(title='Nenhum município com registros suficientes para análise de declaração de renda')
    top = grp.sort_values('total', ascending=False).head(top_n)
    fig = px.scatter(top, x='tx_declaracao', y='nota_media', size='total', hover_name='nome_municipio',
                     title=f'Taxa de declaração de renda vs Nota Média (municípios com ≥{min_records} registros)')
    fig.update_xaxes(tickformat='.0%')
    fig.update_layout(margin=dict(t=40))
    return fig


def create_top_mun_declared_bar(df, top_n=10):
    df_decl = df[df['faixa_renda_legivel'] != 'Desconhecido'].copy()
    if df_decl.empty:
        return px.bar(title='Nenhum município com renda declarada nesta seleção')
    grp = df_decl.groupby('nome_municipio').agg(
        declarados=('faixa_renda_legivel', 'count'),
        nota_media=('nota_media_5_notas', 'mean')
    ).reset_index()
    top = grp.sort_values('declarados', ascending=False).head(top_n)
    fig = px.bar(top, x='nome_municipio', y='declarados', hover_data=['nota_media'], title=f'Top {top_n} Municípios por N° de Rendas Declaradas')
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    fig.update_xaxes(tickangle=45)
    fig.update_layout(margin=dict(t=40, b=120))
    return fig


# -------------------- Visualizações e funções analíticas (Ajustadas) --------------------

def create_pie_chart(df, column, title, textfont_size=10):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    if counts.empty:
        return px.pie(title=title)
    palette = px.colors.qualitative.Plotly
    fig = px.pie(counts, names=column, values='count', title=title, hole=0.35, color_discrete_sequence=palette)
    # para categorias com muitos labels (ex: escolaridade), mostra apenas percentuais dentro e legenda à direita
    fig.update_traces(textinfo='percent')
    fig.update_layout(legend=dict(orientation='v', x=1.02, y=0.5, font=dict(size=10)), margin=dict(t=30, b=30), legend_title_text=column)
    return fig


def create_income_bar_chart(df, only_declared=False):
    renda = df['faixa_renda_legivel'].fillna('Desconhecido')
    renda_counts = renda.value_counts().reset_index()
    renda_counts.columns = ['faixa_renda_legivel', 'count']
    renda_counts['pct'] = renda_counts['count'] / renda_counts['count'].sum() * 100
    ordered_categories = [Q005_MAP[k] for k in Q005_MAP.keys() if Q005_MAP[k] in renda_counts['faixa_renda_legivel'].tolist()]

    if only_declared:
        renda_counts = renda_counts[renda_counts['faixa_renda_legivel'] != 'Desconhecido']
        if renda_counts.empty:
            return px.bar(title='Nenhum registro com faixa de renda declarada')

    if len(ordered_categories) <= 1:
        # alternativa: retornamos None e a interface exibirá a visualização substituta
        return None

    renda_counts['order'] = renda_counts['faixa_renda_legivel'].apply(lambda x: ordered_categories.index(x) if x in ordered_categories else 999)
    renda_counts = renda_counts.sort_values('order')

    fig = px.bar(
        renda_counts,
        x='faixa_renda_legivel', y='count',
        title='Distribuição de Renda (Q005) — Faixas legíveis',
        labels={'faixa_renda_legivel': 'Faixa de Renda (Q005)', 'count': 'Quantidade'},
    )
    fig.update_traces(texttemplate='%{y} (%{customdata[0]:.1f}%)', textposition='outside', customdata=renda_counts[['pct']].values)
    fig.update_xaxes(tickangle=45)
    fig.update_layout(margin=dict(t=50, b=120))
    return fig


def create_notes_box_plot(df):
    notes_columns = [
        'nota_cn_ciencias_da_natureza', 'nota_ch_ciencias_humanas',
        'nota_lc_linguagens_e_codigos', 'nota_mt_matematica', 'nota_redacao'
    ]
    df_melt = df[notes_columns].melt(var_name='Área', value_name='Nota')
    fig = px.box(df_melt, y='Nota', color='Área', title='Distribuição das Notas por Área de Conhecimento', points='outliers')
    fig.update_layout(boxmode='group', margin=dict(t=40))
    return fig


def create_income_vs_math_box_plot(df):
    df_plot = df.copy()
    df_plot['faixa_renda_legivel'] = df_plot['faixa_renda_legivel'].fillna('Desconhecido')
    df_no_unknown = df_plot[df_plot['faixa_renda_legivel'] != 'Desconhecido']

    if df_no_unknown['faixa_renda_legivel'].nunique() < 2:
        # substitui por gráfico de municípios: taxa de declaração vs média de matemática
        grp = df_plot.groupby('nome_municipio').agg(
            declarados=('faixa_renda_legivel', lambda s: (s != 'Desconhecido').sum()),
            total=('faixa_renda_legivel', 'count'),
            nota_mt_media=('nota_mt_matematica', 'mean')
        ).reset_index()
        grp['tx_declaracao'] = grp['declarados'] / grp['total']
        top = grp[grp['total'] >= 10].sort_values('tx_declaracao', ascending=False).head(20)
        if top.empty:
            return px.box(df_plot, y='nota_mt_matematica', points='all', title='Nota de Matemática (dados de renda indisponíveis ou agregados)')
        fig = px.scatter(top, x='tx_declaracao', y='nota_mt_media', size='total', hover_name='nome_municipio',
                         title='Taxa de declaração de renda por município vs Média Matemática (municípios com >=10 registros)')
        fig.update_xaxes(tickformat='.0%')
        fig.update_layout(margin=dict(t=40))
        return fig

    ordered_categories = [Q005_MAP[k] for k in Q005_MAP.keys() if Q005_MAP[k] in df_no_unknown['faixa_renda_legivel'].tolist()]
    fig = px.box(
        df_no_unknown,
        x='faixa_renda_legivel',
        y='nota_mt_matematica',
        title='Impacto da Renda na Nota de Matemática (Box Plot)',
        labels={'faixa_renda_legivel': 'Faixa de Renda', 'nota_mt_matematica': 'Nota Matemática'},
        points='all',
        category_orders={'faixa_renda_legivel': ordered_categories}
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_layout(margin=dict(t=50, b=120))
    return fig


def short_label(s: str, maxlen=18) -> str:
    # reduz rótulos como 'A - Nenhuma/Incompleto' -> 'Nenhuma' e trunca se muito longo
    if pd.isna(s):
        return ''
    if ' - ' in s:
        s2 = s.split(' - ', 1)[1]
    else:
        s2 = s
    s2 = s2.strip()
    return (s2 if len(s2) <= maxlen else s2[:maxlen-1] + '…')


def create_parent_education_vs_mean_note(df):
    pivot_mean = df.groupby(['escolaridade_pai', 'escolaridade_mae'])['nota_media_5_notas'].mean().unstack(fill_value=np.nan)
    pivot_count = df.groupby(['escolaridade_pai', 'escolaridade_mae'])['nota_media_5_notas'].count().unstack(fill_value=0)

    if pivot_mean.size == 0:
        return px.imshow([[np.nan]], title='Média das Notas vs. Escolaridade dos Pais (sem dados)')

    x_labels_full = list(pivot_mean.columns)
    y_labels_full = list(pivot_mean.index)

    x_labels = [short_label(x) for x in x_labels_full]
    y_labels = [short_label(y) for y in y_labels_full]

    show_text = pivot_mean.size <= 64
    text_auto = '.2f' if show_text else False
    height = 300 + 30 * max(len(x_labels), len(y_labels))

    fig = px.imshow(pivot_mean.values,
                    x=x_labels,
                    y=y_labels,
                    labels={'x': 'Escolaridade da Mãe', 'y': 'Escolaridade do Pai', 'color': 'Nota Média'},
                    text_auto=text_auto,
                    aspect='auto',
                    title='Média das Notas (5) por Escolaridade dos Pais')

    # axis text menores e rotacionados para caber
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=9))
    fig.update_layout(margin=dict(t=60), height=height)

    hover_text = []
    for i, y in enumerate(y_labels_full):
        row = []
        for j, x in enumerate(x_labels_full):
            mean_val = pivot_mean.loc[y, x]
            cnt = int(pivot_count.loc[y, x]) if x in pivot_count.columns and y in pivot_count.index else 0
            if np.isnan(mean_val):
                row.append(f"Média: n/a
Contagem: {cnt}")
            else:
                row.append(f"Média: {mean_val:.2f}
Contagem: {cnt}")
        hover_text.append(row)

    fig.data[0].hovertemplate = '%{y}<br>%{x}<br>%{customdata}<extra></extra>'
    fig.data[0].customdata = hover_text
    return fig


def perform_predictive_analysis(df: pd.DataFrame):
    target = 'nota_media_5_notas'
    features = ['faixa_renda', 'escolaridade_pai', 'escolaridade_mae', 'cor_raca', 'sexo']
    df_ml = df.dropna(subset=features + [target]).copy()
    df_ml = df_ml[df_ml[target] > 0]
    if len(df_ml) < 100:
        return 0, pd.DataFrame(columns=['Variável', 'Importância', 'Tipo'])
    X = df_ml[features]
    y = df_ml[target]
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
        ],
        remainder='passthrough'
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    r2 = r2_score(y_test, y_pred)
    feature_names_raw = preprocessor.named_transformers_['onehot'].get_feature_names_out(features)
    importance_df = pd.DataFrame({
        'Variável_raw': feature_names_raw,
        'Importância': model.feature_importances_
    })
    def map_tipo(var_code):
        if var_code.startswith('faixa_renda_'):
            return 'Renda'
        if var_code.startswith('escolaridade_pai_'):
            return 'Escolaridade Pai'
        if var_code.startswith('escolaridade_mae_'):
            return 'Escolaridade Mãe'
        if var_code.startswith('cor_raca_'):
            return 'Cor/Raça'
        if var_code.startswith('sexo_'):
            return 'Sexo'
        return 'Outros'
    def get_legible_name(var_code):
        if var_code.startswith('faixa_renda_'):
            key = var_code.split('_')[-1]
            return Q005_MAP.get(key, f'Renda {key}')
        if var_code.startswith('escolaridade_pai_'):
            key = var_code.split('_')[-1]
            return f'Pai: {escolaridade_map.get(key, key)}'
        if var_code.startswith('escolaridade_mae_'):
            key = var_code.split('_')[-1]
            return f'Mãe: {escolaridade_map.get(key, key)}'
        if var_code.startswith('cor_raca_'):
            key = var_code.split('_')[-1]
            return f'Cor/Raça: {key}'
        if var_code.startswith('sexo_'):
            key = var_code.split('_')[-1]
            return f'Sexo: {key}'
        return var_code
    importance_df['Tipo'] = importance_df['Variável_raw'].apply(map_tipo)
    importance_df['Variável'] = importance_df['Variável_raw'].apply(get_legible_name)
    importance_df = importance_df[['Variável', 'Importância', 'Tipo']]
    importance_df = importance_df.sort_values(by='Importância', ascending=False).reset_index(drop=True)
    return r2, importance_df


# -------------------- Interface Streamlit (ajustes pedidos) --------------------

def main():
    st.title("📊 ENEM 2024 - Dashboard Socioeconômico (Agregado Municipal)")
    st.markdown("---")

    # Carrega os dados
    try:
        df_raw = load_data(sample_size=50000)
    except Exception:
        import traceback
        st.error("Erro ao carregar dados: veja o traceback abaixo:")
        st.text(traceback.format_exc())
        st.stop()

    if df_raw.empty:
        st.error("⚠️ Não foi possível carregar dados. Verifique a conexão/configurações do banco.")
        st.stop()

    # Decodifica categorias e preenche notas com médias municipais quando necessário
    df = decode_enem_categories(df_raw)

    # opção no sidebar: mostrar apenas renda declarada para o gráfico de renda
    st.sidebar.header("Filtros do Dashboard")
    show_only_declared_renda = st.sidebar.checkbox("Mostrar apenas registros com renda declarada (Q005)", value=False)

    regioes = ["Todas"] + sorted(df["regiao"].dropna().unique())
    ufs = ["Todas"] + sorted(df["uf"].dropna().unique())
    generos = ["Todos"] + sorted(df["sexo"].dropna().unique())
    rendas_unicas = sorted(df["faixa_renda"].dropna().unique(), key=lambda x: list(Q005_MAP.keys()).index(x) if x in Q005_MAP else 99)
    rendas_legiveis = [Q005_MAP.get(r, r) for r in rendas_unicas]
    rendas_map_rev = {v: k for k, v in Q005_MAP.items()}

    regiao_sel = st.sidebar.selectbox("Região", regioes)
    uf_sel = st.sidebar.selectbox("UF", ufs)
    genero_sel = st.sidebar.selectbox("Sexo", generos)
    renda_sel_legivel = st.sidebar.selectbox("Faixa de Renda", ["Todas"] + rendas_legiveis)

    # Aplica filtros
    df_filtrado = df.copy()
    if regiao_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado["regiao"] == regiao_sel]
    if uf_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado["uf"] == uf_sel]
    if genero_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["sexo"] == genero_sel]
    if renda_sel_legivel != "Todas":
        renda_sel_codigo = rendas_map_rev.get(renda_sel_legivel)
        if renda_sel_codigo:
            df_filtrado = df_filtrado[df_filtrado["faixa_renda"] == renda_sel_codigo]

    # Mantém apenas registros com nota válida
    df_filtrado = df_filtrado.dropna(subset=['nota_media_5_notas'])
    df_filtrado = df_filtrado[df_filtrado['nota_media_5_notas'] > 0]

    st.markdown("---")

    if len(df_filtrado) == 0:
        st.warning("⚠️ Nenhum registro encontrado com os filtros selecionados. Tente expandir sua seleção.")
        st.stop()

    # Métricas rápidas (sem exibir contagens totais explícitas que você pediu para remover)
    st.header("Análise Descritiva Rápida")
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)

    media_geral = df_filtrado['nota_media_5_notas'].mean()
    media_matematica = df_filtrado['nota_mt_matematica'].mean()
    media_redacao = df_filtrado['nota_redacao'].mean()

    with col_met1:
        st.metric(label="Média Geral (5 Notas)", value=f"{media_geral:.2f} pts")
    with col_met2:
        st.metric(label="Média Matemática", value=f"{media_matematica:.2f} pts")
    with col_met3:
        st.metric(label="Média Redação", value=f"{media_redacao:.2f} pts")
    with col_met4:
        st.metric(label="Regiões na seleção", value=f"{df_filtrado['regiao'].nunique()}")

    st.markdown("---")

    # Abas
    tab1, tab2 = st.tabs(["Análise Exploratória", "Análise Preditiva e Relatório"])

    with tab1:
        st.subheader("Análise Socioeconômica e Notas do ENEM")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = create_pie_chart(df_filtrado, "escolaridade_pai", "Escolaridade do Pai", textfont_size=10)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = create_pie_chart(df_filtrado, "escolaridade_mae", "Escolaridade da Mãe", textfont_size=10)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        col3, col4 = st.columns(2)
        with col3:
            # aplica filtro opcional para renda declarada
            fig3 = create_income_bar_chart(df_filtrado, only_declared=show_only_declared_renda)
            pct_unknown = (df_filtrado['faixa_renda_legivel'].fillna('Desconhecido') == 'Desconhecido').mean()
            if fig3 is None or pct_unknown > 0.7:
                st.warning("Mais de 70% dos registros têm faixa de renda 'Desconhecido'. Mostrando alternativas relevantes.")
                # novo tema: taxa de declaração vs nota média por município
                fig_alt = create_declaration_vs_score_scatter(df_filtrado)
                st.plotly_chart(fig_alt, use_container_width=True)
            else:
                st.plotly_chart(fig3, use_container_width=True)
        with col4:
            fig4 = create_notes_box_plot(df_filtrado)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        fig_parent_notes = create_parent_education_vs_mean_note(df_filtrado)
        st.plotly_chart(fig_parent_notes, use_container_width=True)

        st.markdown("---")

        # mostrar apenas o boxplot de renda vs matemática em largura completa (removido gráfico lateral confuso)
        st.subheader('Impacto da Renda na Nota de Matemática')
        fig5 = create_income_vs_math_box_plot(df_filtrado)
        st.plotly_chart(fig5, use_container_width=True)
        st.caption('Nota: quando nota individual não estiver disponível, usamos a média municipal como proxy/contexto. O gráfico mostrado depende da disponibilidade de faixas de renda declaradas.')

        # agora a distribuição por sexo em bloco separado, com contraste melhor
        st.subheader('Distribuição por Sexo')
        fig6 = create_pie_chart(df_filtrado, "sexo", "Distribuição por Sexo", textfont_size=12)
        st.plotly_chart(fig6, use_container_width=True)

        with st.expander("🔍 Detalhamento das Estatísticas"):
            st.subheader("Estatísticas Descritivas das Notas")
            st.dataframe(df_filtrado[[
                "nota_cn_ciencias_da_natureza", "nota_ch_ciencias_humanas",
                "nota_lc_linguagens_e_codigos", "nota_mt_matematica",
                "nota_redacao", "nota_media_5_notas"
            ]].describe().style.format(precision=2), use_container_width=True)

            st.subheader("Distribuição de Idades")
            fig_idade = px.histogram(df_filtrado, x="idade", nbins=30, title="Distribuição de Idades dos Participantes")
            st.plotly_chart(fig_idade, use_container_width=True)

            st.subheader("Distribuição por Cor/Raça")
            fig_cor = create_pie_chart(df_filtrado, "cor_raca", "Distribuição por Cor/Raça", textfont_size=11)
            st.plotly_chart(fig_cor, use_container_width=True)

        st.markdown("---")

        st.subheader("Análise Exploratória Adicional")
        numeric_cols = [
            "nota_cn_ciencias_da_natureza", "nota_ch_ciencias_humanas",
            "nota_lc_linguagens_e_codigos", "nota_mt_matematica",
            "nota_redacao", "nota_media_5_notas", "idade"
        ]
        corr = df_filtrado[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Matriz de Correlação entre Notas e Idade")
        fig_corr.update_layout(height=620, margin=dict(t=40))
        fig_corr.update_xaxes(tickfont=dict(size=9))
        fig_corr.update_yaxes(tickfont=dict(size=9))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Top 10 Municípios por Nota Média")
        top_mun = df_filtrado.groupby('nome_municipio').agg(
            nota_media=('nota_media_5_notas', 'mean'),
            participantes=('nota_media_5_notas', 'count')
        ).reset_index().sort_values(by='nota_media', ascending=False).head(10)
        st.dataframe(top_mun.style.format({'nota_media': '{:.2f}', 'participantes': '{:,}'}), use_container_width=True)

    with tab2:
        st.subheader("🔬 Análise Preditiva: Impacto Socioeconômico na Nota Média")
        st.markdown("""O modelo Random Forest Regressor foi treinado para prever a Nota Média usando Renda, Escolaridade dos Pais, Cor/Raça e Sexo.

ATENÇÃO: quando a nota individual do participante não existir, utilizamos a média municipal como proxy/contexto (é uma aproximação).""")
        r2, importance_df = perform_predictive_analysis(df_filtrado)
        if r2 == 0:
            st.warning("⚠️ Dados insuficientes (menos de 100 registros) para treinar o modelo preditivo com os filtros atuais.")
        else:
            col_r2, col_samples = st.columns(2)
            with col_r2:
                st.metric(label="Coeficiente de Determinação ($R^2$)", value=f"{r2:.4f}")
            with col_samples:
                st.metric(label="Amostra para Predição", value=f"{len(df_filtrado):,}")

            st.markdown("### Importância das Variáveis")
            fig_importance = px.bar(
                importance_df.head(20),
                x='Importância',
                y='Variável',
                color='Tipo',
                orientation='h',
                title='Top 20 Fatores Socioeconômicos Mais Importantes (Random Forest)',
                height=600
            )
            st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("---")
    with st.expander("📄 Ver Dados Brutos Filtrados"):
        st.dataframe(df_filtrado, use_container_width=True)

    st.caption("Dashboard ENEM 2024 - Agregado Municipal. Dados: PostgreSQL.")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        st.error("Erro ao executar o aplicativo. Trace abaixo:")
        st.text(traceback.format_exc())
        print(traceback.format_exc())
