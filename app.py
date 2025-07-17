import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import requests
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# Função para buscar o índice de medo e ganância (crypto fear and greed) - alternative.me
@st.cache_data(ttl=3600)
def get_fear_and_greed_index():
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url).json()
        value = int(response['data'][0]['value'])
        return value
    except Exception:
        return None

def plot_fear_greed_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Índice de Medo e Ganância (Crypto)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25], 'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"},
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    return fig

# Período de análise
periodo = st.selectbox(
    "Selecione o período de análise:",
    options=[
        ("1 mês", "1mo"),
        ("3 meses", "3mo"),
        ("6 meses", "6mo"),
        ("1 ano", "1y"),
        ("2 anos", "2y"),
        ("5 anos", "5y"),
    ],
    index=2,
    format_func=lambda x: x[0]
)

# Entrada de ativos com exemplo preenchido
ativos_str = st.text_input(
    "Digite os tickers da bolsa separados por vírgula",
    value="",
    placeholder="Ex: PETR4.SA, ITUB3.SA, AAPL, MSFT"
)

if ativos_str:
    tickers = [t.strip().upper() for t in ativos_str.split(",") if t.strip()]
    st.write(f"Analisando: **{', '.join(tickers)}** no período de {periodo[0]}")

    precos = {}
    for t in tickers:
        try:
            dados = yf.download(t, period=periodo[1], progress=False)
            if not dados.empty:
                precos[t] = dados["Close"]
                st.write(f"✅ Dados de {t} carregados com sucesso.")
            else:
                st.warning(f"⚠️ Ticker '{t}' não retornou dados.")
        except Exception as e:
            st.error(f"❌ Erro ao baixar dados de {t}: {e}")

    if precos:
        df_precos = pd.concat(precos, axis=1)
        if isinstance(df_precos.columns, pd.MultiIndex):
            df_precos.columns = df_precos.columns.droplevel(0)

        # Preencher índice com datas contínuas para evitar intervalos vazios no gráfico
        full_index = pd.date_range(start=df_precos.index.min(), end=df_precos.index.max(), freq='B')
        df_precos = df_precos.reindex(full_index)
        df_precos = df_precos.fillna(method='ffill').fillna(method='bfill')

        # Baixa taxa USD-BRL para o período escolhido
        try:
            usd_brl = yf.download("USDBRL=X", period=periodo[1], progress=False)["Close"]
            if usd_brl.empty:
                usd_brl = None
                st.warning("Não foi possível carregar a taxa de câmbio USDBRL.")
        except Exception as e:
            usd_brl = None
            st.warning(f"Erro ao carregar taxa de câmbio USDBRL: {e}")

        # Converte os ativos internacionais para BRL
        if usd_brl is not None:
            usd_brl_series = usd_brl if isinstance(usd_brl, pd.Series) else usd_brl.iloc[:, 0]
            usd_brl_series = usd_brl_series.reindex(df_precos.index).fillna(method="ffill").fillna(method="bfill")

            for t in df_precos.columns:
                # Ignorar ativos que terminam com .SA (BR) e que terminam com '=X' (câmbios)
                if not t.endswith(".SA") and not t.endswith("=X"):
                    df_precos[t] = df_precos[t] * usd_brl_series

            st.info("Ativos internacionais convertidos para BRL usando taxa USDBRL.")

        aba = st.radio("Escolha a aba:", ["Análise de Preços", "Previsão com ARIMA"])

        if aba == "Análise de Preços":
            st.subheader(f"📈 Gráfico Interativo de Preços Ajustados ({periodo[0]})")

            fig = go.Figure()
            for t in df_precos.columns:
                serie = df_precos[t]
                retorno_pct = (serie / serie.iloc[0] - 1) * 100
                hover_text = [f"<b>{t}</b><br>Data: {d.date()}<br>Preço: R$ {p:.2f}<br>Variação: {v:.2f}%" for d, p, v in zip(serie.index, serie.values, retorno_pct)]

                fig.add_trace(go.Scatter(
                    x=serie.index,
                    y=serie.values,
                    mode='lines',
                    name=t,
                    text=hover_text,
                    hoverinfo='text'
                ))

            fig.update_layout(
                xaxis_title="Data",
                yaxis_title="Preço Ajustado (R$)",
                template="plotly_white",
                hovermode="x unified",
                legend_title_text="Ativos",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📊 Métricas Financeiras Básicas")

                # Benchmark para cálculo de alpha e beta: usar o primeiro ativo da lista
                benchmark_name = tickers[0]
                benchmark = df_precos[benchmark_name].dropna() if benchmark_name in df_precos.columns else None

                metrics = {}
                for t in df_precos.columns:
                    serie = df_precos[t].dropna()
                    retornos_diarios = serie.pct_change().dropna()
                    retorno_total = (serie.iloc[-1] / serie.iloc[0]) - 1
                    retorno_medio_ano = retornos_diarios.mean() * 252
                    volatilidade_ano = retornos_diarios.std() * np.sqrt(252)
                    sharpe = retorno_medio_ano / volatilidade_ano if volatilidade_ano != 0 else np.nan
                    max_drawdown = ((serie / serie.cummax()) - 1).min()

                    # Calcular alpha e beta mesmo para internacionais, usando merge de retornos alinhados
                    if benchmark is not None:
                        benchmark_retornos = benchmark.pct_change()
                        ativo_retornos = serie.pct_change()
                        df_merge = pd.concat([benchmark_retornos, ativo_retornos], axis=1).dropna()
                        df_merge.columns = ['benchmark', 'ativo']

                        if len(df_merge) > 1:
                            X = df_merge[['benchmark']].values
                            y = df_merge['ativo'].values
                            modelo = LinearRegression().fit(X, y)
                            alpha = modelo.intercept_
                            beta = modelo.coef_[0]
                        else:
                            alpha = beta = np.nan
                    else:
                        alpha = beta = np.nan

                    metrics[t] = {
                        "Retorno Total (%)": f"{retorno_total:.2%}",
                        "Volatilidade Anualizada (%)": f"{volatilidade_ano:.2%}",
                        "Sharpe": f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A",
                        "Max Drawdown": f"{max_drawdown:.2%}",
                        "Alpha": f"{alpha:.4f}" if not np.isnan(alpha) else "N/A",
                        "Beta": f"{beta:.4f}" if not np.isnan(beta) else "N/A",
                    }

                df_metrics = pd.DataFrame(metrics).T

                # Explicações das métricas básicas
                st.markdown("""
                **Explicações das Métricas Básicas:**  
                - **Retorno Total (%)**: Diferença percentual entre o preço final e inicial do ativo.  
                - **Sharpe**: Índice que mede o retorno ajustado pelo risco (quanto maior, melhor).  
                """)

                df_basicas = df_metrics[["Retorno Total (%)", "Sharpe"]]
                st.dataframe(df_basicas)

                st.markdown("---")
                st.subheader("📊 Métricas Financeiras Avançadas")

                # Explicações das métricas avançadas
                st.markdown("""
                **Explicações das Métricas Avançadas:**  
                - **Volatilidade Anualizada (%)**: Medida da variação dos retornos diária, anualizada; indica risco do ativo.  
                - **Max Drawdown**: Maior queda percentual do pico máximo até o ponto mais baixo no período.  
                - **Alpha**: Excesso de retorno do ativo em relação ao benchmark, indicando habilidade do gestor.  
                - **Beta**: Sensibilidade do ativo em relação ao benchmark; risco sistemático.  
                """)

                df_avancadas = df_metrics[["Volatilidade Anualizada (%)", "Max Drawdown", "Alpha", "Beta"]]
                st.dataframe(df_avancadas)

            with col2:
                st.subheader("🧮 Simulador de Carteira")
                st.write("Informe quantidades e preços médios para calcular retorno.")

                carteira = {}
                for t in tickers:
                    qtd = st.number_input(f"Quantidade de {t}:", min_value=0, step=1, key=f"qtd_{t}")
                    preco_medio = st.number_input(f"Preço médio de compra de {t} (R$):", min_value=0.0, format="%.2f", key=f"pm_{t}")
                    carteira[t] = {"quantidade": qtd, "preco_medio": preco_medio}

                valor_total = 0.0
                valor_investido = 0.0
                resultado = []

                for t in tickers:
                    qtd = carteira[t]["quantidade"]
                    pm = carteira[t]["preco_medio"]
                    serie = df_precos[t].dropna()
                    preco_atual = float(serie.iloc[-1]) if not serie.empty else np.nan

                    valor_posicao = qtd * preco_atual
                    investimento = qtd * pm
                    lucro_prejuizo = valor_posicao - investimento

                    if isinstance(valor_posicao, (int, float, np.floating)) and not np.isnan(valor_posicao):
                        valor_total += valor_posicao
                    else:
                        try:
                            valor_total += float(valor_posicao)
                        except:
                            pass

                    if isinstance(investimento, (int, float, np.floating)) and not np.isnan(investimento):
                        valor_investido += investimento
                    else:
                        try:
                            valor_investido += float(investimento)
                        except:
                            pass

                    resultado.append({
                        "Ativo": t,
                        "Quantidade": qtd,
                        "Preço Médio (R$)": pm,
                        "Preço Atual (R$)": preco_atual,
                        "Valor Posição (R$)": valor_posicao,
                        "Investimento (R$)": investimento,
                        "Lucro/Prejuízo (R$)": lucro_prejuizo,
                    })

                df_resultado = pd.DataFrame(resultado)
                df_resultado["Lucro/Prejuízo (%)"] = (df_resultado["Lucro/Prejuízo (R$)"] / df_resultado["Investimento (R$)"]).fillna(0)

                st.write(f"**Valor total da carteira:** R$ {valor_total:,.2f}")
                st.write(f"**Valor total investido:** R$ {valor_investido:,.2f}")
                st.write(f"**Retorno total da carteira:** {(valor_total / valor_investido - 1) if valor_investido != 0 else 0:.2%}")

                st.dataframe(df_resultado.style.format({
                    "Preço Médio (R$)": "R$ {:,.2f}",
                    "Preço Atual (R$)": "R$ {:,.2f}",
                    "Valor Posição (R$)": "R$ {:,.2f}",
                    "Investimento (R$)": "R$ {:,.2f}",
                    "Lucro/Prejuízo (R$)": "R$ {:,.2f}",
                    "Lucro/Prejuízo (%)": "{:.2%}",
                }))

                # Mostrar índice de medo e ganância crypto (alternative.me)
                fg_value = get_fear_and_greed_index()
                if fg_value is not None:
                    st.subheader("🚀 Índice de Medo e Ganância (Crypto)")
                    st.plotly_chart(plot_fear_greed_gauge(fg_value), use_container_width=True)
                else:
                    st.warning("Não foi possível obter o índice de medo e ganância (Crypto).")

        elif aba == "Previsão com ARIMA":
            st.subheader("📅 Previsão com ARIMA para um ativo selecionado")

            ativo_selecionado = st.selectbox("Escolha o ativo para previsão:", df_precos.columns)

            serie_previsao = df_precos[ativo_selecionado].dropna()

            if len(serie_previsao) < 30:
                st.warning("Dados insuficientes para realizar previsão confiável (menos de 30 pontos).")
            else:
                try:
                    model = ARIMA(serie_previsao, order=(2, 1, 2))
                    model_fit = model.fit()
                    previsao = model_fit.get_forecast(steps=365)
                    previsao_df = previsao.summary_frame()

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=serie_previsao.index,
                        y=serie_previsao.values,
                        mode='lines',
                        name='Histórico'
                    ))

                    fig.add_trace(go.Scatter(
                        x=previsao_df.index,
                        y=previsao_df['mean'],
                        mode='lines',
                        name='Previsão'
                    ))

                    fig.add_trace(go.Scatter(
                        x=previsao_df.index,
                        y=previsao_df['mean_ci_lower'],
                        mode='lines',
                        name='Limite Inferior (95%)',
                        line=dict(dash='dash'),
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=previsao_df.index,
                        y=previsao_df['mean_ci_upper'],
                        mode='lines',
                        name='Limite Superior (95%)',
                        line=dict(dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(0,100,80,0.2)',
                        showlegend=False
                    ))

                    fig.update_layout(
                        xaxis_title="Data",
                        yaxis_title="Preço Ajustado (R$)",
                        template="plotly_white",
                        hovermode="x unified",
                        legend_title_text="Ativo",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao calcular previsão ARIMA: {e}")
