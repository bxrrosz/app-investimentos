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

# Função para buscar o índice de medo e ganância
@st.cache_data(ttl=3600)
def get_fear_and_greed_index():
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url).json()
        value = int(response['data'][0]['value'])
        return value
    except:
        return None

def plot_fear_greed_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Índice de Medo e Ganância"},
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
    value="PETR4.SA, ITUB3.SA, AAPL, MSFT",
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

        full_index = pd.date_range(start=df_precos.index.min(), end=df_precos.index.max(), freq='B')
        df_precos = df_precos.reindex(full_index)
        df_precos = df_precos.fillna(method='ffill').fillna(method='bfill')

        try:
            usd_brl = yf.download("USDBRL=X", period=periodo[1], progress=False)["Close"]
            if usd_brl.empty:
                usd_brl = None
                st.warning("Não foi possível carregar a taxa de câmbio USDBRL.")
        except Exception as e:
            usd_brl = None
            st.warning(f"Erro ao carregar taxa de câmbio USDBRL: {e}")

        if usd_brl is not None:
            usd_brl_series = usd_brl if isinstance(usd_brl, pd.Series) else usd_brl.iloc[:, 0]
            usd_brl_series = usd_brl_series.reindex(df_precos.index).fillna(method="ffill").fillna(method="bfill")
            for t in df_precos.columns:
                if not t.endswith(".SA, =X"):
                    df_precos[t] = df_precos[t] * usd_brl_series
            st.info("Ativos internacionais convertidos para BRL usando taxa USDBRL.")

        aba = st.radio("Escolha a aba:", ["Análise de Preços", "Previsão com ARIMA"])

        if aba == "Análise de Preços":
            st.subheader(f"📈 Gráfico Interativo de Preços Ajustados ({periodo[0]})")

            fig = go.Figure()
            for t in df_precos.columns:
                fig.add_trace(go.Scatter(
                    x=df_precos.index,
                    y=df_precos[t],
                    mode='lines',
                    name=t
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
                st.subheader("📊 Métricas Financeiras dos Ativos")
                metrics = {}
                benchmark = df_precos["^BVSP"] if "^BVSP" in df_precos.columns else None

                for t in df_precos.columns:
                    serie = df_precos[t].dropna()
                    retornos_diarios = serie.pct_change().dropna()
                    retorno_total = (serie.iloc[-1] / serie.iloc[0]) - 1
                    retorno_medio_ano = retornos_diarios.mean() * 252
                    volatilidade_ano = retornos_diarios.std() * np.sqrt(252)
                    sharpe = retorno_medio_ano / volatilidade_ano if volatilidade_ano != 0 else np.nan
                    max_drawdown = ((serie / serie.cummax()) - 1).min()

                    if benchmark is not None:
                        benchmark_retornos = benchmark.pct_change().dropna()
                        df_reg = pd.DataFrame({"x": benchmark_retornos, "y": retornos_diarios}).dropna()
                        X = df_reg[["x"]].values
                        y = df_reg["y"].values
                        modelo = LinearRegression().fit(X, y)
                        alpha = modelo.intercept_
                        beta = modelo.coef_[0]
                    else:
                        alpha = beta = np.nan

                    metrics[t] = {
                        "Retorno Total (%)": f"{retorno_total:.2%}",
                        "Volatilidade Anualizada (%)": f"{volatilidade_ano:.2%}",
                        "Sharpe": f"{sharpe:.2f}",
                        "Max Drawdown": f"{max_drawdown:.2%}",
                        "Alpha": f"{alpha:.4f}" if not np.isnan(alpha) else "N/A",
                        "Beta": f"{beta:.4f}" if not np.isnan(beta) else "N/A",
                    }
                st.dataframe(pd.DataFrame(metrics).T)

            with col2:
                valor_fg = get_fear_and_greed_index()
                if valor_fg is not None:
                    st.subheader("🚀 Índice de Medo e Ganância (Crypto)")
                    st.plotly_chart(plot_fear_greed_gauge(valor_fg), use_container_width=True)
                else:
                    st.warning("Não foi possível obter o índice de medo e ganância.")

        elif aba == "Previsão com ARIMA":
            st.subheader("📅 Previsão com ARIMA para um ativo selecionado")

            ativo_selecionado = st.selectbox("Escolha o ativo para previsão:", df_precos.columns)
            serie_previsao = df_precos[ativo_selecionado].dropna()

            if len(serie_previsao) < 30:
                st.warning("Dados insuficientes para previsão confiável (menos de 30 pontos).")
            else:
                try:
                    model = ARIMA(serie_previsao, order=(2, 1, 2))
                    model_fit = model.fit()
                    previsao = model_fit.get_forecast(steps=10)
                    previsao_df = previsao.summary_frame()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=serie_previsao.index, y=serie_previsao.values, name="Histórico"))
                    fig.add_trace(go.Scatter(x=previsao_df.index, y=previsao_df['mean'], name="Previsão"))
                    fig.add_trace(go.Scatter(x=previsao_df.index, y=previsao_df['mean_ci_lower'], name="Limite Inferior", line=dict(dash='dot')))
                    fig.add_trace(go.Scatter(x=previsao_df.index, y=previsao_df['mean_ci_upper'], name="Limite Superior", line=dict(dash='dot'), fill='tonexty', fillcolor='rgba(0,100,80,0.2)'))

                    fig.update_layout(
                        xaxis_title="Data",
                        yaxis_title="Preço Previsto (R$)",
                        template="plotly_white",
                        hovermode="x unified",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao calcular previsão ARIMA: {e}")

