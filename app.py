import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

st.set_page_config(layout="wide")
st.title("💸 Dashboard de Investimentos")

st.markdown("""
Este aplicativo permite que você acompanhe sua carteira personalizada de ativos, visualize gráficos de desempenho e analise o sentimento do mercado com o índice de Medo e Ganância.
""")

abas = st.tabs(["📊 Minha Carteira", "📉 Análise Avançada", "🧠 Medo & Ganância"])

# =========================== FUNÇÕES ===========================
def calcular_metricas(retornos):
    media = retornos.mean() * 252
    volatilidade = retornos.std() * np.sqrt(252)
    sharpe = media / volatilidade if volatilidade != 0 else np.nan
    retorno_acumulado = (1 + retornos).prod() - 1
    drawdown = (retornos.cumsum() - retornos.cumsum().cummax()).min()
    return {
        "Retorno Acumulado": retorno_acumulado,
        "Volatilidade": volatilidade,
        "Sharpe Ratio": sharpe,
        "Máx. Drawdown": drawdown
    }

def calcular_alpha_beta(df_precos, benchmark_ticker="^BVSP"):
    benchmark = yf.download(benchmark_ticker, start=df_precos.index.min(), end=df_precos.index.max())["Close"]
    benchmark_rets = benchmark.pct_change().dropna()
    resultados = {}
    for t in df_precos.columns:
        rets = df_precos[t].pct_change().dropna()
        df_merged = pd.concat([rets, benchmark_rets], axis=1).dropna()
        X = df_merged.iloc[:, 1].values.reshape(-1, 1)
        y = df_merged.iloc[:, 0].values
        reg = LinearRegression().fit(X, y)
        resultados[t] = {"Alpha": reg.intercept_, "Beta": reg.coef_[0]}
    return pd.DataFrame(resultados).T

# =========================== ABA 1 - MINHA CARTEIRA ===========================
with abas[0]:
    ativos = st.multiselect("Selecione seus ativos", ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "B3SA3.SA", "WEGE3.SA", "MGLU3.SA"], default=["PETR4.SA", "VALE3.SA"])
    data_inicio = st.date_input("Data de Início", value=datetime.date(2023, 1, 1))
    df_precos = yf.download(ativos, start=data_inicio)["Close"]

    if not df_precos.empty:
        df_normalizado = df_precos / df_precos.iloc[0]
        st.line_chart(df_normalizado, use_container_width=True)
        st.dataframe(df_precos.tail(), use_container_width=True)

# =========================== ABA 2 - MÉTRICAS AVANÇADAS ===========================
with abas[1]:
    if not df_precos.empty:
        retornos = df_precos.pct_change().dropna()

        st.subheader("📌 Métricas da Carteira")
        metricas = {ticker: calcular_metricas(retornos[ticker]) for ticker in retornos.columns}
        st.dataframe(pd.DataFrame(metricas).T.style.format("{:.2%}"))

        st.subheader("📐 Alpha e Beta (em relação ao Ibovespa)")
        ab = calcular_alpha_beta(df_precos)
        st.dataframe(ab.style.format("{:.2f}"))

# =========================== ABA 3 - ÍNDICE MEDO E GANÂNCIA ===========================
with abas[2]:
    st.header("😨 Índice de Medo e Ganância (CNN Real-Time)")

    st.markdown("""
    O **Fear & Greed Index** (Índice de Medo e Ganância) é uma medida desenvolvida pela CNN que avalia o sentimento dos investidores no mercado com base em sete indicadores.
    
    **Quanto mais próximo de 0, mais medo domina o mercado. Quanto mais perto de 100, mais ganância.**

    Classificações:
    - 🟥 0 a 24: **Medo Extremo**
    - 🟧 25 a 49: **Medo**
    - 🟩 50 a 74: **Ganância**
    - 🟢 75 a 100: **Ganância Extrema**
    """)

    try:
        url = "https://fear-and-greed-index.p.rapidapi.com/v1/fgi"
        headers = {
            "X-RapidAPI-Key": "SUA_CHAVE_AQUI",  # Substitua por sua chave
            "X-RapidAPI-Host": "fear-and-greed-index.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            fgi = data["fgi"]
            indice_atual = int(fgi["now"]["value"])
            classificacao_atual = fgi["now"]["valueText"]

            historico = {
                "Ontem": (fgi["previousClose"]["value"], fgi["previousClose"]["valueText"]),
                "Semana passada": (fgi["oneWeekAgo"]["value"], fgi["oneWeekAgo"]["valueText"]),
                "Um mês atrás": (fgi["oneMonthAgo"]["value"], fgi["oneMonthAgo"]["valueText"]),
                "Um ano atrás": (fgi["oneYearAgo"]["value"], fgi["oneYearAgo"]["valueText"]),
            }

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=indice_atual,
                title={'text': f"Índice Atual: {classificacao_atual}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 50], 'color': "orange"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🕓 Histórico Recente")
            col1, col2 = st.columns(2)
            with col1:
                for label in ["Ontem", "Semana passada"]:
                    valor, texto = historico[label]
                    st.metric(label=label, value=f"{valor}", help=f"{texto}")
            with col2:
                for label in ["Um mês atrás", "Um ano atrás"]:
                    valor, texto = historico[label]
                    st.metric(label=label, value=f"{valor}", help=f"{texto}")

        else:
            st.error("Erro ao obter índice. Verifique sua chave da RapidAPI.")
    except Exception as e:
        st.error(f"Erro ao buscar o índice: {e}")
