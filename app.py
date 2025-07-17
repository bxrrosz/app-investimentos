import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import requests
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="App de Investimentos")

# Função para índice de medo e ganância
@st.cache_data(ttl=3600)
def get_fear_and_greed():
    try:
        resp = requests.get("https://api.alternative.me/fng/")
        val = int(resp.json()["data"][0]["value"])
        txt = resp.json()["data"][0]["value_classification"]
        return val, txt
    except:
        return None, None

st.title("💼 Aplicativo de Análise de Investimentos")

# Sidebar
st.sidebar.header("Configuração da Carteira")
ativos = st.sidebar.text_input("Digite os ativos (ex: PETR4.SA, VALE3.SA, AAPL):", value="")
pesos = st.sidebar.text_input("Pesos (%):", value="50, 50")
data_inicio = st.sidebar.date_input("Data de início", value=datetime.date(2022, 1, 1))

ativos_lista = [a.strip().upper() for a in nativos.split(",") if a.strip()]
pesos_lista = [float(p) for p in pesos.split(",") if p.strip()]

if len(ativos_lista) != len(pesos_lista):
    st.error("O número de ativos e pesos precisa ser igual.")
    st.stop()

# Dados de preço
dados = yf.download(ativos_lista, start=data_inicio)["Close"].dropna()
retornos = dados.pct_change().dropna()

# Normaliza para plot
precos_normalizados = dados / dados.iloc[0]
st.subheader("📈 Retorno dos Ativos")
st.line_chart(precos_normalizados)

# Análise Avançada
st.subheader("Análise Avançada da Carteira")
vol = retornos.std() * np.sqrt(252)
sharpe = retornos.mean() * 252 / vol

ibov = yf.download("^BVSP", start=data_inicio)["Close"].pct_change().reindex(retornos.index).dropna()
alphas, betas = [], []
for ativo in retornos.columns:
    X = ibov.values.reshape(-1, 1)
    y = retornos[ativo].reindex(ibov.index).dropna()
    X = X[-len(y):]
    reg = LinearRegression().fit(X, y)
    alphas.append(reg.intercept_)
    betas.append(reg.coef_[0])

drawdowns = (retornos.cumsum() - retornos.cumsum().cummax()).min()

metrics_df = pd.DataFrame({
    "Volatilidade (%)": (vol * 100).round(2),
    "Sharpe": sharpe.round(2),
    "Alpha": np.round(alphas, 5),
    "Beta": np.round(betas, 5),
    "Max Drawdown (%)": (drawdowns * 100).round(2)
})

st.dataframe(metrics_df)

# Composição da Carteira
st.subheader("💲 Composição da Carteira")
carteira_valores = {}
for i, a in enumerate(ativos_lista):
    carteira_valores[a] = {
        "peso": pesos_lista[i],
        "preco_atual": dados[a].iloc[-1],
        "valor": pesos_lista[i] * dados[a].iloc[-1]
    }

carteira_df = pd.DataFrame(carteira_valores).T
carteira_df["valor"] = carteira_df["valor"].round(2)
st.dataframe(carteira_df)

# Fear & Greed
st.subheader("😨 Índice de Medo & Ganância")
valor_fear, texto_fear = get_fear_and_greed()
if valor_fear is not None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor_fear,
        title={"text": texto_fear},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "black"},
            "steps": [
                {"range": [0, 25], "color": "red"},
                {"range": [25, 50], "color": "orange"},
                {"range": [50, 75], "color": "lightgreen"},
                {"range": [75, 100], "color": "green"}
            ]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Não foi possível carregar o índice de medo e ganância.")
