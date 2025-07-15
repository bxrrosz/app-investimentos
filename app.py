import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Análise Simples de Investimentos 📈")

ativos = st.text_input("Digite os tickers (ex: PETR4.SA,VALE3.SA)", "PETR4.SA,VALE3.SA")

if ativos:
    lista = [ticker.strip().upper() for ticker in ativos.split(",")]
    dados = yf.download(lista, period="6mo")["Close"]
    st.line_chart(dados)

import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Análise Simples de Investimentos 📈")

uploaded_file = st.file_uploader("Faça upload do arquivo CSV da sua carteira", type="csv")

if uploaded_file:
    carteira = pd.read_csv(uploaded_file)
    st.write("Sua carteira:")
    st.dataframe(carteira)

    tickers = carteira['Ticker'].str.upper().tolist()

    precos = {}
    for t in tickers:
        dados_ticker = yf.download(t, period="6mo")
        if not dados_ticker.empty:
            precos[t] = dados_ticker["Adj Close"]
        else:
            st.warning(f"Ticker '{t}' não encontrado ― confira o código.")

    if precos:
        df_precos = pd.DataFrame(precos)
        st.subheader("Preço ajustado (últimos 6 meses)")
        st.line_chart(df_precos)
