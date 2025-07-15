import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Análise Simples de Investimentos 📈")

ativos_str = st.text_input(
    "Digite os tickers da bolsa separados por vírgula (ex: PETR4.SA, ITUB3.SA, B3SA3.SA)",
    value="PETR4.SA,VALE3.SA",
)

if ativos_str:
    tickers = [t.strip().upper() for t in ativos_str.split(",") if t.strip()]

    precos = {}
    for t in tickers:
        dados = yf.download(t, period="1y")
        if not dados.empty:
            precos[t] = dados["Close"]
        else:
            st.warning(f"Ticker '{t}' não encontrado ou inválido.")

   if precos:
    df_precos = pd.concat(precos.values(), axis=1)
    df_precos.columns = precos.keys()
    st.subheader("Preço Ajustado dos Ativos (últimos 6 meses)")
    st.line_chart(df_precos)

    else:
        st.error("Nenhum dado válido foi carregado. Verifique os tickers.")

