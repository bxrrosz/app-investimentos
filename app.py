import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")

st.markdown("# 💰 Analisador Simples de Investimentos")

# Seleção do período
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
    index=2,  # padrão 6 meses
    format_func=lambda x: x[0]
)

ativos_str = st.text_input(
    "Digite os tickers da bolsa separados por vírgula",
    placeholder="Ex: PETR4.SA, ITUB3.SA, B3SA3.SA"
)

if ativos_str:
    tickers = [t.strip().upper() for t in ativos_str.split(",") if t.strip()]
    st.write(f"Analisando: **{', '.join(tickers)}** no período de {periodo[0]}")

    precos = {}
    for t in tickers:
        dados = yf.download(t, period=periodo[1], progress=False)
        if not dados.empty:
            precos[t] = dados["Close"]
        else:
            st.warning(f"Ticker '{t}' não encontrado ou inválido.")

    if precos:
        if len(precos) == 1:
            serie = list(precos.values())[0]
            st.subheader(f"📈 Preço Ajustado de {list(precos.keys())[0]} ({periodo[0]})")
            st.line_chart(serie)

            rentab = (serie.iloc[-1] / serie.iloc[0]) - 1
            st.markdown(f"**Rentabilidade:** {rentab:.2%}")

        else:
            df_precos = pd.concat(precos.values(), axis=1)




