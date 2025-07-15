import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Análise Simples de Investimentos")

ativos_str = st.text_input(
    "Digite os tickers da bolsa separados por vírgula (ex: PETR4.SA, ITUB3.SA, B3SA3.SA)",
    value=""
)

if ativos_str:
    tickers = [t.strip().upper() for t in ativos_str.split(",") if t.strip()]
    st.write(f"Tickers para baixar: {tickers}")

    precos = {}
    for t in tickers:
        dados = yf.download(t, period="1y", progress=False)
        if not dados.empty:
            precos[t] = dados["Close"]
        else:
            st.warning(f"Ticker '{t}' não encontrado ou inválido.")

    if precos:
        if len(precos) == 1:
            serie = list(precos.values())[0]
            st.subheader(f"Preço Ajustado de {list(precos.keys())[0]} (últimos 6 meses)")
            st.line_chart(serie)

            rentab = (serie.iloc[-1] / serie.iloc[0]) - 1
            st.write(f"Rentabilidade: {rentab:.2%}")

        else:
            df_precos = pd.concat(precos.values(), axis=1)
            df_precos.columns = precos.keys()
            st.subheader("Preço Ajustado dos Ativos (últimos 6 meses)")
            st.line_chart(df_precos)

            rentabilidades = (df_precos.iloc[-1] / df_precos.iloc[0]) - 1
            rentabilidades = rentabilidades.sort_values(ascending=False)

            st.subheader("Rentabilidade dos Ativos (%)")
            st.table(rentabilidades.apply(lambda x: f"{x:.2%}"))

            st.subheader("Gráfico de barras das rentabilidades")
            st.bar_chart(rentabilidades)
    else:
        st.error("Nenhum dado válido foi carregado. Verifique os tickers.")



