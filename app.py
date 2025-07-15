import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Análise Simples de Investimentos")

ativos_str = st.text_input(
    "Digite os tickers da bolsa separados por vírgula (ex: PETR4.SA, ITUB3.SA, B3SA3.SA)",
    value="")

if ativos_str:
    tickers = [t.strip().upper() for t in ativos_str.split(",") if t.strip()]
    st.write(f"Tickers para baixar: {tickers}")

    try:
        dados = yf.download(tickers, period="6mo", group_by='ticker', progress=False)
        st.write("Dados brutos baixados:")
        st.write(dados.head())

        precos = {}
        for t in tickers:
            if t in dados.columns.levels[0]:
                precos[t] = dados[t]['Close']
            else:
                st.warning(f"Ticker '{t}' não encontrado ou inválido.")
        
        if precos:
            df_precos = pd.concat(precos.values(), axis=1)
            df_precos.columns = precos.keys()
            st.subheader("Preço Ajustado dos Ativos (últimos 6 meses)")
            st.line_chart(df_precos)
            st.write("Dados do gráfico:")
            st.write(df_precos.head())
        else:
            st.error("Nenhum dado válido foi carregado. Verifique os tickers.")
    except Exception as e:
        st.error(f"Erro ao baixar os dados: {e}")



