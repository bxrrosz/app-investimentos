import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")

st.markdown("# 💰 Analisador Simples de Investimentos")

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

ativos_str = st.text_input(
    "Digite os tickers da bolsa separados por vírgula",
    placeholder="Ex: PETR4.SA, ITUB3.SA, B3SA3.SA"
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
        if len(precos) == 1:
            serie = list(precos.values())[0]
            st.subheader(f"📈 Preço Ajustado de {list(precos.keys())[0]} ({periodo[0]})")
            st.line_chart(serie)

            if (not serie.empty and
                not np.isnan(serie.iloc[0]) and
                not np.isnan(serie.iloc[-1]) and
                serie.iloc[0] != 0):
                
                rentab = (serie.iloc[-1] / serie.iloc[0]) - 1
                st.markdown(f"**Rentabilidade:** {rentab:.2%}")
            else:
                st.warning("Não foi possível calcular a rentabilidade devido a dados insuficientes ou inválidos.")

        else:
            df_precos = pd.concat(precos.values(), axis=1)
            df_precos.columns = precos.keys()
            st.subheader(f"📈 Preço Ajustado dos Ativos ({periodo[0]})")
            st.line_chart(df_precos)

            rentabilidades = (df_precos.iloc[-1] / df_precos.iloc[0]) - 1
            rentabilidades = rentabilidades.sort_values(ascending=False)

            st.subheader("📊 Rentabilidade dos Ativos (%)")
            st.table(rentabilidades.apply(lambda x: f"{x:.2%}"))

            df_rent = rentabilidades.reset_index()
            df_rent.columns = ['Ativo', 'Rentabilidade']

            chart = alt.Chart(df_rent).mark_bar().encode(
                x=alt.X('Ativo', sort='-y', title='Ativo'),
                y=alt.Y('Rentabilidade', title='Rentabilidade (%)'),
                color=alt.condition(
                    alt.datum.Rentabilidade > 0,
                    alt.value('green'),
                    alt.value('red')
                ),
                tooltip=[alt.Tooltip('Ativo'), alt.Tooltip('Rentabilidade', format='.2%')]
            ).properties(width=700, height=400, title=f"Rentabilidade dos Ativos ({periodo[0]})")

            st.altair_chart(chart, use_container_width=True)
    else:
        st.error("Nenhum dado válido foi carregado. Verifique os tickers.")





