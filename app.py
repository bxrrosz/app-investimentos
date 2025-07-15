import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

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
        # Cálculo de métricas financeiras
        metrics = {}
        for t, serie in precos.items():
            if (
                not serie.empty
                and pd.api.types.is_numeric_dtype(serie)
                and pd.notna(serie.iloc[0])
                and pd.notna(serie.iloc[-1])
                and serie.iloc[0] != 0
            ):
                retornos_diarios = serie.pct_change().dropna()
                retorno_total = (serie.iloc[-1] / serie.iloc[0]) - 1
                retorno_medio_ano = retornos_diarios.mean() * 252
                volatilidade_ano = retornos_diarios.std() * np.sqrt(252)
                sharpe = retorno_medio_ano / volatilidade_ano if volatilidade_ano != 0 else np.nan

                metrics[t] = {
                    "Retorno Total (%)": f"{retorno_total:.2%}",
                    "Retorno Médio Anualizado (%)": f"{retorno_medio_ano:.2%}",
                    "Volatilidade Anualizada (%)": f"{volatilidade_ano:.2%}",
                    "Índice Sharpe": f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A",
                }
            else:
                metrics[t] = {
                    "Retorno Total (%)": "N/A",
                    "Retorno Médio Anualizado (%)": "N/A",
                    "Volatilidade Anualizada (%)": "N/A",
                    "Índice Sharpe": "N/A",
                }

        st.subheader("📊 Métricas Financeiras dos Ativos")
        df_metrics = pd.DataFrame(metrics).T
        st.table(df_metrics)

        # Exibir gráfico de preços
        if len(precos) == 1:
            serie = list(precos.values())[0]
            st.subheader(f"📈 Preço Ajustado de {list(precos.keys())[0]} ({periodo[0]})")
            st.line_chart(serie)
        else:
            df_precos = pd.DataFrame(precos)
            st.subheader(f"📈 Preço Ajustado dos Ativos ({periodo[0]})")
            st.line_chart(df_precos)

        # Simulador de carteira - fica no final para não "sumir" as métricas
        st.subheader("🧮 Simulador de Carteira")
        st.write("Informe as quantidades e preços médios para calcular valor e retorno da carteira.")

        carteira = {}
        for t in tickers:
            qtd = st.number_input(f"Quantidade de {t}:", min_value=0, step=1, key=f"qtd_{t}")
            preco_medio = st.number_input(f"Preço médio de compra de {t} (R$):", min_value=0.0, format="%.2f", key=f"pm_{t}")

            carteira[t] = {"quantidade": qtd, "preco_medio": preco_medio}

        if st.button("Calcular Resultado da Carteira"):
            valor_total = 0
            valor_investido = 0
            resultado = []

            for t in tickers:
                qtd = carteira[t]["quantidade"]
                pm = carteira[t]["preco_medio"]
                preco_atual = precos[t].iloc[-1]

                valor_posicao = qtd * preco_atual
                investimento = qtd * pm
                lucro_prejuizo = valor_posicao - investimento

                valor_total += valor_posicao
                valor_investido += investimento

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







