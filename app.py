import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# --- Entrada de dados ---

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
    value="PETR4.SA, ITUB3.SA, AAPL, MSFT",  # Exemplo fixo
    help="Exemplo: PETR4.SA, ITUB3.SA, AAPL, MSFT"
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

        # Dropar nível extra se MultiIndex
        if isinstance(df_precos.columns, pd.MultiIndex):
            df_precos.columns = df_precos.columns.droplevel(0)

        # --- Taxa USD-BRL ---
        try:
            usd_brl = yf.download("USDBRL=X", period=periodo[1], progress=False)["Close"]
            if usd_brl.empty:
                usd_brl = None
                st.warning("Não foi possível carregar a taxa de câmbio USDBRL.")
        except Exception as e:
            usd_brl = None
            st.warning(f"Erro ao carregar taxa de câmbio USDBRL: {e}")

        if usd_brl is not None:
            # Identifica ativos internacionais (que não terminam com .SA)
            ativos_internacionais = [t for t in df_precos.columns if not t.endswith(".SA")]

            # Alinha o índice da taxa de câmbio com df_precos, preenchendo valores faltantes
            usd_brl_alinhado = usd_brl.reindex(df_precos.index).fillna(method="ffill").fillna(method="bfill")

            for t in ativos_internacionais:
                serie_ativo = df_precos[t].reindex(df_precos.index)
                multiplicado = serie_ativo * usd_brl_alinhado
                df_precos[t] = multiplicado.values  # .values para evitar problemas de alinhamento

            st.info("Ativos internacionais convertidos para BRL usando taxa USDBRL.")

        # --- Resolver problema de intervalos no gráfico ---

        # Preenche dados faltantes via interpolação linear para evitar 'buracos' no gráfico
        df_precos_interp = df_precos.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        # --- Gráfico Interativo ---
        st.subheader(f"📈 Gráfico Interativo de Preços Ajustados ({periodo[0]})")

        fig = go.Figure()

        # Traçar preços históricos
        for t in df_precos_interp.columns:
            fig.add_trace(go.Scatter(
                x=df_precos_interp.index,
                y=df_precos_interp[t],
                mode='lines',
                name=t
            ))

        # --- Previsão simples para 30 dias à frente (usando média e std de retornos) ---

        dias_fwd = 30
        last_date = df_precos_interp.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=dias_fwd, freq='B')

        for t in df_precos_interp.columns:
            serie = df_precos_interp[t]
            if len(serie) < 2:
                continue
            retornos = serie.pct_change().dropna()
            media_retorno = retornos.mean()
            std_retorno = retornos.std()

            # Modelo muito simples de forecast: assume retorno médio constante + banda de volatilidade
            ultimo_preco = serie.iloc[-1]
            previsao_media = [ultimo_preco * ((1 + media_retorno) ** i) for i in range(1, dias_fwd + 1)]
            previsao_q90 = [p * (1 + std_retorno) for p in previsao_media]
            previsao_q10 = [p * (1 - std_retorno) for p in previsao_media]

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=previsao_media,
                mode='lines',
                line=dict(dash='dash'),
                name=f"{t} Previsão Média"
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=previsao_q90,
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip',
                fill=None,
                name=f"{t} Banda Superior"
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=previsao_q10,
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip',
                name=f"{t} Banda Inferior"
            ))

        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Preço Ajustado (R$)",
            template="plotly_white",
            hovermode="x unified",
            legend_title_text="Ativos",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Métricas e Simulador de Carteira ---
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Métricas Financeiras dos Ativos")
            metrics = {}
            for t in df_precos.columns:
                serie = df_precos[t].dropna()
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
            df_metrics = pd.DataFrame(metrics).T
            st.table(df_metrics)

        with col2:
            st.subheader("🧮 Simulador de Carteira")
            st.write("Informe quantidades e preços médios para calcular retorno.")

            carteira = {}
            for t in tickers:
                qtd = st.number_input(f"Quantidade de {t}:", min_value=0, step=1, key=f"qtd_{t}")
                preco_medio = st.number_input(f"Preço médio de compra de {t} (R$):", min_value=0.0, format="%.2f", key=f"pm_{t}")
                carteira[t] = {"quantidade": qtd, "preco_medio": preco_medio}

            valor_total = 0.0
            valor_investido = 0.0
            resultado = []

            for t in tickers:
                qtd = carteira[t]["quantidade"]
                pm = carteira[t]["preco_medio"]
                serie = df_precos[t].dropna()
                preco_atual = float(serie.iloc[-1]) if not serie.empty else np.nan

                valor_posicao = qtd * preco_atual
                investimento = qtd * pm
                lucro_prejuizo = valor_posicao - investimento

                if isinstance(valor_posicao, (int, float, np.floating)) and not np.isnan(valor_posicao):
                    valor_total += valor_posicao

                if isinstance(investimento, (int, float, np.floating)) and not np.isnan(investimento):
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
