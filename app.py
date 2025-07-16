import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# Seleção do período de análise
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

# Input dos tickers
ativos_str = st.text_input(
    "Digite os tickers da bolsa separados por vírgula",
    placeholder="Ex: PETR4.SA, ITUB3.SA, B3SA3.SA"
)

# Função para obter taxa de câmbio USD-BRL diária (fechamento) para conversão
@st.cache_data(ttl=3600)
def get_usd_brl_rates(period):
    dados = yf.download("USDBRL=X", period=period, progress=False)
    if dados.empty:
        return None
    return dados["Close"]

usd_brl_rates = None
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
        df_precos.columns = df_precos.columns.droplevel(0)

        # Obter taxa USD-BRL para conversão dos ativos não BRL
        usd_brl_rates = get_usd_brl_rates(periodo[1])
        if usd_brl_rates is None or usd_brl_rates.empty:
            st.warning("Não foi possível obter taxa USD-BRL para conversão.")

        # Converter preços em USD para BRL quando aplicável
        for t in df_precos.columns:
            if not t.endswith(".SA"):  # assumindo que ativos BR têm .SA
                if usd_brl_rates is not None:
                    # Alinhar índices para multiplicação
                    df_precos[t] = df_precos[t].reindex(usd_brl_rates.index).fillna(method="ffill")
                    df_precos[t] = df_precos[t] * usd_brl_rates
                else:
                    st.warning(f"Não foi possível converter {t} para BRL, dados de câmbio faltando.")

        # Gráfico interativo com Plotly (com linhas contínuas)
        st.subheader(f"📈 Preços ajustados – {periodo[0]}")
        fig_p = go.Figure()
        for tk in df_precos.columns:
            serie_plot = df_precos[tk].copy()
            fig_p.add_trace(go.Scatter(
                x=serie_plot.index,
                y=serie_plot.values,
                mode="lines",
                name=tk,
                connectgaps=True  # evita quebras visuais quando há fins de semana/NaNs
            ))
        fig_p.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=450,
            xaxis_title="Data",
            yaxis_title="Preço (BRL)"
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Layout em colunas para métricas e simulador
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
            rows = []
            v_tot = 0.0
            inv_tot = 0.0

            for tk in tickers:
                qtd = st.number_input(f"Qtd {tk}", 0, step=1, key=f"q_{tk}")
                pm = st.number_input(f"PM {tk} (R$)", 0.0, format="%.2f", key=f"pm_{tk}")

                # preço atual só se o ticker realmente está em precos_df
                if tk in df_precos.columns:
                    serie_price = df_precos[tk].dropna()
                    price = float(serie_price.iloc[-1]) if not serie_price.empty else np.nan
                else:
                    st.warning(f"{tk}: dados não carregados – verifique ticker.")
                    price = np.nan

                val = qtd * price if np.isfinite(price) else 0
                inv = qtd * pm
                v_tot += val
                inv_tot += inv
                rows.append({"Ativo": tk, "Qtd": qtd, "Preço": price, "Valor": val, "Invest": inv, "Res": val - inv})

            df_c = pd.DataFrame(rows)
            if not df_c.empty:
                df_c["Res %"] = np.where(df_c["Invest"] != 0, df_c["Res"] / df_c["Invest"], 0)
            else:
                df_c["Res %"] = []

            st.write(f"**Valor total da carteira:** R$ {v_tot:,.2f}")
            st.write(f"**Valor total investido:** R$ {inv_tot:,.2f}")
            st.write(f"**Retorno total da carteira:** {(v_tot / inv_tot - 1) if inv_tot != 0 else 0:.2%}")

            st.dataframe(df_c.style.format({
                "Preço": "R$ {:,.2f}",
                "Valor": "R$ {:,.2f}",
                "Invest": "R$ {:,.2f}",
                "Res": "R$ {:,.2f}",
                "Res %": "{:.2%}",
            }))

        # --- Projeção simples (exemplo básico de previsão linear) ---
        st.subheader("🔮 Projeção Simples de Preço Futuro (próximos 30 dias)")
        dias_fwd = 30

        fig_fc = go.Figure()
        for t in df_precos.columns:
            serie = df_precos[t].dropna()
            if len(serie) > 10:
                y = serie.values
                x = np.arange(len(y))
                coef = np.polyfit(x, y, 1)  # regressão linear simples
                proj_x = np.arange(len(y), len(y) + dias_fwd)
                proj_y = coef[0] * proj_x + coef[1]

                q10 = np.percentile(proj_y, 10)
                q90 = np.percentile(proj_y, 90)
                q50 = np.percentile(proj_y, 50)

                x_proj = list(range(1, dias_fwd + 1))
                q10_list = [q10] * dias_fwd
                q90_list = [q90] * dias_fwd
                q50_list = [q50] * dias_fwd

                # faixa inferior
                fig_fc.add_trace(go.Scatter(x=x_proj, y=q10_list, line=dict(width=0), name="10% baixa",
                                            mode="lines", hoverinfo="skip", showlegend=False))
                # faixa superior + preenchimento
                fig_fc.add_trace(go.Scatter(x=x_proj, y=q90_list, fill="tonexty", fillcolor="rgba(65,105,225,0.2)",
                                            line=dict(width=0), name="10% alta", hoverinfo="skip", showlegend=False))
                # mediana
                fig_fc.add_trace(go.Scatter(x=x_proj, y=q50_list, line=dict(color="royalblue"), name="Mediana"))

        fig_fc.update_layout(template="plotly_white", height=250,
                             margin=dict(l=0, r=0, t=30, b=0),
                             xaxis_title="Dias", yaxis_title="Preço proj. (BRL)")
        st.plotly_chart(fig_fc, use_container_width=True)



