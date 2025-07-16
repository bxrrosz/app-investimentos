import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# ------------------------ 1. Entrada ------------------------
periodo = st.selectbox(
    "Período de análise:",
    [("1 mês", "1mo"), ("3 meses", "3mo"), ("6 meses", "6mo"), ("1 ano", "1y"), ("2 anos", "2y"), ("5 anos", "5y")],
    index=2,
    format_func=lambda x: x[0],
)

tickers_in = st.text_input("Tickers (vírgula)", "PETR4.SA, AAPL")
if not tickers_in:
    st.stop()

tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]

# --------------------- 2. Download + moeda ------------------
precos, moedas = {}, {}
for tk in tickers:
    df = yf.download(tk, period=periodo[1], progress=False)
    if df.empty:
        st.warning(f"{tk} sem dados.")
        continue
    precos[tk] = df["Close"]
    try:
        moedas[tk] = yf.Ticker(tk).fast_info.get("currency", "BRL")
    except Exception:
        moedas[tk] = "BRL"
    st.success(f"{tk} ({moedas[tk]}) carregado")

if not precos:
    st.stop()

# --------------------- 3. Conversão BRL ---------------------
fx_cache = {}
for tk, serie in precos.items():
    cur = moedas.get(tk, "BRL")
    if cur == "BRL":
        continue
    if cur not in fx_cache:
        pair = f"{cur}BRL=X"
        fx = yf.download(pair, period="5d", progress=False)["Close"].dropna()
        fx_cache[cur] = float(fx.iloc[-1]) if not fx.empty else np.nan
    rate = fx_cache[cur]
    if pd.notna(rate):
        precos[tk] = serie * rate
    else:
        st.warning(f"Sem fx {cur}/BRL – {tk} mantido na moeda original")

precos_df = pd.concat(precos, axis=1)
if isinstance(precos_df.columns, pd.MultiIndex):
    precos_df.columns = precos_df.columns.droplevel(0)
precos_df = precos_df.loc[:, ~precos_df.columns.duplicated()]

# ----------------- 4. Gráfico de preços ---------------------
st.subheader(f"📈 Preços ajustados – {periodo[0]}")
fig_p = go.Figure()
for tk in precos_df.columns:
    fig_p.add_trace(go.Scatter(x=precos_df.index, y=precos_df[tk], mode="lines", name=tk))
fig_p.update_layout(template="plotly_white", hovermode="x unified", height=450,
                    xaxis_title="Data", yaxis_title="Preço (BRL)")
st.plotly_chart(fig_p, use_container_width=True)

# ------------------ 5. Projeção Monte Carlo -----------------

def mc_sim(serie: pd.Series, dias: int, n: int = 500):
    r = serie.pct_change().dropna()
    mu, sigma = r.mean(), r.std()
    steps = np.random.normal(mu, sigma, (dias, n)).cumsum(axis=0)
    return np.exp(steps) * serie.iloc[-1]

with st.expander("🔮 Projeção de preço (experimental)"):
    dias_fwd = st.slider("Horizonte (dias)", 5, 252, 22)
    st.caption("*Estimativa histórica — não é recomendação.*")
    for tk in precos_df.columns:
        s = precos_df[tk].dropna()
        if s.empty:
            continue
        sims = mc_sim(s, dias_fwd)
        q10, q50, q90 = np.quantile(sims[-1], [0.1, 0.5, 0.9])
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{tk} 10% baixa", f"R$ {q10:.2f}")
        c2.metric("Mediana", f"R$ {q50:.2f}")
        c3.metric("10% alta", f"R$ {q90:.2f}")

        x = list(range(1, dias_fwd + 1))
        q10_list = [q10] * dias_fwd
        q90_list = [q90] * dias_fwd
        q50_list = [q50] * dias_fwd

        fig_fc = go.Figure()
        # faixa inferior
        fig_fc.add_trace(go.Scatter(x=x, y=q10_list, line=dict(width=0), name="10% baixa",
                                    mode="lines", hoverinfo="skip", showlegend=False))
        # faixa superior + preenchimento
        fig_fc.add_trace(go.Scatter(x=x, y=q90_list, fill="tonexty", fillcolor="rgba(65,105,225,0.2)",
                                    line=dict(width=0), name="10% alta", hoverinfo="skip", showlegend=False))
        # mediana
        fig_fc.add_trace(go.Scatter(x=x, y=q50_list, line=dict(color="royalblue"), name="Mediana"))

        fig_fc.update_layout(template="plotly_white", height=250,
                             margin=dict(l=0, r=0, t=30, b=0),
                             xaxis_title="Dias", yaxis_title="Preço proj. (BRL)")
        st.plotly_chart(fig_fc, use_container_width=True)

# --------------- 6. Métricas + Simulador --------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Métricas")
    tbl = {}
    for tk in precos_df.columns:
        s = precos_df[tk].dropna()
        if len(s) < 2:
            continue
        r = s.pct_change().dropna()
        ret_tot = (s.iloc[-1] / s.iloc[0]) - 1
        ret_ano = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ret_ano / vol if vol else np.nan
        tbl[tk] = {
            "Ret Total": f"{ret_tot:.2%}",
            "Ret Anual": f"{ret_ano:.2%}",
            "Vol": f"{vol:.2%}",
            "Sharpe": f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A",
        }
    st.table(pd.DataFrame(tbl).T)

with col2:
    st.subheader("🧮 Carteira")
    rows = []
    v_tot = inv_tot = 0.0
    for tk in tickers:
        qtd = st.number_input(f"Qtd {tk}", 0, step=1, key=f"q_{tk}")
        pm = st.number_input(f"PM {tk} (R$)", 0.0, format="%.2f", key=f"pm_{tk}")
        price = precos_df[tk].dropna().iloc[-1]
        val = qtd * price
        inv = qtd * pm
        v_tot += val
        inv_tot += inv
        rows.append({"Ativo": tk, "Qtd": qtd, "Preço": price, "Valor": val, "Invest": inv, "Res": val - inv})

    df_c = pd.DataFrame(rows)
    if not df_c.empty:
        df_c["Res %"] = np.where(df_c["Invest"] != 0, df_c["Res"] / df_c["Invest"], 0)
        st.metric("Valor carteira", f"R$ {v_tot:,.2f}")
        st.metric("Retorno", f"{((v_tot / inv_tot) - 1) if inv_tot else 0:.2%}")
        st.dataframe(df_c.style.format({
            "Preço": "R$ {:.2f}", "Valor": "R$ {:.2f}", "Invest": "R$ {:.2f}",
            "Res": "R$ {:.2f}", "Res %": "{:.2%}",
        }))








