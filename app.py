import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# ----------------------------------------
# 1. Seleção de período e entrada de tickers
# ----------------------------------------
periodo = st.selectbox(
    "Período de análise:",
    options=[("1 mês", "1mo"), ("3 meses", "3mo"), ("6 meses", "6mo"), ("1 ano", "1y"), ("2 anos", "2y"), ("5 anos", "5y")],
    index=2,
    format_func=lambda x: x[0]
)

tickers_input = st.text_input("Tickers (sep. por vírgula)", placeholder="Ex: PETR4.SA, AAPL, TSLA")
if not tickers_input:
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

# ----------------------------------------
# 2. Download de preços + moeda
# ----------------------------------------
precos, moedas = {}, {}
for tk in tickers:
    try:
        dados = yf.download(tk, period=periodo[1], progress=False)
        if dados.empty:
            st.warning(f"{tk} sem dados.")
            continue
        precos[tk] = dados["Close"]
        try:
            moedas[tk] = yf.Ticker(tk).fast_info.get("currency", "BRL")
        except Exception:
            moedas[tk] = "BRL"
        st.success(f"{tk} carregado ({moedas[tk]})")
    except Exception as e:
        st.error(f"Erro {tk}: {e}")

if not precos:
    st.stop()

# ----------------------------------------
# 3. Conversão para BRL
# ----------------------------------------
fx_cache = {}
for tk, serie in precos.items():
    cur = moedas.get(tk, "BRL")
    if cur == "BRL":
        continue

    if cur not in fx_cache:
        pair = f"{cur}BRL=X"
        try:
            fx_raw = yf.download(pair, period="5d")
            fx_series = fx_raw["Close"] if "Close" in fx_raw else fx_raw.squeeze()
            fx_rate = fx_series.dropna().iloc[-1]
            fx_cache[cur] = float(fx_rate)
        except Exception:
            fx_cache[cur] = np.nan

    rate = fx_cache[cur]
    if pd.notna(rate):
        precos[tk] = serie * rate
    else:
        st.warning(f"Sem cotação {cur}/BRL – {tk} mantido na moeda original.")

precos_df = pd.concat(precos, axis=1)
if isinstance(precos_df.columns, pd.MultiIndex):
    precos_df.columns = precos_df.columns.droplevel(0)
precos_df = precos_df.loc[:, ~precos_df.columns.duplicated()]

st.write("💰 Preços exibidos em **BRL** quando possível.")

# ----------------------------------------
# 4. Gráfico de preços
# ----------------------------------------
st.subheader(f"📈 Preços ajustados – {periodo[0]}")
fig_price = go.Figure()
for tk in precos_df.columns:
    fig_price.add_trace(go.Scatter(x=precos_df.index, y=precos_df[tk], mode="lines", name=tk))
fig_price.update_layout(template="plotly_white", hovermode="x unified", height=450,
                        xaxis_title="Data", yaxis_title="Preço (BRL)")
st.plotly_chart(fig_price, use_container_width=True)

# ----------------------------------------
# 5. Projeção Monte Carlo
# ----------------------------------------

def mc_sim(serie: pd.Series, dias: int, n: int = 500):
    r = serie.pct_change().dropna()
    mu, sigma = r.mean(), r.std()
    steps = np.random.normal(mu, sigma, (dias, n)).cumsum(axis=0)
    return pd.DataFrame(np.exp(steps) * serie.iloc[-1])

with st.expander("🔮 Projeção de preço (experimental)"):
    dias_fwd = st.slider("Horizonte (dias)", 5, 252, 22)
    st.caption("*Estimativa baseada em histórico — não é recomendação.*")
    for tk in precos_df.columns:
        s = precos_df[tk].dropna()
        if s.empty:
            continue
        sim = mc_sim(s, dias_fwd)
        q10, q50, q90 = sim.iloc[-1].quantile([.1, .5, .9])
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{tk} 10% baixa", f"R$ {q10:.2f}")
        c2.metric("Mediana", f"R$ {q50:.2f}")
        c3.metric("10% alta", f"R$ {q90:.2f}")
        x_axis = range(1, dias_fwd+1)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=x_axis, y=[q90]*dias_fwd, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_fc.add_trace(go.Scatter(x=x_axis, y=[q10]*dias_fwd, fill="tonexty", fillcolor="rgba(65,105,225,0.2)",
                                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_fc.add_trace(go.Scatter(x=x_axis, y=[q50]*dias_fwd, line=dict(color="royalblue"), name="Mediana"))
        fig_fc.update_layout(template="plotly_white", height=250,
                             xaxis_title="Dias", yaxis_title="Preço projetado (BRL)",
                             margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_fc, use_container_width=True)

# ----------------------------------------
# 6. Métricas simples & Simulador
# ----------------------------------------
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("📊 Métricas")
    metrics = {}
    for tk in precos_df.columns:
        s = precos_df[tk].dropna()
        if len(s) < 2:
            continue
        r = s.pct_change().dropna()
        ret_tot = (s.iloc[-1]/s.iloc[0]) - 1
        ret_ano = r.mean()*252
        vol_ano = r.std()*np.sqrt(252)
        sharpe = ret_ano/vol_ano if vol_ano else np.nan
        metrics[tk] = {
            "Ret Total": f"{ret_tot:.2%}",
            "Ret Anual": f"{ret_ano:.2%}",
            "Vol": f"{vol_ano:.2%}",
            "Sharpe": f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
        }
    st.table(pd.DataFrame(metrics).T)

with col2:
    st.subheader("🧮 Carteira")
    carr = {}
    for tk in tickers:
        q = st.number_input(f"Qtd {tk}", 0, step=1, key=f"q{tk}")
        pm = st.number_input(f"PM {tk} (R$)", 0.0, format="%.2f", key=f"pm{tk}")
        carr[tk] = (q, pm)
    v_tot = inv_tot = 0.0
    rows = []
    for tk in tickers:
        q, pm = carr[tk]
        price = precos_df[tk].dropna().iloc[-1]
        pos = q*price
        inv = q*pm
        v_tot += pos if np.isfinite(pos) else 0
        inv_tot += inv if np.isfinite(inv) else 0
        rows.append({"Ativo": tk, "Qtd": q, "Preço": price, "Valor": pos, "Invest": inv, "Res": pos-inv})
    df_c = pd.DataFrame(rows)
    if not df_c.empty:
        df_c["Res %"] = np.where(df_c["Invest"]!=0, df_c["Res"] / df_c["







