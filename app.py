import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# ---------------- 1. Período ----------------
periodo = st.selectbox(
    "Selecione o período de análise:",
    [("1 mês", "1mo"), ("3 meses", "3mo"), ("6 meses", "6mo"),
     ("1 ano", "1y"), ("2 anos", "2y"), ("5 anos", "5y")],
    index=2, format_func=lambda x: x[0]
)

# ---------------- 2. Tickers ----------------
ativos_str = st.text_input(
    "Digite os tickers separados por vírgula",
    "PETR4.SA, ITUB3.SA, AAPL, MSFT"
)
if not ativos_str:
    st.stop()

tickers = [t.strip().upper() for t in ativos_str.split(",") if t.strip()]
st.write(f"Analisando: **{', '.join(tickers)}** – período {periodo[0]}")

# ---------------- 3. Download preços -------
precos = {}
for tk in tickers:
    try:
        dados = yf.download(tk, period=periodo[1], progress=False)
        if dados.empty:
            st.warning(f"⚠️ {tk} sem dados.")
            continue
        precos[tk] = dados["Close"]
        st.success(f"Dados de {tk} carregados.")
    except Exception as e:
        st.error(f"Erro em {tk}: {e}")

if not precos:
    st.stop()

df_precos = pd.concat(precos, axis=1)
df_precos.columns = df_precos.columns.droplevel(0)

# ---------------- 4. USD‑BRL & conversão ---
try:
    usd_brl_raw = yf.download("USDBRL=X", period=periodo[1], progress=False)["Close"]
    usd_brl = usd_brl_raw.copy() if not usd_brl_raw.empty else None
    if usd_brl is not None:
        usd_brl.index = pd.to_datetime(usd_brl.index)
        usd_brl = usd_brl.fillna(method="ffill").fillna(method="bfill")
except Exception as e:
    usd_brl = None
    st.warning(f"Falha USD‑BRL: {e}")

if usd_brl is not None:
    usd_brl_filled = usd_brl
    for tk in df_precos.columns:
        if not tk.endswith(".SA"):                       # internacional
            serie = df_precos[tk]
            serie.index = pd.to_datetime(serie.index)
            taxa_alinh = usd_brl_filled.reindex(serie.index, method="ffill")
            df_precos[tk] = serie * taxa_alinh
    st.info("Conversão p/ BRL concluída (USD‑BRL).")
else:
    st.info("Sem USD‑BRL – estrangeiros mantidos na moeda original.")

# garantir preenchimento final para gráfico contínuo
df_precos = df_precos.fillna(method="ffill")

# ---------------- 5. Gráfico ----------------
st.subheader(f"📈 Preços ajustados ({periodo[0]})")
fig = go.Figure()
for tk in df_precos.columns:
    fig.add_trace(go.Scatter(
        x=df_precos.index, y=df_precos[tk],
        mode="lines", name=tk, connectgaps=True
    ))
fig.update_layout(template="plotly_white", hovermode="x unified",
                  xaxis_title="Data", yaxis_title="Preço (BRL)",
                  legend_title_text="Ativos", height=500)
st.plotly_chart(fig, use_container_width=True)

# ---------------- 6. Métricas ---------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Métricas financeiras")
    tbl = {}
    for tk in df_precos.columns:
        s = df_precos[tk].dropna()
        if len(s) < 2:
            continue
        r = s.pct_change().dropna()
        ret_tot = (s.iloc[-1]/s.iloc[0]) - 1
        ret_ano = r.mean()*252
        vol_ano = r.std()*np.sqrt(252)
        sharpe = ret_ano/vol_ano if vol_ano else np.nan
        tbl[tk] = {"Ret Total":f"{ret_tot:.2%}",
                   "Ret Anual":f"{ret_ano:.2%}",
                   "Vol":f"{vol_ano:.2%}",
                   "Sharpe":f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"}
    st.table(pd.DataFrame(tbl).T)

# ---------------- 7. Simulador --------------
with col2:
    st.subheader("🧮 Carteira")
    rows, val_tot, inv_tot = [], 0.0, 0.0
    for tk in tickers:
        qtd = st.number_input(f"Qtd {tk}", 0, step=1, key=f"q_{tk}")
        pm  = st.number_input(f"PM {tk} (R$)", 0.0, format="%.2f", key=f"pm_{tk}")
        serie = df_precos[tk].dropna() if tk in df_precos else pd.Series(dtype=float)
        price = float(serie.iloc[-1]) if not serie.empty else np.nan
        pos, inv = qtd*price, qtd*pm
        val_tot += pos
        inv_tot += inv
        rows.append({"Ativo":tk, "Qtd":qtd, "Preço":price,
                     "Valor":pos, "Invest":inv, "Res":pos-inv})
    df_c = pd.DataFrame(rows)
    if not df_c.empty:
        df_c["Res %"] = np.where(df_c["Invest"]!=0, df_c["Res"]/df_c["Invest"], 0)
        st.metric("Valor carteira", f"R$ {val_tot:,.2f}")
        st.metric("Retorno", f"{((val_tot/inv_tot)-1) if inv_tot else 0:.2%}")
        st.dataframe(df_c.style.format({
            "Preço":"R$ {:.2f}", "Valor":"R$ {:.2f}", "Invest":"R$ {:.2f}",
            "Res":"R$ {:.2f}", "Res %":"{:.2%}"}))

