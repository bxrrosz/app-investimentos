import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# ---------------- 1. Período ----------------
periodo = st.selectbox(
    "Período de análise:",
    [("1 mês","1mo"), ("3 meses","3mo"), ("6 meses","6mo"),
     ("1 ano","1y"), ("2 anos","2y"), ("5 anos","5y")],
    index=2, format_func=lambda x: x[0])

# ---------------- 2. Tickers ----------------
ativos_str = st.text_input(
    "Digite os tickers separados por vírgula",
    value="",                                 # começa vazio
    placeholder="Ex: PETR4.SA, ITUB3.SA, AAPL, MSFT"
)
if not ativos_str.strip():
    st.info("Digite ao menos um ticker e pressione Enter.")
    st.stop()

tickers = [t.strip().upper() for t in ativos_str.split(",") if t.strip()]
st.write(f"Analisando **{', '.join(tickers)}** – período {periodo[0]}")

# ---------------- 3. Download preços -------
precos = {}
for tk in tickers:
    try:
        dados = yf.download(tk, period=periodo[1], progress=False)
        if dados.empty:
            st.warning(f"{tk} sem dados.")
            continue
        precos[tk] = dados["Close"]
    except Exception as e:
        st.error(f"{tk}: {e}")

if not precos:
    st.stop()

df_precos = pd.concat(precos, axis=1).droplevel(0, axis=1)

# ---------------- 4. USD‑BRL & conversão ---
try:
    usd_brl = yf.download("USDBRL=X", period=periodo[1], progress=False)["Close"]
    if usd_brl.empty:
        usd_brl = None
except Exception:
    usd_brl = None

if usd_brl is not None:
    usd_brl = usd_brl.fillna(method="ffill").fillna(method="bfill")
    for tk in df_precos.columns:
        if tk.endswith(".SA"):
            continue
        serie = df_precos[tk]
        taxa = usd_brl.reindex(serie.index, method="ffill")
        df_precos.loc[serie.index, tk] = serie * taxa
    st.success("Internacionais convertidos para BRL.")
else:
    st.info("Sem USD‑BRL disponível – internacionais ficam na moeda original.")

# preenche eventuais NaNs finais
df_precos = df_precos.fillna(method="ffill")

# ---------------- 5. Gráfico ----------------
st.subheader("📈 Preços ajustados")
fig = go.Figure()
for tk in df_precos.columns:
    fig.add_trace(go.Scatter(
        x=df_precos.index, y=df_precos[tk],
        mode="lines", name=tk, connectgaps=True))
fig.update_layout(template="plotly_white", hovermode="x unified",
                  xaxis_title="Data", yaxis_title="Preço (BRL)", height=500)
st.plotly_chart(fig, use_container_width=True)

# ---------------- 6. Métricas / Simulador ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Métricas financeiras")
    dados_metricas = {}
    for tk in df_precos.columns:
        s = df_precos[tk].dropna()
        if len(s) < 2: continue
        r = s.pct_change().dropna()
        dados_metricas[tk] = {
            "Ret Total": f"{(s.iloc[-1]/s.iloc[0]-1):.2%}",
            "Ret Anual": f"{(r.mean()*252):.2%}",
            "Vol": f"{(r.std()*np.sqrt(252)):.2%}",
            "Sharpe": f"{(r.mean()*252)/(r.std()*np.sqrt(252)):.2f}"
        }
    st.table(pd.DataFrame(dados_metricas).T)

with col2:
    st.subheader("🧮 Carteira")
    linhas, val_tot, inv_tot = [], 0.0, 0.0
    for tk in tickers:
        qtd = st.number_input(f"Qtd {tk}", 0, step=1, key=f"q_{tk}")
        pm  = st.number_input(f"PM {tk} (R$)", 0.0, format="%.2f", key=f"pm_{tk}")
        serie = df_precos[tk].dropna() if tk in df_precos else pd.Series(dtype=float)
        price = float(serie.iloc[-1]) if not serie.empty else np.nan
        pos, inv = qtd*price, qtd*pm
        val_tot += pos; inv_tot += inv
        linhas.append({"Ativo":tk,"Qtd":qtd,"Preço":price,
                       "Valor":pos,"Invest":inv,"Res":pos-inv})

    df_carteira = pd.DataFrame(linhas)
    if not df_carteira.empty:
        df_carteira["Res %"] = np.where(df_carteira["Invest"]!=0,
                                        df_carteira["Res"]/df_carteira["Invest"], 0)
        st.metric("Valor carteira", f"R$ {val_tot:,.2f}")
        st.metric("Retorno", f"{((val_tot/inv_tot)-1) if inv_tot else 0:.2%}")
        st.dataframe(df_carteira.style.format({
            "Preço":"R$ {:.2f}", "Valor":"R$ {:.2f}", "Invest":"R$ {:.2f}",
            "Res":"R$ {:.2f}", "Res %":"{:.2%}"}))
