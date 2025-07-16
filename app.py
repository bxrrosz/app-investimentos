import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np 
import plotly.graph_objects as go

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# 1. Período -------------------------------------------------
periodo = st.selectbox(
    "Período de análise:",
    [("1 mês","1mo"),("3 meses","3mo"),("6 meses","6mo"),
     ("1 ano","1y"),("2 anos","2y"),("5 anos","5y")],
    index=2, format_func=lambda x: x[0])

# 2. Tickers -------------------------------------------------
ativos_str = st.text_input("Tickers (vírgula)", "")       # vazio por padrão
if not ativos_str.strip():
    st.info("Digite ao menos um ticker para iniciar.")
    st.stop()

tickers = [t.strip().upper() for t in ativos_str.split(",") if t.strip()]
st.write(f"Analisando **{', '.join(tickers)}** – {periodo[0]}")

# 3. Download preços ----------------------------------------
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

# 4. USD‑BRL + conversão ------------------------------------
try:
    usd = yf.download("USDBRL=X", period=periodo[1], progress=False)["Close"]
    usd = usd.fillna(method="ffill").fillna(method="bfill") if not usd.empty else None
except Exception:
    usd = None

if usd is not None:
    for tk in df_precos.columns:
        if tk.endswith(".SA"):    # brasileiro – nada a fazer
            continue
        serie = df_precos[tk].copy()
        if serie.isna().all():    # sem dados válidos
            continue
        aligned_rate = usd.reindex(serie.index, method="ffill")
        produto = serie * aligned_rate
        df_precos.loc[produto.index, tk] = produto
    st.info("Internacionais convertidos p/ BRL (USD‑BRL).")
else:
    st.info("USD‑BRL indisponível – internacionais permanecem na moeda original.")

df_precos = df_precos.fillna(method="ffill")

# 5. Gráfico -------------------------------------------------
st.subheader("📈 Preços ajustados")
fig = go.Figure()
for tk in df_precos.columns:
    fig.add_trace(go.Scatter(x=df_precos.index, y=df_precos[tk],
                             mode="lines", name=tk, connectgaps=True))
fig.update_layout(template="plotly_white", hovermode="x unified",
                  xaxis_title="Data", yaxis_title="Preço (BRL)", height=500)
st.plotly_chart(fig, use_container_width=True)

# 6. Métricas simples ---------------------------------------
st.subheader("📊 Métricas")
metrics = {}
for tk in df_precos.columns:
    s = df_precos[tk].dropna()
    if len(s) < 2: continue
    r = s.pct_change().dropna()
    metrics[tk] = {
        "Ret Total": f"{(s.iloc[-1]/s.iloc[0]-1):.2%}",
        "Ret Anual": f"{(r.mean()*252):.2%}",
        "Vol": f"{(r.std()*np.sqrt(252)):.2%}",
        "Sharpe": f"{(r.mean()*252)/(r.std()*np.sqrt(252)):.2f}"
    }
st.table(pd.DataFrame(metrics).T)

# 7. Simulador de carteira ----------------------------------
st.subheader("🧮 Carteira")
rows, total, invest = [], 0.0, 0.0
for tk in tickers:
    qtd = st.number_input(f"Qtd {tk}", 0, step=1, key=f"q_{tk}")
    pm  = st.number_input(f"PM {tk} (R$)", 0.0, format="%.2f", key=f"pm_{tk}")
    price = df_precos[tk].dropna().iloc[-1] if tk in df_precos and not df_precos[tk].dropna().empty else np.nan
    pos, inv = qtd*price, qtd*pm
    total += pos; invest += inv
    rows.append({"Ativo":tk,"Qtd":qtd,"Preço":price,"Valor":pos,"Invest":inv,"Res":pos-inv})

df_c = pd.DataFrame(rows)
if not df_c.empty:
    df_c["Res %"] = np.where(df_c["Invest"]!=0, df_c["Res"]/df_c["Invest"], 0)
    st.metric("Valor carteira", f"R$ {total:,.2f}")
    st.metric("Retorno", f"{((total/invest)-1) if invest else 0:.2%}")
    st.dataframe(df_c.style.format({
        "Preço":"R$ {:.2f}","Valor":"R$ {:.2f}","Invest":"R$ {:.2f}",
        "Res":"R$ {:.2f}","Res %":"{:.2%}"}))
