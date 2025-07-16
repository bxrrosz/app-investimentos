import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# 1. período
periodo = st.selectbox("Período:",
    [("1 mês","1mo"),("3 meses","3mo"),("6 meses","6mo"),
     ("1 ano","1y"),("2 anos","2y"),("5 anos","5y")],
    index=2, format_func=lambda x: x[0])

# 2. tickers
txt = st.text_input("Tickers (vírgula)", "", placeholder="Ex: PETR4.SA, ITUB3.SA, AAPL, MSFT")
if not txt.strip(): st.stop()
tickers = list(dict.fromkeys([t.strip().upper() for t in txt.split(",") if t.strip()]))

# 3. preços
precos={} 
for tk in tickers:
    df = yf.download(tk, period=periodo[1], progress=False)
    if not df.empty: precos[tk]=df["Close"]
df_precos = pd.concat(precos, axis=1)
if isinstance(df_precos.columns, pd.MultiIndex):
    df_precos.columns = df_precos.columns.droplevel(0)

# 4. USD‑BRL preenchido em todo range
try:
    usd = yf.download("USDBRL=X", period=periodo[1], progress=False)["Close"]
    usd = usd.reindex(df_precos.index)                 # cria mesmo intervalo
    usd = usd.fillna(method="bfill").fillna(method="ffill")  # agora sem buracos
except Exception:
    usd = None

intl = [c for c in df_precos.columns if not c.endswith(".SA")]
if usd is not None and intl:
    df_precos[intl] = df_precos[intl].mul(usd, axis=0)

# preenche NaNs restantes p/ gráfico bonito
df_precos = df_precos.fillna(method="ffill")

# 5. gráfico
st.subheader("📈 Preços ajustados")
fig=go.Figure()
for c in df_precos:
    fig.add_trace(go.Scatter(x=df_precos.index,y=df_precos[c],mode="lines",
                             name=c,connectgaps=True))
fig.update_layout(template="plotly_white",hovermode="x unified",
                  xaxis_title="Data",yaxis_title="Preço (BRL)",height=500)
st.plotly_chart(fig,use_container_width=True)

# 6. métricas
col1,col2=st.columns([2,1])
with col1:
    tbl={}
    for c in df_precos:
        s=df_precos[c]
        if s.isna().all(): continue
        r=s.pct_change().dropna()
        tbl[c]={"Ret Tot":f"{(s[-1]/s[0]-1):.2%}",
                "Ret Anual":f"{(r.mean()*252):.2%}",
                "Vol":f"{(r.std()*np.sqrt(252)):.2%}",
                "Sharpe":f"{(r.mean()*252)/(r.std()*np.sqrt(252)):.2f}"}
    st.table(pd.DataFrame(tbl).T)

# 7. simulador
with col2:
    st.subheader("🧮 Carteira")
    rows,total,invest=[],0.0,0.0
    for tk in tickers:
        qtd=st.number_input(f"Qtd {tk}",0,step=1,key=f"q_{tk}")
        pm =st.number_input(f"PM {tk} (R$)",0.0,format="%.2f",key=f"pm_{tk}")
        price=df_precos[tk].iloc[-1] if tk in df_precos else np.nan
        val,inv=qtd*price,qtd*pm
        total+=val;invest+=inv
        rows.append({"Ativo":tk,"Qtd":qtd,"Preço":price,
                     "Valor":val,"Invest":inv,"Res":val-inv})
    dfc=pd.DataFrame(rows)
    if not dfc.empty:
        dfc["Res %"]=np.where(dfc["Invest"]!=0,dfc["Res"]/dfc["Invest"],0)
        st.metric("Valor",f"R$ {total:,.2f}")
        st.metric("Retorno",f"{((total/invest)-1) if invest else 0:.2%}")
        st.dataframe(dfc.style.format({"Preço":"R$ {:.2f}","Valor":"R$ {:.2f}",
                                       "Invest":"R$ {:.2f}","Res":"R$ {:.2f}",
                                       "Res %":"{:.2%}"}))
