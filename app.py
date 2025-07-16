import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

# ----------------------------------------
# 1. Escolha de período e digitação de ticker
# ----------------------------------------
periodo = st.selectbox(
    "Selecione o período de análise:",
    options=[("1 mês", "1mo"), ("3 meses", "3mo"), ("6 meses", "6mo"), ("1 ano", "1y"), ("2 anos", "2y"), ("5 anos", "5y")],
    index=2,
    format_func=lambda x: x[0]
)

tickers_input = st.text_input("Tickers separados por vírgula (ex: PETR4.SA, AAPL, TSLA)")
if not tickers_input:
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

# ----------------------------------------
# 2. Download de preços + detecção de moeda
# ----------------------------------------
precos, moedas = {}, {}
for t in tickers:
    try:
        dados = yf.download(t, period=periodo[1], progress=False)
        if not dados.empty:
            precos[t] = dados['Close']
            try:
                moedas[t] = yf.Ticker(t).fast_info.get('currency', 'BRL')
            except Exception:
                moedas[t] = 'BRL'
            st.success(f"{t} carregado ({moedas[t]})")
        else:
            st.warning(f"{t} sem dados.")
    except Exception as e:
        st.error(f"Erro em {t}: {e}")

if not precos:
    st.stop()

# ----------------------------------------
# 3. Conversão para BRL se necessário
# ----------------------------------------
fx_cache = {}
for t, serie in precos.items():
    cur = moedas.get(t, 'BRL')
    if cur != 'BRL':
        if cur not in fx_cache:
            pair = f"{cur}BRL=X"
            fx = yf.download(pair, period='1d')['Close']
            fx_cache[cur] = fx.iloc[-1] if not fx.empty else np.nan
        rate = fx_cache[cur]
        if np.isfinite(rate):
            precos[t] = serie * rate
        else:
            st.warning(f"Não foi possível converter {t} de {cur} para BRL.")

precos_df = pd.concat(precos, axis=1)
if isinstance(precos_df.columns, pd.MultiIndex):
    precos_df.columns = precos_df.columns.droplevel(0)
precos_df = precos_df.loc[:, ~precos_df.columns.duplicated()]

st.write("💰 Preços *em BRL* (cotação diária para moedas estrangeiras).")

# ----------------------------------------
# 4. Gráfico Plotly de preços
# ----------------------------------------
st.subheader(f"📈 Preços ajustados – {periodo[0]}")
fig_price = go.Figure()
for t in precos_df.columns:
    fig_price.add_trace(go.Scatter(x=precos_df.index, y=precos_df[t], mode='lines', name=t))
fig_price.update_layout(template='plotly_white', hovermode='x unified', height=450,
                        xaxis_title='Data', yaxis_title='Preço (BRL)')
st.plotly_chart(fig_price, use_container_width=True)

# ----------------------------------------
# 5. Projeção Monte Carlo (visual limpo)
# ----------------------------------------

def simular_caminhos(serie: pd.Series, dias: int, n_sim: int = 500) -> pd.DataFrame:
    r = serie.pct_change().dropna()
    mu, sigma = r.mean(), r.std()
    passos = np.random.normal(mu, sigma, (dias, n_sim)).cumsum(axis=0)
    return pd.DataFrame(np.exp(passos) * serie.iloc[-1])

with st.expander("🔮  Previsão de preço (experimental)"):
    dias_forecast = st.slider("Horizonte (dias de pregão)", 5, 252, 22)
    st.caption("*Estimativa baseada em dados históricos – não é recomendação.*")
    for t in precos_df.columns:
        serie = precos_df[t].dropna()
        if serie.empty:
            continue
        sim = simular_caminhos(serie, dias_forecast)
        q10, q50, q90 = sim.iloc[-1].quantile([0.1, 0.5, 0.9])

        col_l, col_m, col_h = st.columns(3)
        col_l.metric(f"{t} – 10% baixa", f"R$ {q10:.2f}")
        col_m.metric("Mediana", f"R$ {q50:.2f}")
        col_h.metric("10% alta", f"R$ {q90:.2f}")

        x_axis = range(1, dias_forecast + 1)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=x_axis, y=[q90]*dias_forecast, line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_fc.add_trace(go.Scatter(x=x_axis, y=[q10]*dias_forecast, fill='tonexty', fillcolor='rgba(65,105,225,0.2)',
                                    line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_fc.add_trace(go.Scatter(x=x_axis, y=[q50]*dias_forecast, line=dict(color='royalblue'), name='Mediana'))
        fig_fc.update_layout(template='plotly_white', height=250,
                             xaxis_title='Dias à frente', yaxis_title='Preço projetado (BRL)',
                             margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_fc, use_container_width=True)

# ----------------------------------------
# 6. Métricas x Simulador
# ----------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Métricas simples")
    metrics = {}
    for t in precos_df.columns:
        s = precos_df[t].dropna()
        if len(s) < 2:
            continue
        r = s.pct_change().dropna()
        ret_total = (s.iloc[-1] / s.iloc[0]) - 1
        ret_ano = r.mean() * 252
        vol_ano = r.std() * np.sqrt(252)
        sharpe = ret_ano / vol_ano if vol_ano else np.nan
        metrics[t] = {
            'Retorno Total': f"{ret_total:.2%}",
            'Retorno Anual': f"{ret_ano:.2%}",
            'Volatilidade': f"{vol_ano:.2%}",
            'Sharpe': f"{sharpe:.2f}" if not np.isnan(sharpe) else 'N/A'
        }
    st.table(pd.DataFrame(metrics).T)

with col2:
    st.subheader("🧮 Simulador de Carteira")
    carteira = {}
    for t in tickers:
        qtd = st.number_input(f"Qtd {t}", 0, step=1, key=f"q_{t}")
        pm = st.number_input(f"PM {t} (R$)", 0.0, format="%.2f", key=f"pm_{t}")
        carteira[t] = (qtd, pm)

    valor_total = invest_tot = 0.0
    rows = []
    for t in tickers:
        qtd, pm = carteira[t]
        atual = precos_df[t].dropna().iloc[-1]
        pos = qtd * atual
        inv = qtd * pm
        lucro = pos - inv
        valor_total += pos if np.isfinite(pos) else 0
        invest_tot += inv if np.isfinite(inv) else 0
        rows.append({'Ativo': t, 'Qtd': qtd, 'Preço Atual': atual, 'Valor': pos, 'Invest': inv, 'Resultado': lucro})

    df_cart = pd.DataFrame(rows)
    if not df_cart.empty:
        df_cart['Resultado %'] = np.where(df_cart['Invest'] != 0, df_cart['Resultado'] / df_cart['Invest'], 0)
        st.metric("Valor da carteira", f"R$ {valor_total:,.2f}")
        st.metric("Retorno", f"{((valor_total / invest_tot) - 1) if invest_tot else 0:.2%}")
        st.dataframe(df_cart.style.format({
            'Preço Atual': 'R$ {:.2f}', 'Valor': 'R$ {:.2f}', 'Invest': 'R$ {:.2f}',
            'Resultado': 'R$ {:.2f}', 'Resultado %': '{:.2%}'
        }))







