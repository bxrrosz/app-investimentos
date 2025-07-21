import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import requests
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="App Investimentos", page_icon="💰", layout="wide")
st.markdown("# 💰 Analisador Simples de Investimentos")

@st.cache_data(ttl=3600)
def get_fear_and_greed_index():
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url).json()
        value = int(response['data'][0]['value'])
        return value
    except Exception:
        return None

def plot_fear_greed_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Índice de Medo e Ganância (Crypto)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25], 'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"},
            ],
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    return fig

# --- Restante do código igual até chegar na seção da carteira ---

# ...

if ativos_str:
    tickers = [t.strip().upper() for t in ativos_str.split(",") if t.strip()]
    st.write(f"Analisando: **{', '.join(tickers)}** no período de {periodo[0]}")

    # (Download e tratamento dos dados)

    # (Processamento métricas básicas e avançadas e simulador carteira...)

    # Ao chegar no layout do gráfico da carteira + índice:

    st.markdown("---")

    col_carteira, col_lateral = st.columns([3,1])

    with col_carteira:
        # Simulador e métricas aqui, conforme antes
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

            valor_total += valor_posicao if not np.isnan(valor_posicao) else 0
            valor_investido += investimento if not np.isnan(investimento) else 0

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

    with col_lateral:
        st.subheader("⚖️ Pesos na Carteira (%)")
        pesos = {}
        soma_pesos = 0.0
        for t in tickers:
            peso = st.number_input(f"Peso {t} (%):", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"peso_{t}")
            pesos[t] = peso
            soma_pesos += peso

        if soma_pesos == 100:
            retornos_diarios = df_precos.pct_change().dropna()
            pesos_array = np.array([pesos[t] / 100 for t in df_precos.columns])
            retorno_carteira_diario = retornos_diarios.dot(pesos_array)
            retorno_total = ((1 + retorno_carteira_diario).prod()) - 1
            volatilidade_ano = retorno_carteira_diario.std() * np.sqrt(252)

            benchmark_name = tickers[0]
            benchmark_retornos = retornos_diarios[benchmark_name]

            X = benchmark_retornos.values.reshape(-1, 1)
            y = retorno_carteira_diario.values
            modelo = LinearRegression().fit(X, y)
            alpha = modelo.intercept_
            beta = modelo.coef_[0]

            st.markdown("### 📈 Resultados da Carteira com Pesos")
            st.write(f"- **Retorno Total:** {retorno_total:.2%}")
            st.write(f"- **Volatilidade Anualizada:** {volatilidade_ano:.2%}")
            st.write(f"- **Alpha:** {alpha:.4f}")
            st.write(f"- **Beta:** {beta:.4f}")
        else:
            st.warning(f"A soma dos pesos é {soma_pesos:.2f}%. Ela deve ser exatamente 100%. Ajuste os pesos.")

        fg_value = get_fear_and_greed_index()

        # Layout lado a lado gráfico carteira e índice de medo e ganância
        if soma_pesos == 100:
            fig_carteira = go.Figure()
            fig_carteira.add_trace(go.Scatter(
                x=retorno_carteira_diario.index,
                y=(1 + retorno_carteira_diario).cumprod(),
                mode='lines',
                name='Carteira'
            ))

            for t in df_precos.columns:
                retorno_ativo = (1 + df_precos[t].pct_change().dropna()).cumprod()
                fig_carteira.add_trace(go.Scatter(
                    x=retorno_ativo.index,
                    y=retorno_ativo.values,
                    mode='lines',
                    name=t,
                    line=dict(dash='dot'),
                    opacity=0.6
                ))

            fig_carteira.update_layout(
                title="Evolução da Carteira e dos Ativos",
                xaxis_title="Data",
                yaxis_title="Valor Acumulado (Índice)",
                legend_title_text="Ativos / Carteira",
                template="plotly_white",
                height=350,
                margin=dict(t=40, b=40, l=40, r=40)
            )

            col_graf, col_fg = st.columns([2, 1])

            with col_graf:
                st.plotly_chart(fig_carteira, use_container_width=True)

            with col_fg:
                if fg_value is not None:
                    st.plotly_chart(plot_fear_greed_gauge(fg_value), use_container_width=True, height=350)
                else:
                    st.warning("Não foi possível obter o índice de medo e ganância (Crypto).")
        else:
            if fg_value is not None:
                st.plotly_chart(plot_fear_greed_gauge(fg_value), use_container_width=True, height=350)
            else:
                st.warning("Não foi possível obter o índice de medo e ganância (Crypto).")
