import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Análise Simples de Investimentos 📈")

ativos = st.text_input("Digite os tickers (ex: PETR4.SA,VALE3.SA)", "PETR4.SA,VALE3.SA")

if ativos:
    lista = [ticker.strip().upper() for ticker in ativos.split(",")]
    dados = yf.download(lista, period="6mo")["Adj Close"]
    st.line_chart(dados)
