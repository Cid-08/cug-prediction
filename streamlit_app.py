
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.api import Holt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("Prévision CUG – Hackathon SEN’EAU")

@st.cache_data
def make_forecasts(filepath):
    # 1. Chargement et préparation
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
      'Année':'année',
      'Populations':'population',
      'Consommation en eau m3/an':'consommation_totale_m3'
    })
    df['CUG'] = df['consommation_totale_m3'] / df['population']

    # 2. Holt population
    train_pop = pd.Series(df[df.année<=2018]['population'].values,
                          index=pd.to_datetime(df[df.année<=2018]['année'].astype(str), format='%Y'))
    train_pop.index.freq = 'YS'
    pop_model = Holt(train_pop).fit(optimized=True)
    pop_fc = pop_model.forecast(2030-2018).round().astype(int)
    years = pop_fc.index.year
    df_pop = pd.DataFrame({'année':years,'population_pred':pop_fc.values})

    # 3. Holt CUG
    train_cug = pd.Series(df['CUG'].values,
                          index=pd.to_datetime(df['année'].astype(str), format='%Y'))
    train_cug.index.freq = 'YS'
    cug_model = Holt(train_cug).fit(optimized=True)
    cug_fc = cug_model.forecast(2030-1996).round(2)[:len(years)]
    df_cug = pd.DataFrame({'année':cug_fc.index.year,'CUG_pred':cug_fc.values})

    # 4. Fusion et calcul conso
    df_fc = pd.merge(df_pop, df_cug, on='année')
    df_fc['consommation_pred_m3'] = (df_fc['population_pred'] * df_fc['CUG_pred']).round().astype(int)
    return df, df_fc

# Écrivez le fichier
with open("streamlit_app.py","w") as f:
    f.write(code)
print("✅ streamlit_app.py créé.")
