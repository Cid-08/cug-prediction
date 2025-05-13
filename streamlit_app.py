import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.api import Holt

st.set_page_config(layout="wide")
st.title("Prévision CUG – Hackathon SEN’EAU")

@st.cache_data
def make_forecasts(filepath):
    # 1. Chargement et préparation
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Année': 'année',
        'Populations': 'population',
        'Consommation en eau m3/an': 'consommation_totale_m3'
    })
    df['CUG'] = df['consommation_totale_m3'] / df['population']

    # 2. Holt population (1997–2018 → 2019–2030)
    pop_train = pd.Series(
        df[df.année <= 2018]['population'].values,
        index=pd.to_datetime(df[df.année <= 2018]['année'].astype(str), format='%Y')
    )
    pop_train.index.freq = 'YS'
    pop_model = Holt(pop_train).fit(optimized=True)
    pop_fc = pop_model.forecast(2030 - 2018).round().astype(int)
    years = pop_fc.index.year
    df_pop = pd.DataFrame({'année': years, 'population': pop_fc.values})

    # 3. Holt CUG (1997–2018 → 2019–2030)
    cug_train = pd.Series(
        df['CUG'].values,
        index=pd.to_datetime(df['année'].astype(str), format='%Y')
    )
    cug_train.index.freq = 'YS'
    cug_model = Holt(cug_train).fit(optimized=True)
    cug_fc = cug_model.forecast(2030 - 1996).round(2)[:len(years)]
    df_cug = pd.DataFrame({'année': cug_fc.index.year, 'CUG': cug_fc.values})

    # 4. Fusion et calcul consommation totale
    df_fc = pd.merge(df_pop, df_cug, on='année')
    df_fc['consommation'] = (df_fc['population'] * df_fc['CUG']).round().astype(int)

    # 5. Historique formaté
    df_hist = df[['année', 'population', 'CUG', 'consommation_totale_m3']].rename(
        columns={'consommation_totale_m3': 'consommation'}
    )

    # 6. Concaténer historique + prévisions
    return pd.concat([df_hist[df_hist.année >= 1997], df_fc], ignore_index=True)

# --- Exécution de l’app ---
df_plot = make_forecasts("données consommations eau.xlsx")

st.subheader("Évolution CUG, population et consommation (1997–2030)")
# Après avoir produit `df_plot`
col1, col2 = st.columns(2)

with col1:
    st.subheader("Population vs Consommation")
    st.line_chart(
        df_plot.set_index("année")[["population", "consommation"]],
        height=350
    )

with col2:
    st.subheader("CUG (m³/habitant)")
    st.line_chart(
        df_plot.set_index("année")[["CUG"]],
        height=350
    )

csv = df_plot.to_csv(index=False).encode('utf-8')
st.download_button("Exporter les données", csv, file_name="previsions_cug.csv")
