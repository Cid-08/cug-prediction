import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.api import Holt
import altair as alt

# Configuration de la page
st.set_page_config(page_title="Prévision CUG – SEN’EAU", layout="wide")
st.title("Prédiction de la CUG de l'eau à Dakar")

# 1) Fonction pour charger les données et calculer df_plot
@st.cache_data
def make_forecasts(filepath):
    # Chargement
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Année': 'année',
        'Populations': 'population',
        'Consommation en eau m3/an': 'consommation_totale_m3'
    })
    # Calcul CUG
    df['CUG'] = df['consommation_totale_m3'] / df['population']

    # Prévision population via Holt
    pop_train = pd.Series(
        df[df.année <= 2018]['population'].values,
        index=pd.to_datetime(df[df.année <= 2018]['année'].astype(str), format='%Y')
    )
    pop_train.index.freq = 'YS'
    pop_model = Holt(pop_train).fit(optimized=True)
    pop_fc = pop_model.forecast(2030 - 2018).round().astype(int)
    years = pop_fc.index.year
    df_pop = pd.DataFrame({'année': years, 'population': pop_fc.values})

    # Prévision CUG via Holt
    cug_train = pd.Series(
        df['CUG'].values,
        index=pd.to_datetime(df['année'].astype(str), format='%Y')
    )
    cug_train.index.freq = 'YS'
    cug_model = Holt(cug_train).fit(optimized=True)
    cug_fc = cug_model.forecast(2030 - 1996).round(2)[:len(years)]
    df_cug = pd.DataFrame({'année': cug_fc.index.year, 'CUG': cug_fc.values})

    # Fusion et calcul consommation totale
    df_fc = pd.merge(df_pop, df_cug, on='année')
    df_fc['consommation'] = (df_fc['population'] * df_fc['CUG']).round().astype(int)

    # Historique formaté
    df_hist = df[['année', 'population', 'CUG', 'consommation_totale_m3']].rename(
        columns={'consommation_totale_m3': 'consommation'}
    )

    # Concaténation historique + prévisions
    return pd.concat([df_hist[df_hist.année >= 1997], df_fc], ignore_index=True)

# 2) Génération de df_plot
df_plot = make_forecasts("données consommations eau.xlsx")

# 3) Choix du type de graphique
chart_type = st.sidebar.radio(
    "Choisissez le type de graphique",
    (
        "Deux graphiques côte à côte",
        "Un seul graphique (ordre forcé)",
        "Altair dual-axis"
    )
)

# 4) Affichage conditionnel
if chart_type == "Deux graphiques côte à côte":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Population vs Consommation")
        st.line_chart(df_plot.set_index("année")[["population", "consommation"]], height=350)
    with col2:
        st.subheader("CUG (m³/hab)")
        st.line_chart(df_plot.set_index("année")[["CUG"]], height=350)

elif chart_type == "Un seul graphique (ordre forcé)":
    st.subheader("Population, Consommation et CUG (ordre personnalisé)")
    df2 = df_plot.set_index("année")[["CUG", "population", "consommation"]]
    st.line_chart(df2, height=450)

else:  # Altair dual-axis
    st.subheader("Altair: Population/Consommation vs CUG (double axe)")
    # Mise en forme long
    dfm = df_plot.melt("année", var_name="metric", value_name="value")

    base = alt.Chart(dfm).encode(
        x=alt.X("année:O", axis=alt.Axis(title="Année"))
    )

    # Couche population + consommation (axe gauche)
    pop_cons = (
        base.transform_filter(
            alt.FieldOneOfPredicate(field="metric", oneOf=["population", "consommation"])
        )
        .mark_line()
        .encode(
            y=alt.Y("value:Q", axis=alt.Axis(title="Population / Consommation", orient="left")),
            color=alt.Color("metric:N", legend=None),
        )
    )

    # Couche CUG (axe droit)
    cug_layer = (
        base.transform_filter(
            alt.FieldEqualPredicate(field="metric", equal="CUG")
        )
        .mark_line(strokeDash=[5, 5], size=2)
        .encode(
            y=alt.Y("value:Q", axis=alt.Axis(title="CUG (m³/hab)", orient="right")),
            color=alt.value("steelblue"),
        )
    )

    chart = alt.layer(pop_cons, cug_layer).resolve_scale(y="independent")
    st.altair_chart(chart, use_container_width=True)

# 5) Bouton d’export CSV
csv = df_plot.to_csv(index=False).encode("utf-8")
st.download_button("Exporter les données", csv, file_name="previsions_cug.csv")
