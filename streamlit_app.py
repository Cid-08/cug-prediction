import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.api import Holt
import altair as alt

# … votre code de chargement / forecast ici …

chart_type = st.sidebar.radio(
    "Choisissez le type de graphique",
    ("Deux graphiques côte à côte", "Un seul graphique (ordre)", "Altair dual-axis")
)

if chart_type == "Deux graphiques côte à côte":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Population vs Consommation")
        st.line_chart(df_plot.set_index("année")[["population", "consommation"]])
    with col2:
        st.subheader("CUG (m³/hab)")
        st.line_chart(df_plot.set_index("année")[["CUG"]])

elif chart_type == "Un seul graphique (ordre)":
    st.subheader("Population, Consommation et CUG")
    df2 = df_plot.set_index("année")[["CUG", "population", "consommation"]]
    st.line_chart(df2)

else:  # Altair dual-axis
    st.subheader("Altair: dual-axis chart")
    dfm = df_plot.melt("année", var_name="metric", value_name="value")
    base = alt.Chart(dfm).encode(x=alt.X("année:O"))
    pop_cons = (base
                .transform_filter(alt.FieldOneOfPredicate(field="metric",
                                                         oneOf=["population", "consommation"]))
                .mark_line()
                .encode(y=alt.Y("value:Q", axis=alt.Axis(title="Pop/Cons")),
                        color="metric:N"))
    cug = (base
           .transform_filter(alt.FieldEqualPredicate(field="metric", equal="CUG"))
           .mark_line(strokeDash=[5,5])
           .encode(y=alt.Y("value:Q", axis=alt.Axis(title="CUG"), axisY=alt.Y2()),
                   color=alt.value("steelblue")))
    st.altair_chart(alt.layer(pop_cons, cug).resolve_scale(y="independent"),
                    use_container_width=True)
