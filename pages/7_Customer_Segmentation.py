import streamlit as st
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px

st.title("👥 Customer Segmentation")

df = pd.read_csv("data/telecom.csv")

X = df.select_dtypes(include="number")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

pca = PCA(n_components=2)

points = pca.fit_transform(X)

plot = pd.DataFrame({

    "PC1": points[:, 0],

    "PC2": points[:, 1],

    "Cluster": df["Cluster"]

})

fig = px.scatter(
    plot,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="Customer Segments"
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(df.groupby("Cluster").mean(numeric_only=True))