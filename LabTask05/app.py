import streamlit as st
import json
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("Business Location Explorer")
st.write("Learning how to use streamlit")


@st.cache_data
def load_data(path="business_locations.geojson"):
    """Load point GeoJSON into a flat DataFrame."""
    with open(path) as f:
        geojson = json.load(f)
    rows = []
    # this function can read geojson files that have only points.
    for feat in geojson["features"]:
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        rows.append({**props, "lon": lon, "lat": lat})
    return pd.DataFrame(rows)


df = load_data()

with st.expander("Look at Data:"):
    st.dataframe(df.head(20))
    st.write(f"{len(df)} locations, {df['Neighborhood'].nunique()} neighborhoods.")

st.sidebar.header("1. Select Features")

NUMERIC_COLS = [
    "Floor_Area_sqm",
    "Daily_Foot_Traffic",
    "Community_Impact_Score",
    "Annual_Revenue_k",
]
selected_features = st.sidebar.multiselect(
    "Features to be used in models",
    options=NUMERIC_COLS,
    default=NUMERIC_COLS,
)

if len(selected_features) < 2:
    st.warning("Pick at least two features to continue")
    st.stop()

X = df[selected_features].to_numpy()
X_scaled = StandardScaler().fit_transform(X)

st.sidebar.header("2. Clustering")
algo = st.sidebar.selectbox("Algorithm", ["Kmeans", "DBSCAN"])

if algo == "Kmeans":
    k = st.sidebar.slider("Number of Clusters", 2, 10, 4)
elif algo == "DBSCAN":
    eps = st.sidebar.slider(
        "Epsilon (eps)",
        min_value=0.05,
        max_value=2.0,
        value=0.22,
        step=0.01,
        help="Maximum distance between two samples for one to be considered in the neighborhood of the other.",
    )
    min_samples = st.sidebar.slider(
        "Min samples",
        min_value=2,
        max_value=20,
        value=5,
        help="Minimum number of samples in a neighborhood for a point to be a core point.",
    )

if algo == "Kmeans":
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
elif algo == "DBSCAN":
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

# Catch error if no labels — do not continue
if len(labels) == 0:
    st.warning("there is no clustering labels")
    st.stop()

df = df.copy()
df["cluster"] = pd.Categorical(labels.astype(str))

# DBSCAN labels noise as -1; report real clusters separately
n_noise = int((labels == -1).sum()) if algo == "DBSCAN" else 0
n_clusters_found = len(set(labels) - {-1}) if algo == "DBSCAN" else df["cluster"].nunique()

col1, col2 = st.columns(2)
col1.metric("Number of Clusters", n_clusters_found)
if algo == "DBSCAN":
    col2.metric("Noise points", n_noise)

map_tab, dr_tab = st.tabs(["Map", "Dimensionality Reduction"])

with map_tab:
    st.write("MAP")
    fig = px.scatter_map(
        df,
        lat="lat",
        lon="lon",
        color="cluster",
        zoom=10,
        height=550,
        map_style="carto-darkmatter",
    )
    st.plotly_chart(fig, width="stretch")

with dr_tab:
    reducer = PCA(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    df["dim_1"] = embedding[:, 0]
    df["dim_2"] = embedding[:, 1]

    fig_dr = px.scatter(
        df,
        x="dim_1",
        y="dim_2",
        color="cluster",
        height=550,
    )
    st.plotly_chart(fig_dr, width="stretch")
