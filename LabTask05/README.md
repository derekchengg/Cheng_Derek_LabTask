# Lab Task 05 — Clustering (Streamlit)

Add **DBSCAN** to the provided Streamlit app that already supports K-Means.

## What you need to do

1. Repeat the clustering setup already used for K-Means.
2. Add DBSCAN parameter UI in the sidebar (at least **epsilon**; also **min samples**).
3. Fit DBSCAN on the scaled feature matrix and color the map / PCA plot by cluster labels.

## Run the app

```bash
cd LabTask05
pip install -r requirements.txt
# also need: streamlit plotly (if not already installed)
streamlit run app.py
```

## Defaults

Parameter defaults match the in-class notebook walkthrough on scaled data:

| Parameter    | Default |
|-------------|---------|
| `eps`       | `0.22`  |
| `min_samples` | `5`   |

Noise points are labeled `-1` by DBSCAN and shown as their own category on the map.
