"""
Agriculture Crop Pattern Mining
--------------------------------
Pipeline:
Preprocessing -> Apriori association rules -> NumPy K-Means clustering
-> PCA visualization -> IQR/robust-score anomaly flags -> CSV tables + PNG charts

Usage:
    python crop_pattern_mining.py
    python crop_pattern_mining.py my_dataset.csv

Expected columns (flexible names):
    Crop, Season, Region/State/District, Soil (optional), Rainfall (optional),
    Temperature (optional), Area (optional), Production (optional), Yield (optional)

If Yield is missing but Production and Area exist:
    Yield = Production / Area

Note:
This version avoids scikit-learn imports because Windows Code Integrity blocked
a compiled scikit-learn component on the user's computer. Clustering and PCA
are implemented with NumPy. Isolation Forest is replaced by a transparent
robust-score flag alongside per-crop IQR; these are not the same algorithm.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
DEFAULT_CSV = os.path.join(BASE_DIR, "data", "crop_data.csv")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.5
MIN_LIFT = 1.0


def make_synthetic_data(n=3000, seed=42):
    """Create a demo dataset so the pipeline can be demonstrated offline."""
    rng = np.random.default_rng(seed)
    crops = {
        "Rice": ("Kharif", "Clay", 1200, 27, 3.2),
        "Maize": ("Kharif", "Loamy", 800, 25, 2.8),
        "Cotton": ("Kharif", "Black", 700, 28, 1.6),
        "Wheat": ("Rabi", "Loamy", 450, 18, 3.5),
        "Mustard": ("Rabi", "Sandy", 400, 17, 1.4),
        "Chickpea": ("Rabi", "Black", 500, 20, 1.2),
        "Groundnut": ("Zaid", "Sandy", 600, 30, 1.8),
        "Watermelon": ("Zaid", "Sandy", 350, 32, 2.5),
    }
    regions = ["North", "South", "East", "West", "Central"]
    rows = []
    for _ in range(n):
        crop = rng.choice(list(crops))
        season, soil, rain, temp, yld = crops[crop]
        if rng.random() < 0.1:
            soil = rng.choice(["Clay", "Loamy", "Black", "Sandy"])
        rows.append({
            "Region": rng.choice(regions),
            "Season": season,
            "Crop": crop,
            "Soil_Type": soil,
            "Rainfall": rng.normal(rain, rain * 0.15),
            "Temperature": rng.normal(temp, 2),
            "Area": abs(rng.normal(120, 40)),
            "Yield": abs(rng.normal(yld, yld * 0.2)),
        })
    df = pd.DataFrame(rows)
    for col in ["Rainfall", "Temperature", "Soil_Type", "Season"]:
        df.loc[df.sample(frac=0.03, random_state=1).index, col] = np.nan
    df = pd.concat([df, df.sample(60, random_state=2)], ignore_index=True)
    df.loc[df.sample(20, random_state=3).index, "Crop"] = " rice "
    df.loc[df.sample(15, random_state=4).index, "Yield"] *= 8
    return df


def load_data():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif os.path.exists(DEFAULT_CSV):
        path = DEFAULT_CSV
    else:
        print("No CSV found -> using synthetic demo dataset.")
        return make_synthetic_data()
    print(f"Loading dataset: {path}")
    return pd.read_csv(path)


def standardize_columns(df):
    aliases = {
        "Crop": ["crop", "crop_name", "label", "crop_type"],
        "Season": ["season"],
        "Region": ["region", "state", "state_name", "district", "district_name", "location"],
        "Soil_Type": ["soil", "soil_type", "soiltype"],
        "Rainfall": ["rainfall", "rain", "annual_rainfall"],
        "Temperature": ["temperature", "temp"],
        "Area": ["area", "area_ha", "area_harvested"],
        "Production": ["production", "production_tonnes"],
        "Yield": ["yield", "yield_per_hectare", "crop_yield"],
    }
    lower = {str(c).lower().strip(): c for c in df.columns}
    rename = {}
    for std, names in aliases.items():
        for name in names:
            if name in lower and lower[name] not in rename:
                rename[lower[name]] = std
                break
    df = df.rename(columns=rename)
    if "Crop" not in df.columns:
        raise ValueError("Dataset needs a crop column (e.g. 'Crop').")
    if "Yield" not in df.columns and {"Production", "Area"} <= set(df.columns):
        df["Yield"] = df["Production"] / df["Area"].replace(0, np.nan)
    return df


def preprocess(df):
    print("\n=== PREPROCESSING ===")
    print(f"Initial shape: {df.shape}")
    missing = df.isna().sum()
    print(f"Missing values per column:\n{missing[missing > 0]}")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (df[col].astype("string").str.strip()
                   .str.replace(r"\s+", " ", regex=True).str.title())

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed: {duplicate_count}")

    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_value = df[col].median()
                df[col] = df[col].fillna(0 if pd.isna(median_value) else median_value)
            else:
                modes = df[col].mode(dropna=True)
                df[col] = df[col].fillna(modes.iloc[0] if not modes.empty else "Unknown")

    for col in ["Rainfall", "Area", "Production", "Yield"]:
        if col in df.columns:
            df = df[df[col] >= 0]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    print(f"Final shape: {df.shape}")
    df.to_csv(os.path.join(OUT_DIR, "cleaned_data.csv"), index=False)
    return df


def discretize(df):
    d = pd.DataFrame(index=df.index)
    for col in ["Crop", "Season", "Region", "Soil_Type"]:
        if col in df.columns:
            d[col] = col + "=" + df[col].astype(str)
    for col in ["Rainfall", "Temperature", "Yield"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            try:
                bins = pd.qcut(values, 3, labels=["Low", "Medium", "High"],
                               duplicates="drop")
                d[col] = col + "=" + bins.astype(str)
            except ValueError:
                d[col] = col + "=Medium"
    return d


def run_apriori(df):
    print("\n=== ASSOCIATION RULES (APRIORI) ===")
    items = pd.get_dummies(discretize(df), prefix="", prefix_sep="").astype(bool)
    freq = apriori(items, min_support=MIN_SUPPORT, use_colnames=True)
    if freq.empty:
        print("No frequent itemsets - lower MIN_SUPPORT.")
        return freq, pd.DataFrame()

    freq["itemsets_str"] = freq["itemsets"].apply(lambda s: ", ".join(sorted(s)))
    freq.drop(columns="itemsets").sort_values("support", ascending=False).to_csv(
        os.path.join(OUT_DIR, "frequent_itemsets.csv"), index=False
    )

    rules = association_rules(freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[rules["lift"] >= MIN_LIFT].copy()
    if rules.empty:
        print("No rules passed thresholds - lower MIN_CONFIDENCE / MIN_SUPPORT.")
        return freq, rules

    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    rules[cols].to_csv(os.path.join(OUT_DIR, "association_rules.csv"), index=False)

    print(f"Frequent itemsets: {len(freq)} | Rules: {len(rules)}")
    print("\nTop 10 rules by lift:")
    print(rules[cols].head(10).round(3).to_string(index=False))

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"],
                     cmap="viridis", alpha=0.7, s=40)
    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules: Support vs Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rules_scatter.png"), dpi=150)
    plt.close()

    top = rules.head(10).copy()
    top["rule"] = top["antecedents"] + " => " + top["consequents"]
    plt.figure(figsize=(11, 5))
    sns.barplot(data=top, y="rule", x="lift", color="seagreen")
    plt.title("Top Association Rules by Lift")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top_rules.png"), dpi=150)
    plt.close()
    return freq, rules


def standardize_array(values):
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    return (values - means) / stds


def numpy_kmeans(X, k, seed=42, max_iter=300):
    """Basic K-Means using NumPy; returns integer cluster labels and centroids."""
    rng = np.random.default_rng(seed)
    n_rows = X.shape[0]
    centers = X[rng.choice(n_rows, size=k, replace=False)].copy()
    labels = np.zeros(n_rows, dtype=int)

    for _ in range(max_iter):
        distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = X[new_labels == cluster_id]
            if len(members):
                new_centers[cluster_id] = members.mean(axis=0)
            else:
                new_centers[cluster_id] = X[rng.integers(0, n_rows)]
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers
    return labels, centers


def silhouette_mean(X, labels):
    """Compute mean silhouette score without scikit-learn."""
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(X):
        return -1.0
    distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i, label in enumerate(labels):
        same = np.where(labels == label)[0]
        same = same[same != i]
        a = distances[i, same].mean() if len(same) else 0.0
        other_means = []
        for other in unique:
            if other == label:
                continue
            other_idx = np.where(labels == other)[0]
            if len(other_idx):
                other_means.append(distances[i, other_idx].mean())
        b = min(other_means) if other_means else 0.0
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def pca_2d(X):
    """Two-component PCA projection using NumPy SVD."""
    centered = X - X.mean(axis=0)
    if centered.shape[1] == 1:
        return np.column_stack([centered[:, 0], np.zeros(len(centered))])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:min(2, len(vt))]
    projected = centered @ components.T
    if projected.shape[1] == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(len(projected))])
    return projected[:, :2]


def run_clustering(df):
    print("\n=== CLUSTERING (NUMPY K-MEANS) ===")
    num_cols = [c for c in ["Rainfall", "Temperature", "Area", "Yield"] if c in df.columns]
    if len(num_cols) < 2:
        print("Not enough numeric columns for clustering - skipped.")
        return df

    profile = df.groupby("Crop")[num_cols].mean()
    if len(profile) < 3:
        print("Need at least 3 crop profiles for clustering - skipped.")
        return df

    X = standardize_array(profile.to_numpy(dtype=float))
    k_values = range(2, min(8, len(profile)))
    scores = {}
    for k in k_values:
        labels, _ = numpy_kmeans(X, k, seed=42)
        scores[k] = silhouette_mean(X, labels)
    if not scores:
        print("Not enough crop profiles to select a cluster count - skipped.")
        return df

    best_k = max(scores, key=scores.get)
    print(f"Silhouette scores: { {k: round(v, 3) for k, v in scores.items()} }")
    print(f"Selected k = {best_k} (highest computed silhouette score)")
    plt.figure(figsize=(6, 4))
    plt.plot(list(scores), list(scores.values()), marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Choosing k")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "silhouette.png"), dpi=150)
    plt.close()

    labels, _ = numpy_kmeans(X, best_k, seed=42)
    profile["Cluster"] = labels
    profile.round(2).sort_values("Cluster").to_csv(
        os.path.join(OUT_DIR, "crop_clusters.csv")
    )
    print("\nCrop groups:")
    for cluster_id, group in profile.groupby("Cluster"):
        print(f"  Cluster {cluster_id}: {', '.join(group.index.astype(str))}")
    print("\nCluster averages:")
    print(profile.groupby("Cluster")[num_cols].mean().round(2))

    coords = pca_2d(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=profile["Cluster"],
                    palette="Set1", s=150)
    for (x, y), name in zip(coords, profile.index):
        plt.annotate(str(name), (x, y), xytext=(5, 5), textcoords="offset points")
    plt.title("Crop Clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "crop_clusters.png"), dpi=150)
    plt.close()

    return df.merge(profile[["Cluster"]], left_on="Crop", right_index=True, how="left")


def run_outliers(df):
    print("\n=== OUTLIER DETECTION ===")
    if "Yield" not in df.columns:
        print("No Yield column - skipped.")
        return df

    def iqr_flag(series):
        q1, q3 = series.quantile([0.25, 0.75])
        spread = q3 - q1
        return (series < q1 - 1.5 * spread) | (series > q3 + 1.5 * spread)

    df["Outlier_IQR"] = df.groupby("Crop")["Yield"].transform(iqr_flag)

    # Robust modified z-score over available numeric variables; this is a
    # transparent alternative flag, not Isolation Forest.
    numeric_cols = [c for c in ["Rainfall", "Temperature", "Area", "Yield"] if c in df.columns]
    values = df[numeric_cols].astype(float)
    medians = values.median()
    deviations = (values - medians).abs()
    mads = deviations.median().replace(0, np.nan)
    modified_z = 0.6745 * deviations.div(mads).replace([np.inf, -np.inf], np.nan)
    df["Outlier_RobustScore"] = modified_z.fillna(0).gt(3.5).any(axis=1)

    print(f"IQR outliers (per crop): {int(df['Outlier_IQR'].sum())}")
    print(f"Robust-score outliers: {int(df['Outlier_RobustScore'].sum())}")
    flagged = df[df["Outlier_IQR"] | df["Outlier_RobustScore"]]
    flagged.to_csv(os.path.join(OUT_DIR, "outliers.csv"), index=False)

    plt.figure(figsize=(11, 5))
    sns.boxplot(data=df, x="Crop", y="Yield", color="lightgreen")
    plt.xticks(rotation=45)
    plt.title("Yield Distribution by Crop")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "yield_boxplot.png"), dpi=150)
    plt.close()
    return df


def run_visuals(df):
    print("\n=== VISUALIZATIONS ===")
    if "Season" in df.columns:
        table = pd.crosstab(df["Crop"], df["Season"])
        table.to_csv(os.path.join(OUT_DIR, "crop_by_season.csv"))
        plt.figure(figsize=(8, 6))
        sns.heatmap(table, annot=True, fmt="d", cmap="YlGn")
        plt.title("Crop vs Season (record counts)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "crop_season_heatmap.png"), dpi=150)
        plt.close()

    if "Region" in df.columns:
        table = pd.crosstab(df["Crop"], df["Region"])
        table.to_csv(os.path.join(OUT_DIR, "crop_by_region.csv"))
        plt.figure(figsize=(9, 6))
        sns.heatmap(table, annot=True, fmt="d", cmap="YlOrBr")
        plt.title("Crop vs Region (record counts)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "crop_region_heatmap.png"), dpi=150)
        plt.close()

    numeric_cols = [c for c in ["Rainfall", "Temperature", "Area", "Yield"] if c in df.columns]
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(6, 5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation: Environmental Factors vs Yield")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "correlation.png"), dpi=150)
        plt.close()

    plt.figure(figsize=(8, 4))
    df["Crop"].value_counts().plot(kind="bar", color="seagreen")
    plt.title("Crop Frequency")
    plt.ylabel("Records")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "crop_frequency.png"), dpi=150)
    plt.close()

    if "Yield" in df.columns and "Soil_Type" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Soil_Type", y="Yield",
                    hue="Crop" if df["Crop"].nunique() <= 8 else None)
        plt.title("Average Yield by Soil Type")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "yield_by_soil.png"), dpi=150)
        plt.close()


def main():
    df = load_data()
    df = standardize_columns(df)
    df = preprocess(df)
    run_apriori(df)
    df = run_clustering(df)
    df = run_outliers(df)
    run_visuals(df)
    df.to_csv(os.path.join(OUT_DIR, "final_results.csv"), index=False)
    print(f"\nDone. Tables and charts are saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
